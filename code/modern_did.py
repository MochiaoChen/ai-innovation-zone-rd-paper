"""Heterogeneity-robust staggered difference-in-differences.

The estimands follow the equivalence in Wooldridge (2025, Empirical Economics):
with linear parallel trends and no anticipation, the cohort-by-event-time
coefficients from the extended two-way fixed effects regression are the same
object as the imputation estimator of Borusyak, Jaravel and Spiess (2024,
Review of Economic Studies). Sun and Abraham (2021) is the aggregation of
those cells. Gardner's two-step estimator is the same imputation with
covariates partialled out of untreated observations.

Nothing in this module invents results for the A-share panel. Callers pass
arrays. ``code/05c_modern_did.py`` reads ``data/clean/analysis_data.csv``
when that file exists.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr


def _as_codes(values):
    """Map an array of labels to contiguous codes, preserving order of appearance."""
    values = np.asarray(values)
    _, codes = np.unique(values, return_inverse=True)
    return codes.astype(np.int32), int(codes.max()) + 1 if len(codes) else 0


def _event_time(year, g):
    year = np.asarray(year)
    g = np.asarray(g)
    rel = np.full(len(year), np.nan)
    treated_unit = g > 0
    rel[treated_unit] = year[treated_unit] - g[treated_unit]
    return rel


def twfe_factors(y, firm, year, mask, n_iter=500, tol=1e-12):
    """Fit firm and year effects on the rows selected by ``mask``.

    The year effects are centered at zero on the years that appear in ``mask``.
    Firms with no selected row keep a missing intercept and must not be used
    for imputation.
    """
    y = np.asarray(y, dtype=float)
    firm = np.asarray(firm, dtype=np.int32)
    year = np.asarray(year, dtype=np.int32)
    n_firms = int(firm.max()) + 1
    n_years = int(year.max()) + 1
    alpha = np.zeros(n_firms)
    lam = np.zeros(n_years)
    seen_firm = np.zeros(n_firms, dtype=bool)
    f = firm[mask]
    t = year[mask]
    yy = y[mask]
    seen_firm[f] = True
    for _ in range(n_iter):
        previous = alpha.copy()
        resid = yy - lam[t]
        sum_f = np.zeros(n_firms)
        cnt_f = np.zeros(n_firms)
        np.add.at(sum_f, f, resid)
        np.add.at(cnt_f, f, 1.0)
        ok = cnt_f > 0
        alpha[ok] = sum_f[ok] / cnt_f[ok]
        resid = yy - alpha[f]
        sum_t = np.zeros(n_years)
        cnt_t = np.zeros(n_years)
        np.add.at(sum_t, t, resid)
        np.add.at(cnt_t, t, 1.0)
        ok = cnt_t > 0
        lam[ok] = sum_t[ok] / cnt_t[ok]
        lam[ok] -= lam[ok].mean()
        if np.max(np.abs(alpha - previous)) < tol:
            break
    alpha[~seen_firm] = np.nan
    return alpha, lam


def imputation_effects(y, firm, year, g, covariates=None):
    """Borusyak-Jaravel-Spiess imputation on a staggered design.

    Parameters
    ----------
    y, firm, year, g :
        Outcome, unit id, calendar time, and first treatment time.
        Never-treated units have ``g == 0``. ``g`` must be on the same
        scale as ``year``.
    covariates :
        Optional array of shape (n, p), already lagged if the columns are
        outcomes of the policy. They are partialled out on untreated rows
        only. Contemporaneous post-treatment covariates are not a valid input.

    Returns
    -------
    dict with unit-level effects on treated rows (``tau``), an overall
    observation-weighted ATT, and cell means by cohort and by event time.
    """
    y = np.asarray(y, dtype=float)
    firm_codes, _ = _as_codes(firm)
    year_codes, _ = _as_codes(year)
    g = np.asarray(g)
    year_raw = np.asarray(year)
    untreated = (g == 0) | (year_raw < g)
    treated = (g > 0) & (year_raw >= g)
    if covariates is None:
        alpha, lam = twfe_factors(y, firm_codes, year_codes, untreated)
        fitted = alpha[firm_codes] + lam[year_codes]
    else:
        fitted = _impute_with_covariates(
            y, firm_codes, year_codes, untreated, np.asarray(covariates, dtype=float)
        )
    ok = treated & np.isfinite(fitted) & np.isfinite(y)
    tau = np.full(len(y), np.nan)
    tau[ok] = y[ok] - fitted[ok]
    rel = _event_time(year_raw, g)
    return {
        "tau": tau,
        "treated": ok,
        "rel": rel,
        "att": float(np.mean(tau[ok])) if ok.any() else np.nan,
        "by_cohort": _cell_means(tau, g, ok),
        "by_event": _cell_means(tau, rel, ok),
        "by_cohort_event": _two_way_means(tau, g, rel, ok),
    }


def _cell_means(tau, key, mask):
    out = {}
    keys = np.asarray(key)
    for value in np.unique(keys[mask]):
        cell = mask & (keys == value)
        out[float(value) if not isinstance(value, str) else value] = {
            "estimate": float(np.mean(tau[cell])),
            "n": int(cell.sum()),
        }
    return out


def _two_way_means(tau, g, rel, mask):
    out = {}
    gg = np.asarray(g)
    rr = np.asarray(rel)
    for cohort in np.unique(gg[mask]):
        for event in np.unique(rr[mask & (gg == cohort)]):
            cell = mask & (gg == cohort) & (rr == event)
            out[(int(cohort), int(event))] = {
                "estimate": float(np.mean(tau[cell])),
                "n": int(cell.sum()),
            }
    return out


def _impute_with_covariates(y, firm, year, untreated, x):
    """First stage on untreated rows: firm FE, year FE, and covariates."""
    n = len(y)
    n_firms = int(firm.max()) + 1
    n_years = int(year.max()) + 1
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    # Drop rows whose covariates are missing. Those units can still receive
    # an intercept-only imputation only if every covariate row is finite.
    finite = np.isfinite(y) & np.isfinite(x).all(axis=1)
    use = untreated & finite
    columns = [np.ones(n, dtype=bool)]
    # Intercept, firm dummies with one firm omitted, year dummies with one year omitted.
    for i in range(1, n_firms):
        columns.append(firm == i)
    for t in range(1, n_years):
        columns.append(year == t)
    design_u = np.column_stack([c[use].astype(float) for c in columns] + [x[use]])
    coef, _, rank, _ = linalg.lstsq(design_u, y[use])
    if rank < design_u.shape[1]:
        # Fall back to an unpenalised least-squares solution on the column space.
        coef = linalg.lstsq(design_u, y[use])[0]
    design = np.column_stack([c.astype(float) for c in columns] + [np.where(np.isfinite(x), x, 0.0)])
    fitted = design @ coef
    fitted[~finite] = np.nan
    return fitted


def etwfe_coefficients(y, firm, year, g, event_window=(-4, 4)):
    """Cohort-by-event-time coefficients from the extended TWFE regression.

    The omitted event time is -1. Never-treated units (``g == 0``) identify
    the year effects. Coefficients for event time >= 0 are cohort-time ATTs
    under no anticipation and linear parallel trends.
    """
    y = np.asarray(y, dtype=float)
    firm_codes, n_firms = _as_codes(firm)
    year_codes, n_years = _as_codes(year)
    g = np.asarray(g)
    year_raw = np.asarray(year)
    rel = _event_time(year_raw, g)
    lo, hi = event_window
    cohorts = [int(v) for v in np.unique(g) if v > 0]
    cells = []
    for cohort in cohorts:
        present = rel[g == cohort]
        present = present[np.isfinite(present)]
        for event in range(lo, hi + 1):
            if event == -1:
                continue
            if np.any(present == event):
                cells.append((cohort, event))
    n = len(y)
    # Intercept, then firm dummies with one firm omitted, then year dummies
    # with one year omitted. Omitting both without an intercept drops a
    # dimension the two-way model needs.
    n_fe = 1 + (n_firms - 1) + (n_years - 1)
    n_cols = n_fe + len(cells)
    rows, cols, data = [], [], []
    for i in range(n):
        rows.append(i)
        cols.append(0)
        data.append(1.0)
        if firm_codes[i] > 0:
            rows.append(i)
            cols.append(int(firm_codes[i]))
            data.append(1.0)
        if year_codes[i] > 0:
            rows.append(i)
            cols.append(n_firms + int(year_codes[i] - 1))
            data.append(1.0)
    cell_index = {cell: n_fe + j for j, cell in enumerate(cells)}
    for i in range(n):
        if g[i] == 0 or not np.isfinite(rel[i]):
            continue
        key = (int(g[i]), int(rel[i]))
        if key in cell_index:
            rows.append(i)
            cols.append(cell_index[key])
            data.append(1.0)
    matrix = csc_matrix((data, (rows, cols)), shape=(n, n_cols))
    solution = lsqr(matrix, y, atol=1e-12, btol=1e-12)[0]
    coefficients = {}
    for cell, column in cell_index.items():
        coefficients[cell] = float(solution[column])
    return coefficients


def static_twfe(y, firm, year, g):
    """Coefficient on a single post indicator, with firm and year effects."""
    y = np.asarray(y, dtype=float)
    firm_codes, n_firms = _as_codes(firm)
    year_codes, n_years = _as_codes(year)
    g = np.asarray(g)
    year_raw = np.asarray(year)
    post = ((g > 0) & (year_raw >= g)).astype(float)
    n = len(y)
    n_cols = 1 + (n_firms - 1) + (n_years - 1) + 1
    rows, cols, data = [], [], []
    for i in range(n):
        rows.append(i)
        cols.append(0)
        data.append(1.0)
        if firm_codes[i] > 0:
            rows.append(i)
            cols.append(int(firm_codes[i]))
            data.append(1.0)
        if year_codes[i] > 0:
            rows.append(i)
            cols.append(n_firms + int(year_codes[i] - 1))
            data.append(1.0)
        if post[i] == 1.0:
            rows.append(i)
            cols.append(n_cols - 1)
            data.append(1.0)
    matrix = csc_matrix((data, (rows, cols)), shape=(n, n_cols))
    return float(lsqr(matrix, y, atol=1e-12, btol=1e-12)[0][-1])


def bootstrap_att(y, firm, year, g, n_boot=200, seed=7):
    """Firm-cluster bootstrap of the imputation ATT. Returns estimate, se."""
    y = np.asarray(y, dtype=float)
    firm = np.asarray(firm)
    year = np.asarray(year)
    g = np.asarray(g)
    point = imputation_effects(y, firm, year, g)["att"]
    units = np.unique(firm)
    # First treatment time is a unit-level attribute.
    g_by_unit = {}
    for unit in units:
        g_by_unit[unit] = int(g[firm == unit][0])
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    # Index rows by unit once.
    rows_of = {unit: np.flatnonzero(firm == unit) for unit in units}
    for b in range(n_boot):
        chosen = rng.choice(units, size=len(units), replace=True)
        parts_y, parts_f, parts_t, parts_g = [], [], [], []
        for j, unit in enumerate(chosen):
            idx = rows_of[unit]
            parts_y.append(y[idx])
            parts_t.append(year[idx])
            parts_f.append(np.full(len(idx), j, dtype=int))
            parts_g.append(np.full(len(idx), g_by_unit[unit], dtype=int))
        draws[b] = imputation_effects(
            np.concatenate(parts_y),
            np.concatenate(parts_f),
            np.concatenate(parts_t),
            np.concatenate(parts_g),
        )["att"]
    return point, float(np.std(draws, ddof=1))
