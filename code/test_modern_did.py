"""Synthetic checks for the staggered estimators. Not the A-share results."""

import numpy as np

from modern_did import etwfe_coefficients, imputation_effects, static_twfe


def _panel(n_per_group, periods, cohorts, tau_fn, sigma, seed):
    rng = np.random.default_rng(seed)
    groups = [0] + list(cohorts)
    firm, year, g, y = [], [], [], []
    unit = 0
    alpha = {}
    for cohort in groups:
        for _ in range(n_per_group):
            alpha[unit] = rng.normal()
            for t in periods:
                firm.append(unit)
                year.append(t)
                g.append(cohort)
                shock = rng.normal(0.0, sigma)
                effect = 0.0
                if cohort > 0 and t >= cohort:
                    effect = tau_fn(cohort, t - cohort)
                y.append(alpha[unit] + 0.01 * t + effect + shock)
            unit += 1
    return (
        np.array(y),
        np.array(firm),
        np.array(year),
        np.array(g),
    )


def test_homogeneous_effect_matches_static_twfe():
    periods = list(range(2013, 2021))
    cohorts = [2016, 2018]
    y, firm, year, g = _panel(
        80, periods, cohorts, lambda cohort, k: 0.4, sigma=0.05, seed=1
    )
    imputed = imputation_effects(y, firm, year, g)["att"]
    twfe = static_twfe(y, firm, year, g)
    assert abs(imputed - 0.4) < 0.03, imputed
    assert abs(twfe - 0.4) < 0.03, twfe


def test_heterogeneous_path_leaves_static_twfe_behind():
    periods = list(range(2013, 2023))
    cohorts = [2016, 2019]

    def tau_fn(cohort, k):
        if cohort == 2016:
            return 1.0 + 0.4 * k
        return 0.0

    y, firm, year, g = _panel(100, periods, cohorts, tau_fn, sigma=0.02, seed=2)
    imputed = imputation_effects(y, firm, year, g)
    # Observation-weighted truth on the same treated cells.
    truth = []
    for cohort, t, yi in zip(g, year, y):
        if cohort > 0 and t >= cohort:
            truth.append(tau_fn(cohort, t - cohort))
    truth = float(np.mean(truth))
    twfe = static_twfe(y, firm, year, g)
    assert abs(imputed["att"] - truth) < 0.03, (imputed["att"], truth)
    assert abs(twfe - truth) > 0.2, (twfe, truth)


def test_etwfe_matches_imputation_cell_by_cell():
    periods = list(range(2013, 2020))
    cohorts = [2016, 2018]

    def tau_fn(cohort, k):
        return 0.2 * (cohort - 2015) + 0.1 * k

    y, firm, year, g = _panel(40, periods, cohorts, tau_fn, sigma=0.0, seed=3)
    imputed = imputation_effects(y, firm, year, g)["by_cohort_event"]
    coefficients = etwfe_coefficients(y, firm, year, g, event_window=(2013 - 2018, 2019 - 2016))
    # Window must cover every realized event time. Recompute with a wide window.
    coefficients = etwfe_coefficients(y, firm, year, g, event_window=(-6, 6))
    for cell, payload in imputed.items():
        assert cell in coefficients, cell
        gap = abs(coefficients[cell] - payload["estimate"])
        assert gap < 1e-6, (cell, coefficients[cell], payload["estimate"], gap)


if __name__ == "__main__":
    test_homogeneous_effect_matches_static_twfe()
    test_heterogeneous_path_leaves_static_twfe_behind()
    test_etwfe_matches_imputation_cell_by_cell()
    print("modern_did synthetic checks passed")
