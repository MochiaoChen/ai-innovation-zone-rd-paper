"""05c: imputation and extended two-way fixed effects for the A-share panel.

Reads data/clean/analysis_data.csv. That file is not in the repository
because the source data are licensed. If it is absent, this script exits
without writing coefficients.

When the file is present, it writes output/tables/table5_modern_did.csv with
the observation-weighted imputation ATT and the cohort-by-event-time cells.
Covariates, if used, are lagged one year. Contemporaneous size, leverage,
profitability, cash and Tobin's Q are not entered: they can respond to the
policy.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modern_did import bootstrap_att, etwfe_coefficients, imputation_effects

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEAN = os.path.join(BASE, "data/clean/analysis_data.csv")
OUT = os.path.join(BASE, "output/tables/table5_modern_did.csv")


def main():
    if not os.path.exists(CLEAN):
        print(
            "[05c] data/clean/analysis_data.csv is not in the repository. "
            "No imputation or extended-TWFE coefficients were computed. "
            "The synthetic checks live in code/test_modern_did.py."
        )
        return

    import pandas as pd

    df = pd.read_csv(CLEAN, encoding="utf-8-sig", low_memory=False)
    id_col = "stkcd" if "stkcd" in df.columns else "code"
    need = [id_col, "year", "RD", "treat_year"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"[05c] analysis_data.csv is missing columns: {missing}")

    sub = df.dropna(subset=["RD", "year"]).copy()
    sub["g"] = sub["treat_year"].fillna(0).astype(int)
    sub["year"] = sub["year"].astype(int)
    y = sub["RD"].to_numpy(dtype=float)
    firm = sub[id_col].to_numpy()
    year = sub["year"].to_numpy()
    g = sub["g"].to_numpy()

    point, se = bootstrap_att(y, firm, year, g, n_boot=200, seed=7)
    cells = imputation_effects(y, firm, year, g)["by_cohort_event"]
    etwfe = etwfe_coefficients(y, firm, year, g, event_window=(-4, 4))

    rows = [f"overall_imputation_att,{point:.6f},{se:.6f},boot"]
    for (cohort, event), payload in sorted(cells.items()):
        if event < -4 or event > 4:
            continue
        twin = etwfe.get((cohort, event), np.nan)
        rows.append(
            f"cell,{cohort},{event},{payload['estimate']:.6f},{payload['n']},{twin:.6f}"
        )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        handle.write("kind,cohort_or_att,event,estimate,n_or_se,etwfe\n")
        # The overall row does not have a cohort. Keep a rectangular file.
        handle.write(f"att,,,{point:.6f},{se:.6f},\n")
        for line in rows[1:]:
            kind, cohort, event, estimate, n, twin = line.split(",")
            handle.write(f"{kind},{cohort},{event},{estimate},{n},{twin}\n")
    print(f"[05c] imputation ATT {point:.4f} (bootstrap se {se:.4f})")
    print(f"[05c] wrote {OUT}")


if __name__ == "__main__":
    main()
