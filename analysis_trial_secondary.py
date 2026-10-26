from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE

from analysis_primary import load_trials


def fit_response_time_gee(df: pd.DataFrame) -> pd.DataFrame:
    """Fit the Gaussian GEE for log-transformed positive response times."""
    valid = df.loc[df["response_time_s"].notna()].copy()
    valid["response_time_s"] = pd.to_numeric(valid["response_time_s"], errors="raise")
    valid = valid.loc[valid["response_time_s"] > 0].copy()
    if valid.empty:
        raise ValueError("No positive response-time values were available.")
    valid["log_response_time"] = np.log(valid["response_time_s"])
    formula = (
        "log_response_time ~ C(light, Treatment('L1')) + C(task) + C(period) + "
        "C(group)"
    )
    fit = GEE.from_formula(
        formula,
        groups="participant",
        data=valid,
        family=Gaussian(),
        cov_struct=Exchangeable(),
    ).fit()
    rows = []
    for term, estimate in fit.params.items():
        standard_error = float(fit.bse[term])
        rows.append(
            {
                "term": term,
                "estimate_log_response_time": float(estimate),
                "robust_se": standard_error,
                "geometric_mean_ratio": float(np.exp(estimate)),
                "approx_95_ci_low": float(np.exp(estimate - 1.96 * standard_error)),
                "approx_95_ci_high": float(np.exp(estimate + 1.96 * standard_error)),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Secondary trial-level response-time and confidence analyses."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    df = load_trials(args.input)
    response_time = fit_response_time_gee(df)
    response_time.to_csv(args.output / "response_time_gee_fixed_effects.csv", index=False)
    response_time_metadata = {
        "n_positive_response_time_trials": int((pd.to_numeric(df["response_time_s"], errors="raise") > 0).sum()),
        "formula": "log_response_time ~ C(light, Treatment('L1')) + C(task) + C(period) + C(group)",
        "working_correlation": "exchangeable within participant",
        "sample_fixed_effect_excluded_reason": "Sample was nested within task; including both fixed effects produced a rank-deficient design matrix and unstable estimates.",
    }
    (args.output / "response_time_model_metadata.json").write_text(
        json.dumps(response_time_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    df["confidence"] = pd.to_numeric(df["confidence"], errors="raise")
    confidence_summary = (
        df.groupby("light", observed=True)
        .agg(
            trials=("confidence", "size"),
            confidence_mean=("confidence", "mean"),
            confidence_sd=("confidence", "std"),
        )
        .reset_index()
    )
    confidence_summary.to_csv(args.output / "confidence_summary_by_light.csv", index=False)


if __name__ == "__main__":
    main()
