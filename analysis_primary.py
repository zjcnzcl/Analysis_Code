from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
from response_scoring import normalize_accuracy


TRIAL_SHEET = "05_分析随机顺序"
TRIAL_COLUMNS = {
    "受试者编号": "participant",
    "分组": "group",
    "光源编号": "light",
    "光源顺序位次": "period",
    "任务类型": "task",
    "样本编号": "sample",
    "是否正确1_0": "correct",
    "反应时间s": "response_time_s",
    "信心评分1_5": "confidence",
}

FORMULA = "correct ~ C(light, Treatment('L1')) + C(task) + C(period) + C(group)"
HEALTHCARE_ONLY_FORMULA = "correct ~ C(light, Treatment('L1')) + C(task) + C(period)"
PRIMARY_VC = {
    "participant": "0 + C(participant)",
    "sample": "0 + C(sample)",
    "participant_by_light": "0 + C(participant):C(light)",
}
RANDOM_INTERCEPT_VC = {
    "participant": "0 + C(participant)",
    "sample": "0 + C(sample)",
}


def load_trials(path: Path, sheet: str = TRIAL_SHEET) -> pd.DataFrame:
    """Load the deidentified trial-level workbook without modifying it."""
    raw = pd.read_excel(path, sheet_name=sheet)
    raw, _ = normalize_accuracy(raw)
    missing = [column for column in TRIAL_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns in {sheet}: {missing}")

    df = raw.rename(columns=TRIAL_COLUMNS)[list(TRIAL_COLUMNS.values())].copy()
    for column in ["participant", "group", "light", "task", "sample"]:
        df[column] = df[column].astype("category")
    df["period"] = pd.to_numeric(df["period"], errors="raise").astype(int)
    df["correct"] = pd.to_numeric(df["correct"], errors="raise").astype(int)
    if not set(df["correct"].dropna().unique()).issubset({0, 1}):
        raise ValueError("是否正确1_0 must contain only 0 and 1")

    expected_lights = ["L1", "L2", "L3", "L4"]
    present = [light for light in expected_lights if light in df["light"].cat.categories]
    if present != expected_lights:
        raise ValueError(f"Expected light conditions {expected_lights}; found {present}")
    df["light"] = df["light"].cat.reorder_categories(expected_lights, ordered=True)
    return df


def _fit_bayesian_model(
    df: pd.DataFrame,
    label: str,
    variance_components: dict[str, str],
    fe_p: float,
    vcp_p: float,
    formula: str = FORMULA,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fit one variational-Bayes logistic mixed model and summarize its coefficients."""
    model = BinomialBayesMixedGLM.from_formula(
        formula, variance_components, df, fe_p=fe_p, vcp_p=vcp_p
    )
    fit = model.fit_vb()

    coefficient_rows = []
    for term, mean, sd in zip(model.exog_names, fit.fe_mean, fit.fe_sd):
        coefficient_rows.append(
            {
                "analysis": label,
                "term": term,
                "posterior_mean_log_odds": float(mean),
                "posterior_sd_log_odds": float(sd),
                "odds_ratio": float(np.exp(mean)),
                "approx_95_cri_low": float(np.exp(mean - 1.96 * sd)),
                "approx_95_cri_high": float(np.exp(mean + 1.96 * sd)),
            }
        )

    optimizer = getattr(fit, "optim_retvals", {})
    gradient = optimizer.get("jac") if isinstance(optimizer, dict) else None
    metadata = {
        "analysis": label,
        "formula": formula,
        "n_rows": int(len(df)),
        "n_participants": int(df["participant"].nunique()),
        "fixed_effect_prior_sd_fe_p": fe_p,
        "log_random_effect_sd_prior_sd_vcp_p": vcp_p,
        "variance_components": variance_components,
        "optimizer_success": bool(optimizer.get("success", False)) if isinstance(optimizer, dict) else None,
        "optimizer_gradient_norm": float(np.linalg.norm(gradient)) if gradient is not None else None,
    }
    return pd.DataFrame(coefficient_rows), metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Primary Bayesian trial-level accuracy analysis and prespecified sensitivity analyses."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sheet", default=TRIAL_SHEET)
    parser.add_argument("--seed", default=20260924, type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    raw = pd.read_excel(args.input, sheet_name=args.sheet)
    _, audit = normalize_accuracy(raw)
    audit.loc[audit["changed_from_original"]].to_csv(
        args.output / "response_scoring_changes.csv", index=False, encoding="utf-8-sig"
    )
    (args.output / "response_scoring_quality.json").write_text(
        json.dumps({
            "n_trials": len(audit),
            "changes_from_original": int(audit["changed_from_original"].sum()),
            "changes_from_input": int(audit["changed_from_input"].sum()),
            "changed_scores_by_light": audit.groupby("light")["changed_from_original"].sum().astype(int).to_dict(),
            "raw_response_labels_preserved": True,
        }, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    df = load_trials(args.input, args.sheet)
    analyses = [
        ("primary_participant_by_light_random_variation", df, PRIMARY_VC, 2.0, 1.0, FORMULA),
        ("tighter_random_effect_sd_prior", df, PRIMARY_VC, 2.0, 0.5, FORMULA),
        ("tighter_fixed_effect_prior", df, PRIMARY_VC, 1.0, 1.0, FORMULA),
        ("random_intercepts_only_structure_sensitivity", df, RANDOM_INTERCEPT_VC, 2.0, 1.0, FORMULA),
        (
            "healthcare_professionals_only_sensitivity",
            df.loc[df["group"] == "医护人员"].copy(),
            PRIMARY_VC,
            2.0,
            1.0,
            HEALTHCARE_ONLY_FORMULA,
        ),
    ]

    summaries = []
    model_metadata = []
    for label, analysis_df, variance_components, fe_p, vcp_p, formula in analyses:
        summary, metadata = _fit_bayesian_model(
            analysis_df, label, variance_components, fe_p, vcp_p, formula
        )
        summaries.append(summary)
        model_metadata.append(metadata)

    all_coefficients = pd.concat(summaries, ignore_index=True)
    all_coefficients.to_csv(args.output / "bayesian_model_coefficients_all.csv", index=False)
    all_coefficients.loc[
        all_coefficients["term"].str.contains("C\\(light", regex=True)
    ].to_csv(args.output / "bayesian_light_effect_sensitivities.csv", index=False)

    descriptives_by_light = (
        df.groupby("light", observed=True)
        .agg(trials=("correct", "size"), correct=("correct", "sum"), accuracy=("correct", "mean"))
        .reset_index()
    )
    descriptives_by_light.to_csv(args.output / "trial_descriptives_by_light.csv", index=False)
    descriptives_by_task = (
        df.groupby(["light", "task"], observed=True)
        .agg(trials=("correct", "size"), correct=("correct", "sum"), accuracy=("correct", "mean"))
        .reset_index()
    )
    descriptives_by_task.to_csv(args.output / "trial_descriptives_by_light_task.csv", index=False)

    metadata = {
        "python_packages": {"statsmodels": statsmodels.__version__},
        "n_rows": int(len(df)),
        "n_participants": int(df["participant"].nunique()),
        "n_samples": int(df["sample"].nunique()),
        "formula": FORMULA,
        "reference_light": "L1",
        "random_seed": args.seed,
        "models": model_metadata,
        "interval_definition": "Approximate 95% credible interval: posterior mean +/- 1.96 posterior SD on the log-odds scale, exponentiated to odds ratios.",
    }
    (args.output / "bayesian_model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
