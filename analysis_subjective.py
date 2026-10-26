from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Gaussian
from statsmodels.genmod.generalized_estimating_equations import GEE


SHEET = "07_主观评分记录"
COLUMNS = {
    "受试者编号": "participant",
    "分组": "group",
    "光源编号": "light",
    "光源顺序位次": "period",
    "心理需求0_10": "mental_demand",
    "体力需求0_10": "physical_demand",
    "时间压力0_10": "temporal_demand",
    "表现不满意0_10": "performance_dissatisfaction",
    "努力程度0_10": "effort",
    "挫败感0_10": "frustration",
    "TLX平均分": "recorded_tlx_mean",
    "眼干涩1_5": "eye_dryness",
    "眼酸胀1_5": "eye_strain",
    "眩光不适1_5": "glare_discomfort",
    "视觉疲劳总分": "recorded_visual_fatigue",
    "颜色真实感1_5": "color_realism",
    "整体舒适度1_5": "visual_comfort",
}
TLX = [
    "mental_demand",
    "physical_demand",
    "temporal_demand",
    "performance_dissatisfaction",
    "effort",
    "frustration",
]
FATIGUE = ["eye_dryness", "eye_strain", "glare_discomfort"]


def load_subjective(path: Path) -> pd.DataFrame:
    """Load and validate the deidentified condition-level subjective ratings."""
    raw = pd.read_excel(path, sheet_name=SHEET)
    missing = [column for column in COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns in {SHEET}: {missing}")

    df = raw.rename(columns=COLUMNS)[list(COLUMNS.values())].copy()
    for column in ["participant", "group", "light"]:
        df[column] = df[column].astype("category")
    df["period"] = pd.to_numeric(df["period"], errors="raise").astype(int)
    for column in TLX + FATIGUE + [
        "recorded_tlx_mean",
        "recorded_visual_fatigue",
        "color_realism",
        "visual_comfort",
    ]:
        df[column] = pd.to_numeric(df[column], errors="raise")
    df["tlx_mean_recomputed"] = df[TLX].mean(axis=1)
    df["visual_fatigue_recomputed"] = df[FATIGUE].sum(axis=1)
    df["tlx_difference"] = df["recorded_tlx_mean"] - df["tlx_mean_recomputed"]
    df["visual_fatigue_difference"] = (
        df["recorded_visual_fatigue"] - df["visual_fatigue_recomputed"]
    )

    expected_lights = ["L1", "L2", "L3", "L4"]
    present = [light for light in expected_lights if light in df["light"].cat.categories]
    if present != expected_lights:
        raise ValueError(f"Expected light conditions {expected_lights}; found {present}")
    df["light"] = df["light"].cat.reorder_categories(expected_lights, ordered=True)
    return df


def fit_subjective_gee(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Fit the Gaussian GEE used for the continuous condition-level outcomes."""
    formula = f"{outcome} ~ C(light, Treatment('L1')) + C(period) + C(group)"
    fit = GEE.from_formula(
        formula,
        groups="participant",
        data=df,
        family=Gaussian(),
        cov_struct=Exchangeable(),
    ).fit()
    rows = []
    for term, estimate in fit.params.items():
        standard_error = float(fit.bse[term])
        rows.append(
            {
                "outcome": outcome,
                "term": term,
                "estimate_mean_difference": float(estimate),
                "robust_se": standard_error,
                "approx_95_ci_low": float(estimate - 1.96 * standard_error),
                "approx_95_ci_high": float(estimate + 1.96 * standard_error),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Raw NASA-TLX and subjective-outcome analyses."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    df = load_subjective(args.input)
    summary_by_light = (
        df.groupby("light", observed=True)
        .agg(
            rows=("light", "size"),
            raw_nasa_tlx_mean=("tlx_mean_recomputed", "mean"),
            visual_fatigue_mean=("visual_fatigue_recomputed", "mean"),
            color_realism_mean=("color_realism", "mean"),
            visual_comfort_mean=("visual_comfort", "mean"),
        )
        .reset_index()
    )
    summary_by_light.to_csv(args.output / "subjective_summary_by_light.csv", index=False)
    summary_by_light_group = (
        df.groupby(["light", "group"], observed=True)
        .agg(
            rows=("light", "size"),
            raw_nasa_tlx_mean=("tlx_mean_recomputed", "mean"),
            visual_fatigue_mean=("visual_fatigue_recomputed", "mean"),
            color_realism_mean=("color_realism", "mean"),
            visual_comfort_mean=("visual_comfort", "mean"),
        )
        .reset_index()
    )
    summary_by_light_group.to_csv(
        args.output / "subjective_summary_by_light_group.csv", index=False
    )

    gee_results = pd.concat(
        [
            fit_subjective_gee(df, "tlx_mean_recomputed"),
            fit_subjective_gee(df, "visual_fatigue_recomputed"),
        ],
        ignore_index=True,
    )
    gee_results.to_csv(args.output / "subjective_gee_fixed_effects.csv", index=False)

    checks = {
        "n_rows": int(len(df)),
        "tlx_mean_max_absolute_difference": float(df["tlx_difference"].abs().max()),
        "visual_fatigue_max_absolute_difference": float(
            df["visual_fatigue_difference"].abs().max()
        ),
        "tlx_is_unweighted_mean": True,
        "raw_nasa_tlx_pairwise_weighting_used": False,
        "visual_fatigue_is_sum_of_three_items": True,
        "gee_formula": "outcome ~ C(light, Treatment('L1')) + C(period) + C(group)",
        "gee_working_correlation": "exchangeable within participant",
    }
    (args.output / "subjective_quality_checks.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
