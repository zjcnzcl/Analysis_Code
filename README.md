# Reproducible analysis code

This package contains the Python analysis code for a within-participant crossover pilot study of color-recognition performance under four light-source conditions. Eight prespecified source-order sequences (R01-R08) were applied across the four light-source conditions.

## Contents

* `analysis_primary.py`: primary Bayesian mixed-effects logistic model and four Bayesian sensitivity analyses reported in the revision.

* `analysis_trial_secondary.py`: response-time GEE and confidence-score summaries.

* `analysis_subjective.py`: Raw NASA-TLX, visual-fatigue, realism, and comfort analyses.

* `run_all.py`: runs all analyses from one command.

* `data_dictionary.csv`: analysis-facing data dictionary and public-release notes.

* `subjective_scale_endpoints.csv`: verbal endpoints for the 1-5 subjective items.

* `requirements.txt`: tested Python and package versions.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run_all.py --input data_v1_deidentified.xlsx --output results
```

The input workbook should contain these sheets:

* `05_分析随机顺序` for trial-level accuracy, response time, confidence, light source, task, period, group, participant, and sample.

* `07_主观评分记录` for the six NASA-TLX dimensions and subjective outcomes.

The scripts accept the Chinese column names used in the study workbook. The input workbook is not modified. Results are written as CSV and JSON files to the output directory.

## Statistical specification

The primary outcome is trial-level recognition accuracy (`correct`, 0/1). The primary model is a Bayesian mixed-effects logistic model:

`correct ~ light + task + period + group`

with random intercepts for participant and sample plus a participant-by-light random-variation component. The reference light is `L1`. In `statsmodels`, `fe_p=2.0` is the standard deviation of the Gaussian prior for fixed effects, while `vcp_p=1.0` is the standard deviation of the Gaussian prior for the log standard deviations of the random effects. Reported intervals are approximate 95% credible intervals calculated as posterior mean +/- 1.96 posterior SD on the log-odds scale and exponentiated to odds ratios.

The Bayesian sensitivity analyses use (1) a tighter random-effect standard-deviation prior (`vcp_p=0.5`), (2) a tighter fixed-effect prior (`fe_p=1.0`), (3) a reduced random-intercepts-only structure, and (4) a healthcare-professionals-only refit using the primary random-effect structure with the participant-group fixed effect removed. The package records optimizer status and gradient norms for each fitted Bayesian model. It does not include the exploratory logistic GEE for accuracy because that model did not yield stable, interpretable estimates in this small pilot dataset.

For reproducible variational-Bayes initialization, `analysis_primary.py` uses the default random seed `20260924`; it can be overridden with `--seed` when the script is run directly.

NASA-TLX is reported as Raw NASA-TLX: the unweighted arithmetic mean of the six 0-10 dimensions. No pairwise weighting data are required or reconstructed. Visual fatigue is the sum of the three 1-5 items (dryness, eye strain, and glare discomfort); realism and comfort are single 1-5 items.

For secondary analyses, response times are log-transformed and analysed using a participant-clustered Gaussian GEE with exchangeable working correlation, adjusted for light, task, period, and group. Sample is not included as a simultaneous fixed effect because it is nested within task, producing a rank-deficient design matrix. Raw NASA-TLX and visual fatigue are analysed with analogous participant-clustered Gaussian GEEs adjusted for light, period, and group. Confidence, color realism, and visual comfort are summarized descriptively by light condition.

## Response-label normalization and validation (2026-10-26)

Regenerate Figure 3 with `python plot_figure3.py --input data_v1_deidentified.xlsx --output figures`. Its error bars are participant-level t-based 95% confidence intervals. Arial is used when installed; a system fallback font may otherwise be selected.

`response_scoring.py` normalizes five synonymous tissue-response labels using the exact mapping supplied in `response_label_mapping.csv`, and then recomputes correctness from the recorded reference and response. Both columns `标准答案大类` and `受试者答案`, and the existing `是否正确1_0` column, must be available in the trial sheet. Raw responses are not overwritten. Where present, `原始计分1_0` is retained as the pre-normalization audit baseline.

The correction changes 54 records from 0 to 1 (L1: 23; L2: 15; L3: 10; L4: 6). Correct counts are 817, 945, 1064, and 1038, respectively, out of 1200 trials per light. The scripts export the record-level scoring audit and quality-check totals. Reprocessing the corrected workbook makes no additional scoring changes. L2 is described neutrally as LED (Ra 83; low R9).

All five Bayesian model fits converged in the validation run. Primary odds ratios rounded to two decimal places were 1.80, 3.90, and 3.15 versus L1. The workbook and participant-level result files are not included in this public code package; access is governed by the manuscript's Data Availability Statement. The supplemental reflectance dataset is supplied separately.

Raw NASA-TLX is recomputed from its six dimensions without intermediate rounding. Installation versions in `requirements.txt` reflect the successful validation run.
