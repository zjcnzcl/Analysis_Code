"""Normalize equivalent response labels without modifying recorded responses.

The mapping implements the five tissue response categories in the study's
response-option sheet. Original response strings are preserved for audit.
"""
import pandas as pd

RESPONSE_LABEL_MAP = {
    "正常组织色": "正常肌肉",
    "脂肪样色": "脂肪样",
    "充血样色": "充血样",
    "缺血样色": "缺血样",
    "暗褐样色": "坏死/暗褐",
}

def normalize_accuracy(raw: pd.DataFrame):
    required = ["标准答案大类", "受试者答案", "是否正确1_0"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing scoring fields: {missing}")
    if raw[required[:2]].isna().any().any():
        raise ValueError("Missing reference answers or participant responses")
    scored = raw.copy()
    reference = raw["标准答案大类"].replace(RESPONSE_LABEL_MAP)
    response = raw["受试者答案"].replace(RESPONSE_LABEL_MAP)
    final = reference.eq(response).astype(int)
    original_column = "原始计分1_0" if "原始计分1_0" in raw else "是否正确1_0"
    original = pd.to_numeric(raw[original_column], errors="raise").astype(int)
    current = pd.to_numeric(raw["是否正确1_0"], errors="raise").astype(int)
    audit = pd.DataFrame({
        "excel_row": raw.index + 2,
        "participant": raw["受试者编号"],
        "light": raw["光源编号"],
        "sample": raw["样本编号"],
        "reference_recorded": raw["标准答案大类"],
        "response_recorded": raw["受试者答案"],
        "reference_normalized": reference,
        "response_normalized": response,
        "original_score": original,
        "input_score": current,
        "final_score": final,
        "changed_from_original": final.ne(original),
        "changed_from_input": final.ne(current),
    })
    scored["是否正确1_0"] = final
    return scored, audit
