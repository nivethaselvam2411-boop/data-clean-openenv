"""
Deterministic graders for DataClean environment.
Each grader scores agent performance from 0.0 to 1.0.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from environment.models import RewardBreakdown, Reward


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _null_score(df: pd.DataFrame, ref: pd.DataFrame, columns: list) -> float:
    """Score based on remaining nulls vs reference."""
    total_ref_nulls = sum(ref[c].isna().sum() for c in columns if c in ref.columns)
    total_df_nulls = sum(df[c].isna().sum() for c in columns if c in df.columns)
    original_max = max(total_df_nulls, 1)
    if total_df_nulls == 0 and total_ref_nulls == 0:
        return 1.0
    if total_ref_nulls == 0:
        return max(0.0, 1.0 - total_df_nulls / original_max)
    ratio = total_df_nulls / max(total_ref_nulls, 1)
    return max(0.0, min(1.0, 1.0 - (ratio - 1.0) * 0.5))


def _null_pct_score(df: pd.DataFrame, original_df: pd.DataFrame, columns: list) -> float:
    """Score based on percentage of nulls fixed."""
    orig_nulls = sum(original_df[c].isna().sum() for c in columns if c in original_df.columns)
    curr_nulls = sum(df[c].isna().sum() for c in columns if c in df.columns)
    if orig_nulls == 0:
        return 1.0
    fixed = orig_nulls - curr_nulls
    return max(0.0, min(1.0, fixed / orig_nulls))


def _dtype_score(df: pd.DataFrame, expected_dtypes: Dict[str, str]) -> float:
    """Score based on how many columns have correct dtype."""
    if not expected_dtypes:
        return 1.0
    correct = 0
    for col, expected in expected_dtypes.items():
        if col not in df.columns:
            continue
        actual = str(df[col].dtype)
        if expected == "numeric" and pd.api.types.is_numeric_dtype(df[col]):
            correct += 1
        elif expected == "datetime" and pd.api.types.is_datetime64_any_dtype(df[col]):
            correct += 1
        elif expected == "string" and pd.api.types.is_string_dtype(df[col]):
            correct += 1
        elif expected in actual:
            correct += 1
    return correct / len(expected_dtypes)


def _duplicate_score(df: pd.DataFrame, subset: list | None = None) -> float:
    """Score: 1.0 if no duplicates, else proportional."""
    total = len(df)
    if total == 0:
        return 0.0
    dups = df.duplicated(subset=subset).sum()
    return max(0.0, 1.0 - dups / total)


def _outlier_score(df: pd.DataFrame, columns: list, method: str = "iqr", threshold: float = 1.5) -> float:
    """Score based on outliers remaining in numeric columns."""
    if not columns:
        return 1.0
    scores = []
    for col in columns:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) == 0:
            continue
        if method == "iqr":
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            outliers = ((series < q1 - threshold * iqr) | (series > q3 + threshold * iqr)).sum()
        elif method == "zscore":
            z = np.abs((series - series.mean()) / series.std())
            outliers = (z > threshold).sum()
        else:
            outliers = 0
        scores.append(max(0.0, 1.0 - outliers / len(series)))
    return float(np.mean(scores)) if scores else 1.0


def _format_score(df: pd.DataFrame, columns: Dict[str, str]) -> float:
    """Score string formatting consistency."""
    if not columns:
        return 1.0
    scores = []
    for col, fmt in columns.items():
        if col not in df.columns:
            continue
        series = df[col].dropna().astype(str)
        if len(series) == 0:
            scores.append(1.0)
            continue
        if fmt == "titlecase":
            pct = (series == series.str.title()).mean()
        elif fmt == "uppercase":
            pct = (series == series.str.upper()).mean()
        elif fmt == "lowercase":
            pct = (series == series.str.lower()).mean()
        elif fmt == "stripped":
            pct = (series == series.str.strip()).mean()
        else:
            pct = 1.0
        scores.append(float(pct))
    return float(np.mean(scores)) if scores else 1.0


def _efficiency_penalty(n_steps: int, optimal_steps: int, max_steps: int) -> float:
    """Penalize excessive steps. Returns 0.0 (max penalty) to 1.0 (no penalty)."""
    if n_steps <= optimal_steps:
        return 1.0
    excess = n_steps - optimal_steps
    allowance = max_steps - optimal_steps
    return max(0.0, 1.0 - (excess / allowance) * 0.3)


# ---------------------------------------------------------------------------
# Task 1 Grader — Easy (Missing Values)
# ---------------------------------------------------------------------------

def grade_task1(
    df: pd.DataFrame,
    original_df: pd.DataFrame,
    n_steps: int,
    max_steps: int,
) -> Reward:
    nullable_cols = ["salary", "department", "tenure_years"]

    null_s = _null_pct_score(df, original_df, nullable_cols)
    dtype_s = _dtype_score(df, {"salary": "numeric", "tenure_years": "numeric"})
    dup_s = _duplicate_score(df)
    outlier_s = 1.0  # not required for task 1
    schema_s = 1.0   # not required for task 1
    format_s = 1.0   # not required for task 1
    eff_s = _efficiency_penalty(n_steps, optimal_steps=3, max_steps=max_steps)

    # Weighted total
    total = (
        null_s * 0.55
        + dtype_s * 0.15
        + dup_s * 0.10
        + eff_s * 0.20
    )
    total = round(min(1.0, max(0.0, total)), 4)

    breakdown = RewardBreakdown(
        null_score=round(null_s, 4),
        dtype_score=round(dtype_s, 4),
        duplicate_score=round(dup_s, 4),
        outlier_score=round(outlier_s, 4),
        schema_score=round(schema_s, 4),
        format_score=round(format_s, 4),
        efficiency_penalty=round(eff_s, 4),
    )

    remaining_nulls = sum(df[c].isna().sum() for c in nullable_cols if c in df.columns)
    done = remaining_nulls == 0 or n_steps >= max_steps

    return Reward(
        total=total,
        breakdown=breakdown,
        done=done,
        info={
            "remaining_nulls": int(remaining_nulls),
            "task": "task1_fill_missing",
            "grade": "A" if total >= 0.85 else "B" if total >= 0.70 else "C" if total >= 0.50 else "F",
        },
    )


# ---------------------------------------------------------------------------
# Task 2 Grader — Medium (Dtypes + Duplicates)
# ---------------------------------------------------------------------------

def grade_task2(
    df: pd.DataFrame,
    original_df: pd.DataFrame,
    n_steps: int,
    max_steps: int,
) -> Reward:
    expected_dtypes = {
        "quantity": "numeric",
        "unit_price": "numeric",
        "customer_id": "numeric",
        "order_date": "datetime",
    }
    nullable_cols = ["quantity", "unit_price"]
    subset_cols = ["order_id"]

    null_s = _null_pct_score(df, original_df, nullable_cols)
    dtype_s = _dtype_score(df, expected_dtypes)
    dup_s = _duplicate_score(df, subset=subset_cols)
    outlier_s = 1.0
    schema_s = 1.0
    format_s = 1.0
    eff_s = _efficiency_penalty(n_steps, optimal_steps=6, max_steps=max_steps)

    total = (
        dtype_s * 0.35
        + dup_s * 0.30
        + null_s * 0.15
        + eff_s * 0.20
    )
    total = round(min(1.0, max(0.0, total)), 4)

    breakdown = RewardBreakdown(
        null_score=round(null_s, 4),
        dtype_score=round(dtype_s, 4),
        duplicate_score=round(dup_s, 4),
        outlier_score=round(outlier_s, 4),
        schema_score=round(schema_s, 4),
        format_score=round(format_s, 4),
        efficiency_penalty=round(eff_s, 4),
    )

    remaining_dups = df.duplicated(subset=subset_cols).sum()
    dtype_done = dtype_s >= 0.99
    done = (remaining_dups == 0 and dtype_done) or n_steps >= max_steps

    return Reward(
        total=total,
        breakdown=breakdown,
        done=done,
        info={
            "remaining_duplicates": int(remaining_dups),
            "dtype_score": round(dtype_s, 4),
            "task": "task2_dtype_dedup",
            "grade": "A" if total >= 0.85 else "B" if total >= 0.70 else "C" if total >= 0.50 else "F",
        },
    )


# ---------------------------------------------------------------------------
# Task 3 Grader — Hard (Full Pipeline)
# ---------------------------------------------------------------------------

def grade_task3(
    df: pd.DataFrame,
    original_df: pd.DataFrame,
    n_steps: int,
    max_steps: int,
) -> Reward:
    expected_dtypes = {
        "age": "numeric",
        "systolic_bp": "numeric",
        "bmi": "numeric",
        "admission_date": "datetime",
    }
    nullable_cols = ["age", "gender", "bmi", "cholesterol", "glucose"]
    outlier_cols = ["systolic_bp", "bmi"]
    format_cols = {
        "gender": "titlecase",
        "blood_type": "stripped",
        "diagnosis_code": "uppercase",
    }

    null_s = _null_pct_score(df, original_df, nullable_cols)
    dtype_s = _dtype_score(df, expected_dtypes)
    dup_s = _duplicate_score(df, subset=["patient_id"])
    outlier_s = _outlier_score(df, outlier_cols, method="iqr", threshold=1.5)
    format_s = _format_score(df, format_cols)
    schema_s = 1.0 if all(c in df.columns for c in [
        "patient_id", "age", "gender", "blood_type", "systolic_bp",
        "diastolic_bp", "bmi", "cholesterol", "glucose", "admission_date", "diagnosis_code"
    ]) else 0.5
    eff_s = _efficiency_penalty(n_steps, optimal_steps=10, max_steps=max_steps)

    total = (
        null_s * 0.20
        + dtype_s * 0.20
        + dup_s * 0.15
        + outlier_s * 0.20
        + format_s * 0.10
        + schema_s * 0.05
        + eff_s * 0.10
    )
    total = round(min(1.0, max(0.0, total)), 4)

    breakdown = RewardBreakdown(
        null_score=round(null_s, 4),
        dtype_score=round(dtype_s, 4),
        duplicate_score=round(dup_s, 4),
        outlier_score=round(outlier_s, 4),
        schema_score=round(schema_s, 4),
        format_score=round(format_s, 4),
        efficiency_penalty=round(eff_s, 4),
    )

    done = total >= 0.90 or n_steps >= max_steps

    return Reward(
        total=total,
        breakdown=breakdown,
        done=done,
        info={
            "task": "task3_full_pipeline",
            "grade": "A" if total >= 0.85 else "B" if total >= 0.70 else "C" if total >= 0.50 else "F",
            "pipeline_complete": total >= 0.90,
        },
    )
