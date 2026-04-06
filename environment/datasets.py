"""
Synthetic dataset generator for DataClean environment.
Produces reproducible, realistic messy datasets for each task.
"""

from __future__ import annotations
import random
import numpy as np
import pandas as pd
from typing import Tuple


# ---------------------------------------------------------------------------
# Task 1 — Easy: Missing Values Only
# ---------------------------------------------------------------------------

def generate_task1_dataset(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    HR Employee dataset with missing values in salary, department, and tenure.
    Returns (dirty_df, clean_df).
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    n = 120

    departments = ["Engineering", "Marketing", "Sales", "HR", "Finance"]
    names = [f"Employee_{i:03d}" for i in range(n)]
    dept = rng.choice(departments, size=n).tolist()
    salary = (rng.normal(75000, 20000, size=n).clip(30000, 150000)).tolist()
    tenure = (rng.integers(1, 20, size=n)).tolist()
    age = (rng.integers(22, 60, size=n)).tolist()
    performance = rng.choice(["Low", "Medium", "High", "Excellent"], size=n).tolist()

    df = pd.DataFrame({
        "employee_id": [f"EMP{i:04d}" for i in range(n)],
        "name": names,
        "department": dept,
        "salary": salary,
        "tenure_years": tenure,
        "age": age,
        "performance_rating": performance,
    })

    # Introduce missing values
    missing_idx_salary = rng.choice(n, size=18, replace=False)
    missing_idx_dept = rng.choice(n, size=12, replace=False)
    missing_idx_tenure = rng.choice(n, size=10, replace=False)

    dirty = df.copy()
    dirty.loc[missing_idx_salary, "salary"] = np.nan
    dirty.loc[missing_idx_dept, "department"] = np.nan
    dirty.loc[missing_idx_tenure, "tenure_years"] = np.nan

    # Clean version: fill with sensible strategies
    clean = dirty.copy()
    clean["salary"] = clean["salary"].fillna(clean["salary"].median())
    clean["department"] = clean["department"].fillna(clean["department"].mode()[0])
    clean["tenure_years"] = clean["tenure_years"].fillna(clean["tenure_years"].median())

    return dirty, clean


# ---------------------------------------------------------------------------
# Task 2 — Medium: Wrong Dtypes + Duplicates
# ---------------------------------------------------------------------------

def generate_task2_dataset(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    E-commerce Orders dataset with wrong dtypes, duplicates, and missing values.
    Returns (dirty_df, clean_df).
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    n = 200

    categories = ["Electronics", "Clothing", "Books", "Home", "Sports"]
    order_ids = [f"ORD{i:05d}" for i in range(n)]
    product_ids = [f"PROD{rng.integers(100, 999)}" for _ in range(n)]
    quantity = rng.integers(1, 20, size=n).tolist()
    price = rng.uniform(5.0, 500.0, size=n).round(2).tolist()
    category = rng.choice(categories, size=n).tolist()
    customer_id = rng.integers(1000, 9999, size=n).tolist()
    order_dates = pd.date_range("2023-01-01", periods=n, freq="2h")

    clean = pd.DataFrame({
        "order_id": order_ids,
        "product_id": product_ids,
        "customer_id": customer_id,
        "category": category,
        "quantity": quantity,
        "unit_price": price,
        "order_date": order_dates,
        "total_value": [q * p for q, p in zip(quantity, price)],
    })

    dirty = clean.copy()

    # Wrong dtypes: quantity as string, price as string with $, date as string
    dirty["quantity"] = dirty["quantity"].astype(str)
    dirty["unit_price"] = dirty["unit_price"].apply(lambda x: f"${x:.2f}")
    dirty["order_date"] = dirty["order_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    dirty["customer_id"] = dirty["customer_id"].astype(str)

    # Inject duplicates (15 exact duplicate rows)
    dup_idx = rng.choice(n, size=15, replace=False)
    dup_rows = dirty.iloc[dup_idx].copy()
    dirty = pd.concat([dirty, dup_rows], ignore_index=True)

    # Inject missing values
    missing_qty = rng.choice(len(dirty), size=10, replace=False)
    missing_price = rng.choice(len(dirty), size=8, replace=False)
    dirty.loc[missing_qty, "quantity"] = np.nan
    dirty.loc[missing_price, "unit_price"] = np.nan

    dirty = dirty.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Clean: remove dups, fix types, fill missing
    clean_out = dirty.copy()
    clean_out = clean_out.drop_duplicates(subset=["order_id"])
    clean_out["quantity"] = pd.to_numeric(clean_out["quantity"], errors="coerce")
    clean_out["unit_price"] = clean_out["unit_price"].astype(str).str.replace("$", "", regex=False)
    clean_out["unit_price"] = pd.to_numeric(clean_out["unit_price"], errors="coerce")
    clean_out["order_date"] = pd.to_datetime(clean_out["order_date"], errors="coerce")
    clean_out["customer_id"] = pd.to_numeric(clean_out["customer_id"], errors="coerce").astype("Int64")
    clean_out["quantity"] = clean_out["quantity"].fillna(clean_out["quantity"].median())
    clean_out["unit_price"] = clean_out["unit_price"].fillna(clean_out["unit_price"].median())
    clean_out = clean_out.reset_index(drop=True)

    return dirty, clean_out


# ---------------------------------------------------------------------------
# Task 3 — Hard: Full Pipeline (Nulls + Types + Dups + Outliers + Schema + Format)
# ---------------------------------------------------------------------------

def generate_task3_dataset(seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Healthcare Patient Records dataset — full pipeline challenge.
    Returns (dirty_df, clean_df).
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)
    n = 300

    blood_types = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]
    genders = ["Male", "Female", "Other"]

    patient_ids = [f"PAT{i:05d}" for i in range(n)]
    age = rng.integers(18, 90, size=n).tolist()
    gender = rng.choice(genders, size=n).tolist()
    blood_type = rng.choice(blood_types, size=n).tolist()
    systolic_bp = rng.integers(90, 180, size=n).tolist()
    diastolic_bp = rng.integers(60, 120, size=n).tolist()
    bmi = rng.uniform(16.0, 45.0, size=n).round(1).tolist()
    cholesterol = rng.integers(120, 300, size=n).tolist()
    glucose = rng.integers(70, 200, size=n).tolist()
    admission_dates = pd.date_range("2022-01-01", periods=n, freq="1D")
    diagnosis_codes = [f"ICD{rng.integers(100, 999)}" for _ in range(n)]

    clean = pd.DataFrame({
        "patient_id": patient_ids,
        "age": age,
        "gender": gender,
        "blood_type": blood_type,
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "bmi": bmi,
        "cholesterol": cholesterol,
        "glucose": glucose,
        "admission_date": admission_dates,
        "diagnosis_code": diagnosis_codes,
    })

    dirty = clean.copy()

    # 1. Wrong types
    dirty["age"] = dirty["age"].astype(str)
    dirty["admission_date"] = dirty["admission_date"].dt.strftime("%d/%m/%Y")
    dirty["systolic_bp"] = dirty["systolic_bp"].astype(str) + " mmHg"
    dirty["bmi"] = dirty["bmi"].astype(str)

    # 2. Inject extreme outliers
    outlier_bp_idx = rng.choice(n, size=8, replace=False)
    outlier_bmi_idx = rng.choice(n, size=6, replace=False)
    dirty.loc[outlier_bp_idx, "systolic_bp"] = [str(v) + " mmHg" for v in [350, 400, 500, 380, 420, 390, 410, 370]]
    dirty.loc[outlier_bmi_idx, "bmi"] = ["150.0", "200.0", "180.0", "0.5", "1.2", "0.3"]

    # 3. Missing values
    missing_age = rng.choice(n, size=20, replace=False)
    missing_gender = rng.choice(n, size=15, replace=False)
    missing_bmi = rng.choice(n, size=12, replace=False)
    missing_chol = rng.choice(n, size=10, replace=False)
    missing_glucose = rng.choice(n, size=8, replace=False)
    dirty.loc[missing_age, "age"] = np.nan
    dirty.loc[missing_gender, "gender"] = np.nan
    dirty.loc[missing_bmi, "bmi"] = np.nan
    dirty.loc[missing_chol, "cholesterol"] = np.nan
    dirty.loc[missing_glucose, "glucose"] = np.nan

    # 4. Duplicates
    dup_idx = rng.choice(n, size=20, replace=False)
    dup_rows = dirty.iloc[dup_idx].copy()
    dirty = pd.concat([dirty, dup_rows], ignore_index=True)

    # 5. Inconsistent string formats
    dirty["gender"] = dirty["gender"].apply(
        lambda x: x.upper() if isinstance(x, str) and rng.random() > 0.5 else x
    )
    dirty["diagnosis_code"] = dirty["diagnosis_code"].apply(
        lambda x: x.lower() if rng.random() > 0.6 else x
    )
    dirty["blood_type"] = dirty["blood_type"].apply(
        lambda x: " " + x + " " if rng.random() > 0.5 else x
    )

    dirty = dirty.sample(frac=1, random_state=seed).reset_index(drop=True)

    # --- Build reference clean ---
    c = dirty.copy()
    # Fix types
    c["systolic_bp"] = c["systolic_bp"].astype(str).str.replace(r"\s*mmHg", "", regex=True)
    c["age"] = pd.to_numeric(c["age"], errors="coerce")
    c["systolic_bp"] = pd.to_numeric(c["systolic_bp"], errors="coerce")
    c["bmi"] = pd.to_numeric(c["bmi"], errors="coerce")
    c["admission_date"] = pd.to_datetime(c["admission_date"], dayfirst=True, errors="coerce")
    # Remove dups
    c = c.drop_duplicates(subset=["patient_id"])
    # Fill missing
    c["age"] = c["age"].fillna(c["age"].median())
    c["gender"] = c["gender"].fillna(c["gender"].mode()[0])
    c["bmi"] = c["bmi"].fillna(c["bmi"].median())
    c["cholesterol"] = c["cholesterol"].fillna(c["cholesterol"].median())
    c["glucose"] = c["glucose"].fillna(c["glucose"].median())
    # Remove outliers (IQR for systolic_bp, bmi)
    for col in ["systolic_bp", "bmi"]:
        if c[col].dtype in [np.float64, np.int64, float, int]:
            q1, q3 = c[col].quantile(0.25), c[col].quantile(0.75)
            iqr = q3 - q1
            c = c[(c[col] >= q1 - 1.5 * iqr) & (c[col] <= q3 + 1.5 * iqr)]
    # Standardize formats
    c["gender"] = c["gender"].str.strip().str.title()
    c["blood_type"] = c["blood_type"].str.strip()
    c["diagnosis_code"] = c["diagnosis_code"].str.upper()
    c = c.reset_index(drop=True)

    return dirty, c
