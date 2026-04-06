"""
DataClean OpenEnv Environment
Real-world data cleaning agent environment with 3 progressive tasks.
"""

from __future__ import annotations
import copy
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from environment.models import (
    Action, Observation, Reward, DatasetProfile, ColumnProfile,
    FillMissingAction, FixDtypeAction, RemoveDuplicatesAction, RemoveOutliersAction,
    RenameColumnAction, StandardizeFormatAction, FilterRowsAction,
    ValidateSchemaAction, SubmitAction,
)
from environment.datasets import (
    generate_task1_dataset,
    generate_task2_dataset,
    generate_task3_dataset,
)
from graders.graders import grade_task1, grade_task2, grade_task3


TASK_CONFIG = {
    "task1": {
        "description": (
            "EASY — HR Employee Dataset: Fix all missing values in the dataset. "
            "The columns 'salary', 'department', and 'tenure_years' contain missing values "
            "that must be filled using appropriate imputation strategies. "
            "Use median for numeric columns and mode for categorical columns."
        ),
        "max_steps": 15,
        "generator": generate_task1_dataset,
        "grader": grade_task1,
        "hints": [
            "Use 'fill_missing' action with strategy='median' for 'salary' and 'tenure_years'",
            "Use 'fill_missing' action with strategy='mode' for 'department'",
            "Submit when all missing values are filled",
        ],
        "schema_requirements": None,
    },
    "task2": {
        "description": (
            "MEDIUM — E-commerce Orders Dataset: Fix wrong data types, remove duplicate rows, "
            "and handle missing values. "
            "Columns to fix: 'quantity' (str→int), 'unit_price' (str with $ → float), "
            "'order_date' (str → datetime), 'customer_id' (str → int). "
            "Remove duplicate orders (by 'order_id'). Fill missing numeric values with median."
        ),
        "max_steps": 25,
        "generator": generate_task2_dataset,
        "grader": grade_task2,
        "hints": [
            "Fix dtypes first: quantity→int, unit_price→float (strip $), order_date→datetime",
            "Then remove duplicates using subset=['order_id']",
            "Finally fill any remaining missing values",
        ],
        "schema_requirements": {
            "quantity": "int",
            "unit_price": "float",
            "order_date": "datetime",
            "customer_id": "int",
        },
    },
    "task3": {
        "description": (
            "HARD — Healthcare Patient Records: Complete full data cleaning pipeline. "
            "Step 1: Fix data types (age, bmi, systolic_bp → numeric; admission_date → datetime from DD/MM/YYYY). "
            "Step 2: Remove 'mmHg' from systolic_bp string values before numeric conversion. "
            "Step 3: Remove duplicate patients (by 'patient_id'). "
            "Step 4: Fill missing values (age, bmi → median; gender → mode; cholesterol, glucose → median). "
            "Step 5: Remove outliers in 'systolic_bp' and 'bmi' using IQR method. "
            "Step 6: Standardize formats — gender→titlecase, blood_type→strip whitespace, diagnosis_code→uppercase."
        ),
        "max_steps": 35,
        "generator": generate_task3_dataset,
        "grader": grade_task3,
        "hints": [
            "Fix systolic_bp: standardize_format strip_special first, then fix_dtype to float",
            "Use datetime_format='%d/%m/%Y' for admission_date",
            "Remove outliers with method='iqr', threshold=1.5 for systolic_bp and bmi",
            "Standardize: gender→titlecase, blood_type→strip, diagnosis_code→uppercase",
        ],
        "schema_requirements": {
            "patient_id": "str",
            "age": "numeric",
            "gender": "str",
            "systolic_bp": "numeric",
            "bmi": "numeric",
            "admission_date": "datetime",
        },
    },
}


class DataCleanEnvironment:
    """
    OpenEnv-compatible environment for data cleaning tasks.
    Simulates real-world data quality work done by data engineers and analysts.
    """

    def __init__(self, task_id: str = "task1", seed: int = 42):
        assert task_id in TASK_CONFIG, f"Unknown task: {task_id}. Choose from {list(TASK_CONFIG.keys())}"
        self.task_id = task_id
        self.seed = seed
        self._config = TASK_CONFIG[task_id]
        self._max_steps = self._config["max_steps"]

        # These get populated on reset()
        self._df: pd.DataFrame = pd.DataFrame()
        self._original_df: pd.DataFrame = pd.DataFrame()
        self._step_num: int = 0
        self._done: bool = False
        self._action_history: List[Dict[str, Any]] = []
        self._current_reward: float = 0.0

        self.reset()

    # -----------------------------------------------------------------------
    # OpenEnv Core API
    # -----------------------------------------------------------------------
   def reset(self) -> Observation:
        """Reset environment to initial state, return first observation."""
        dirty, _ = self._config["generator"](seed=self.seed)
        self._df = dirty.copy()
        self._original_df = dirty.copy()
        self._step_num = 0
        self._done = False
        self._action_history = []
        self._current_reward = 0.0
        print(f"START: Task {self.task_id} initialized with seed {self.seed}")
        return self._build_observation()

    def step(self, action: Action | Dict[str, Any]) -> Tuple[Observation, float, bool, Dict]:
        """
        Apply action to dataset, return (observation, reward, done, info).
        Implements full OpenEnv step() contract.
        """
        if self._done:
            # Return final state gracefully instead of raising
            obs = self._build_observation()
            return obs, self._current_reward, True, {"reason": "already_done"}

        # Accept dict or model
        if isinstance(action, dict):
            action_type = action.get("action_type", "unknown") # Get name for printing
            action = self._parse_action(action)
        else:
            action_type = action.action_type

        self._step_num += 1
        
        # --- ADDED FOR CHECKLIST #5 ---
        print(f"STEP {self._step_num}: Executing {action_type}")
        # ------------------------------

        action_result = {"step": self._step_num, "action": action.model_dump(), "success": False, "message": ""}

        try:
            if isinstance(action, SubmitAction):
                action_result["success"] = True
                action_result["message"] = "Episode submitted for grading."
                self._done = True
                
                # --- ADDED FOR CHECKLIST #5 ---
                print(f"END: Task {self.task_id} completed at step {self._step_num}")
                # ------------------------------
            else:
                self._apply_action(action, action_result)
        except Exception as e:
            action_result["success"] = False
            action_result["message"] = f"Action failed: {str(e)}"

        self._action_history.append(action_result)

        # Grade current state
        reward_obj: Reward = self._config["grader"](
            df=self._df,
            original_df=self._original_df,
            n_steps=self._step_num,
            max_steps=self._max_steps,
        )
        self._current_reward = reward_obj.total

        # Force done if max steps reached
        if self._step_num >= self._max_steps:
            self._done = True
            
            # --- ADDED FOR CHECKLIST #5 ---
            print(f"END: Task {self.task_id} reached max steps ({self._max_steps})")
            # ------------------------------
            
            reward_obj = Reward(
                total=reward_obj.total,
                breakdown=reward_obj.breakdown,
                done=True,
                info={**reward_obj.info, "reason": "max_steps_reached"},
            )
        elif reward_obj.done:
            self._done = True
            # --- ADDED FOR CHECKLIST #5 ---
            print(f"END: Task {self.task_id} succeeded early")
            # ------------------------------

        obs = self._build_observation()
        return obs, reward_obj.total, self._done, reward_obj.info

  
    def state(self) -> Dict[str, Any]:
        """Return full current state (for checkpointing/inspection)."""
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "step_number": self._step_num,
            "max_steps": self._max_steps,
            "done": self._done,
            "current_reward": self._current_reward,
            "df_shape": list(self._df.shape),
            "df_columns": list(self._df.columns),
            "df_dtypes": {col: str(dtype) for col, dtype in self._df.dtypes.items()},
            "null_counts": self._df.isnull().sum().to_dict(),
            "duplicate_count": int(self._df.duplicated().sum()),
            "action_history": self._action_history,
            "dataset_preview": self._df.head(5).to_dict(orient="records"),
        }

    # -----------------------------------------------------------------------
    # Action Execution
    # -----------------------------------------------------------------------

    def _apply_action(self, action: Action, result: Dict):
        """Dispatch and apply typed action to self._df."""
        df = self._df

        if isinstance(action, FillMissingAction):
            col = action.column
            self._assert_column_exists(col)
            before = df[col].isna().sum()

            if action.strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif action.strategy == "median":
                df[col] = df[col].fillna(df[col].median())
            elif action.strategy == "mode":
                df[col] = df[col].fillna(df[col].mode()[0])
            elif action.strategy == "constant":
                if action.value is None:
                    raise ValueError("strategy='constant' requires a value")
                df[col] = df[col].fillna(action.value)
            elif action.strategy == "ffill":
                df[col] = df[col].ffill()
            elif action.strategy == "bfill":
                df[col] = df[col].bfill()
            elif action.strategy == "drop":
                df.dropna(subset=[col], inplace=True)
                df.reset_index(drop=True, inplace=True)

            after = df[col].isna().sum()
            result["success"] = True
            result["message"] = f"Filled {before - after} missing values in '{col}' using {action.strategy}"

        elif isinstance(action, FixDtypeAction):
            col = action.column
            self._assert_column_exists(col)

            if action.target_type == "int":
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].fillna(df[col].median())
            elif action.target_type == "float":
                # Strip common currency/unit symbols
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(r"[\$,£€\s]", "", regex=True)
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif action.target_type == "str":
                df[col] = df[col].astype(str)
            elif action.target_type == "bool":
                df[col] = df[col].astype(bool)
            elif action.target_type == "datetime":
                fmt = action.datetime_format
                if fmt:
                    df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")
                else:
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")

            result["success"] = True
            result["message"] = f"Converted '{col}' to {action.target_type}. New dtype: {df[col].dtype}"

        elif isinstance(action, RemoveDuplicatesAction):
            before = len(df)
            keep = None if action.keep == "none" else action.keep
            df.drop_duplicates(subset=action.subset, keep=keep, inplace=True)
            df.reset_index(drop=True, inplace=True)
            after = len(df)
            result["success"] = True
            result["message"] = f"Removed {before - after} duplicate rows. Rows: {before} → {after}"

        elif isinstance(action, RemoveOutliersAction):
            col = action.column
            self._assert_column_exists(col)
            numeric_series = pd.to_numeric(df[col], errors="coerce")
            before = len(df)

            if action.method == "iqr":
                q1 = numeric_series.quantile(0.25)
                q3 = numeric_series.quantile(0.75)
                iqr = q3 - q1
                mask = (numeric_series >= q1 - action.threshold * iqr) & \
                       (numeric_series <= q3 + action.threshold * iqr)
                df = df[mask | numeric_series.isna()].reset_index(drop=True)
            elif action.method == "zscore":
                mean, std = numeric_series.mean(), numeric_series.std()
                z = (numeric_series - mean) / std
                mask = z.abs() <= action.threshold
                df = df[mask | numeric_series.isna()].reset_index(drop=True)
            elif action.method == "clip":
                q1 = numeric_series.quantile(0.25)
                q3 = numeric_series.quantile(0.75)
                iqr = q3 - q1
                df[col] = numeric_series.clip(
                    lower=q1 - action.threshold * iqr,
                    upper=q3 + action.threshold * iqr
                )

            after = len(df)
            self._df = df
            result["success"] = True
            result["message"] = f"Outlier removal on '{col}': {before - after} rows removed (method={action.method})"
            return  # _df already updated above

        elif isinstance(action, RenameColumnAction):
            if action.old_name not in df.columns:
                raise ValueError(f"Column '{action.old_name}' not found")
            df.rename(columns={action.old_name: action.new_name}, inplace=True)
            result["success"] = True
            result["message"] = f"Renamed '{action.old_name}' → '{action.new_name}'"

        elif isinstance(action, StandardizeFormatAction):
            col = action.column
            self._assert_column_exists(col)
            series = df[col].astype(str)

            if action.format_type == "lowercase":
                df[col] = series.str.lower()
            elif action.format_type == "uppercase":
                df[col] = series.str.upper()
            elif action.format_type == "titlecase":
                df[col] = series.str.title()
            elif action.format_type == "strip":
                df[col] = series.str.strip()
            elif action.format_type == "strip_special":
                df[col] = series.str.replace(r"[^\w\s\.\-\+]", "", regex=True).str.strip()

            # Restore NaN where original was null
            orig_nulls = self._df[col].isna()
            df.loc[orig_nulls, col] = np.nan
            result["success"] = True
            result["message"] = f"Applied {action.format_type} to '{col}'"

        elif isinstance(action, FilterRowsAction):
            col = action.column
            self._assert_column_exists(col)
            before = len(df)
            val = action.value

            op = action.operator
            if op == "eq":
                mask = df[col] == val
            elif op == "ne":
                mask = df[col] != val
            elif op == "gt":
                mask = pd.to_numeric(df[col], errors="coerce") > float(val)
            elif op == "lt":
                mask = pd.to_numeric(df[col], errors="coerce") < float(val)
            elif op == "gte":
                mask = pd.to_numeric(df[col], errors="coerce") >= float(val)
            elif op == "lte":
                mask = pd.to_numeric(df[col], errors="coerce") <= float(val)
            elif op == "isin":
                mask = df[col].isin(val if isinstance(val, list) else [val])
            elif op == "notin":
                mask = ~df[col].isin(val if isinstance(val, list) else [val])
            elif op == "contains":
                mask = df[col].astype(str).str.contains(str(val), na=False)
            else:
                raise ValueError(f"Unknown operator: {op}")

            df = df[mask].reset_index(drop=True)
            self._df = df
            result["success"] = True
            result["message"] = f"Filtered '{col}' ({op} {val}): {before} → {len(df)} rows"
            return

        elif isinstance(action, ValidateSchemaAction):
            missing_cols = [c for c in action.expected_columns if c not in df.columns]
            dtype_mismatches = {}
            for col, expected in action.expected_dtypes.items():
                if col in df.columns:
                    actual = str(df[col].dtype)
                    if expected not in actual and not (
                        expected == "numeric" and pd.api.types.is_numeric_dtype(df[col])
                    ):
                        dtype_mismatches[col] = {"expected": expected, "actual": actual}

            result["success"] = True
            result["message"] = (
                f"Schema validation: missing_cols={missing_cols}, "
                f"dtype_mismatches={dtype_mismatches}"
            )

        self._df = df

    def _assert_column_exists(self, col: str):
        if col not in self._df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(self._df.columns)}")

    # -----------------------------------------------------------------------
    # Observation Building
    # -----------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        df = self._df
        issues = self._detect_issues()
        profile = self._build_profile(df)

        return Observation(
            task_id=self.task_id,
            task_description=self._config["description"],
            step_number=self._step_num,
            max_steps=self._max_steps,
            dataset_profile=profile,
            action_history=self._action_history[-10:],  # last 10 actions
            current_score=self._current_reward,
            issues_remaining=issues,
            schema_requirements=self._config.get("schema_requirements"),
            hints=self._config["hints"],
        )

    def _build_profile(self, df: pd.DataFrame) -> DatasetProfile:
        columns = []
        for col in df.columns:
            series = df[col]
            null_count = int(series.isna().sum())
            profile = ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                null_count=null_count,
                null_pct=round(null_count / max(len(df), 1), 4),
                unique_count=int(series.nunique()),
                sample_values=series.dropna().head(3).tolist(),
            )
            if pd.api.types.is_numeric_dtype(series):
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                if len(numeric) > 0:
                    profile.min = float(numeric.min())
                    profile.max = float(numeric.max())
                    profile.mean = round(float(numeric.mean()), 2)
                    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
                    iqr = q3 - q1
                    profile.has_outliers = bool(
                        ((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).any()
                    )
            columns.append(profile)

        return DatasetProfile(
            row_count=len(df),
            col_count=len(df.columns),
            total_nulls=int(df.isnull().sum().sum()),
            total_null_pct=round(df.isnull().sum().sum() / max(df.size, 1), 4),
            duplicate_row_count=int(df.duplicated().sum()),
            columns=columns,
        )

    def _detect_issues(self) -> List[str]:
        df = self._df
        issues = []

        total_nulls = df.isnull().sum().sum()
        if total_nulls > 0:
            null_cols = df.columns[df.isnull().any()].tolist()
            issues.append(f"Missing values: {total_nulls} nulls across columns {null_cols}")

        dups = df.duplicated().sum()
        if dups > 0:
            issues.append(f"Duplicate rows: {dups} duplicate rows detected")

        for col in df.columns:
            if df[col].dtype == object:
                numeric_attempt = pd.to_numeric(
                    df[col].astype(str).str.replace(r"[\$,£€\s]", "", regex=True),
                    errors="coerce"
                )
                valid_pct = numeric_attempt.notna().mean()
                if valid_pct > 0.7 and df[col].dtype == object:
                    issues.append(f"Possible dtype issue: '{col}' looks numeric but stored as string")

        for col in df.select_dtypes(include=[np.number]).columns:
            series = df[col].dropna()
            if len(series) > 10:
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                outliers = ((series < q1 - 1.5 * iqr) | (series > q3 + 1.5 * iqr)).sum()
                if outliers > 0:
                    issues.append(f"Outliers: {outliers} outliers in '{col}'")

        for col in df.select_dtypes(include=[object]).columns:
            series = df[col].dropna().astype(str)
            if len(series) > 0:
                has_leading_trailing = (series != series.str.strip()).any()
                if has_leading_trailing:
                    issues.append(f"Format issue: '{col}' has leading/trailing whitespace")

                unique_cases = series.str.lower().nunique()
                actual_unique = series.nunique()
                if actual_unique > unique_cases:
                    issues.append(f"Format issue: '{col}' has inconsistent casing")

        return issues

    def _parse_action(self, d: Dict[str, Any]) -> Action:
        """Parse dict into typed Action model."""
        action_type = d.get("action_type", "")
        type_map = {
            "fill_missing": FillMissingAction,
            "fix_dtype": FixDtypeAction,
            "remove_duplicates": RemoveDuplicatesAction,
            "remove_outliers": RemoveOutliersAction,
            "rename_column": RenameColumnAction,
            "standardize_format": StandardizeFormatAction,
            "filter_rows": FilterRowsAction,
            "validate_schema": ValidateSchemaAction,
            "submit": SubmitAction,
        }
        cls = type_map.get(action_type)
        if cls is None:
            raise ValueError(f"Unknown action_type '{action_type}'. Valid: {list(type_map.keys())}")
        return cls(**d)
