"""
OpenEnv typed models for DataClean Environment.
Full pydantic v2 spec — with a minimal fallback for test environments.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union

try:
    from pydantic import BaseModel, Field
    PYDANTIC_AVAILABLE = True

    class FillMissingAction(BaseModel):
        action_type: str = "fill_missing"
        column: str
        strategy: str  # mean|median|mode|constant|ffill|bfill|drop
        value: Optional[Any] = None

    class FixDtypeAction(BaseModel):
        action_type: str = "fix_dtype"
        column: str
        target_type: str  # int|float|str|bool|datetime
        datetime_format: Optional[str] = None

    class RemoveDuplicatesAction(BaseModel):
        action_type: str = "remove_duplicates"
        subset: Optional[List[str]] = None
        keep: str = "first"

    class RemoveOutliersAction(BaseModel):
        action_type: str = "remove_outliers"
        column: str
        method: str = "iqr"
        threshold: float = 1.5

    class RenameColumnAction(BaseModel):
        action_type: str = "rename_column"
        old_name: str
        new_name: str

    class StandardizeFormatAction(BaseModel):
        action_type: str = "standardize_format"
        column: str
        format_type: str  # lowercase|uppercase|titlecase|strip|strip_special

    class FilterRowsAction(BaseModel):
        action_type: str = "filter_rows"
        column: str
        operator: str
        value: Any

    class ValidateSchemaAction(BaseModel):
        action_type: str = "validate_schema"
        expected_columns: List[str]
        expected_dtypes: Dict[str, str] = {}

    class SubmitAction(BaseModel):
        action_type: str = "submit"
        message: Optional[str] = None

    class ColumnProfile(BaseModel):
        name: str
        dtype: str
        null_count: int
        null_pct: float
        unique_count: int
        sample_values: List[Any]
        min: Optional[float] = None
        max: Optional[float] = None
        mean: Optional[float] = None
        has_outliers: Optional[bool] = None

    class DatasetProfile(BaseModel):
        row_count: int
        col_count: int
        total_nulls: int
        total_null_pct: float
        duplicate_row_count: int
        columns: List[ColumnProfile]

    class Observation(BaseModel):
        task_id: str
        task_description: str
        step_number: int
        max_steps: int
        dataset_profile: DatasetProfile
        action_history: List[Dict[str, Any]] = []
        current_score: float
        issues_remaining: List[str] = []
        schema_requirements: Optional[Dict[str, Any]] = None
        hints: List[str] = []

    class RewardBreakdown(BaseModel):
        null_score: float
        dtype_score: float
        duplicate_score: float
        outlier_score: float
        schema_score: float
        format_score: float
        efficiency_penalty: float

    class Reward(BaseModel):
        total: float
        breakdown: RewardBreakdown
        done: bool
        info: Dict[str, Any] = {}

except ImportError:
    PYDANTIC_AVAILABLE = False

    # -----------------------------------------------------------------------
    # Minimal fallback — pure Python dataclass-style models
    # -----------------------------------------------------------------------

    class _Base:
        def model_dump(self) -> Dict[str, Any]:
            result = {}
            for k, v in self.__dict__.items():
                if hasattr(v, "model_dump"):
                    result[k] = v.model_dump()
                elif isinstance(v, list):
                    result[k] = [
                        x.model_dump() if hasattr(x, "model_dump") else x
                        for x in v
                    ]
                else:
                    result[k] = v
            return result

    class FillMissingAction(_Base):
        def __init__(self, column: str, strategy: str, value=None, action_type="fill_missing"):
            self.action_type = action_type
            self.column = column
            self.strategy = strategy
            self.value = value

    class FixDtypeAction(_Base):
        def __init__(self, column: str, target_type: str, datetime_format=None, action_type="fix_dtype"):
            self.action_type = action_type
            self.column = column
            self.target_type = target_type
            self.datetime_format = datetime_format

    class RemoveDuplicatesAction(_Base):
        def __init__(self, subset=None, keep="first", action_type="remove_duplicates"):
            self.action_type = action_type
            self.subset = subset
            self.keep = keep

    class RemoveOutliersAction(_Base):
        def __init__(self, column: str, method: str = "iqr", threshold: float = 1.5, action_type="remove_outliers"):
            self.action_type = action_type
            self.column = column
            self.method = method
            self.threshold = threshold

    class RenameColumnAction(_Base):
        def __init__(self, old_name: str, new_name: str, action_type="rename_column"):
            self.action_type = action_type
            self.old_name = old_name
            self.new_name = new_name

    class StandardizeFormatAction(_Base):
        def __init__(self, column: str, format_type: str, action_type="standardize_format"):
            self.action_type = action_type
            self.column = column
            self.format_type = format_type

    class FilterRowsAction(_Base):
        def __init__(self, column: str, operator: str, value, action_type="filter_rows"):
            self.action_type = action_type
            self.column = column
            self.operator = operator
            self.value = value

    class ValidateSchemaAction(_Base):
        def __init__(self, expected_columns, expected_dtypes=None, action_type="validate_schema"):
            self.action_type = action_type
            self.expected_columns = expected_columns
            self.expected_dtypes = expected_dtypes or {}

    class SubmitAction(_Base):
        def __init__(self, message=None, action_type="submit"):
            self.action_type = action_type
            self.message = message

    class ColumnProfile(_Base):
        def __init__(self, name, dtype, null_count, null_pct, unique_count,
                     sample_values, min=None, max=None, mean=None, has_outliers=None):
            self.name = name
            self.dtype = dtype
            self.null_count = null_count
            self.null_pct = null_pct
            self.unique_count = unique_count
            self.sample_values = sample_values
            self.min = min
            self.max = max
            self.mean = mean
            self.has_outliers = has_outliers

    class DatasetProfile(_Base):
        def __init__(self, row_count, col_count, total_nulls, total_null_pct,
                     duplicate_row_count, columns):
            self.row_count = row_count
            self.col_count = col_count
            self.total_nulls = total_nulls
            self.total_null_pct = total_null_pct
            self.duplicate_row_count = duplicate_row_count
            self.columns = columns

    class Observation(_Base):
        def __init__(self, task_id, task_description, step_number, max_steps,
                     dataset_profile, current_score, action_history=None,
                     issues_remaining=None, schema_requirements=None, hints=None):
            self.task_id = task_id
            self.task_description = task_description
            self.step_number = step_number
            self.max_steps = max_steps
            self.dataset_profile = dataset_profile
            self.current_score = current_score
            self.action_history = action_history or []
            self.issues_remaining = issues_remaining or []
            self.schema_requirements = schema_requirements
            self.hints = hints or []

    class RewardBreakdown(_Base):
        def __init__(self, null_score, dtype_score, duplicate_score,
                     outlier_score, schema_score, format_score, efficiency_penalty):
            self.null_score = null_score
            self.dtype_score = dtype_score
            self.duplicate_score = duplicate_score
            self.outlier_score = outlier_score
            self.schema_score = schema_score
            self.format_score = format_score
            self.efficiency_penalty = efficiency_penalty

    class Reward(_Base):
        def __init__(self, total, breakdown, done, info=None):
            self.total = total
            self.breakdown = breakdown
            self.done = done
            self.info = info or {}


# Convenience union type annotation (runtime not enforced without pydantic)
Action = Union[
    FillMissingAction, FixDtypeAction, RemoveDuplicatesAction,
    RemoveOutliersAction, RenameColumnAction, StandardizeFormatAction,
    FilterRowsAction, ValidateSchemaAction, SubmitAction,
]
