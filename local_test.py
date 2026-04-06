"""
local_test.py — Runs the full DataClean environment locally without a server.
Use this to validate everything works before deploying.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment.env import DataCleanEnvironment, TASK_CONFIG


def run_optimal_task1():
    env = DataCleanEnvironment("task1", seed=42)
    obs = env.reset()
    print(f"  Initial nulls: {obs.dataset_profile.total_nulls}")
    actions = [
        {"action_type": "fill_missing", "column": "salary", "strategy": "median"},
        {"action_type": "fill_missing", "column": "department", "strategy": "mode"},
        {"action_type": "fill_missing", "column": "tenure_years", "strategy": "median"},
        {"action_type": "submit"},
    ]
    rewards = []
    for a in actions:
        _, r, done, info = env.step(a)
        rewards.append(r)
    print(f"  Reward trajectory: {[round(r, 3) for r in rewards]}")
    print(f"  Final score: {rewards[-2]:.4f}")
    return rewards[-2]


def run_optimal_task2():
    env = DataCleanEnvironment("task2", seed=42)
    obs = env.reset()
    print(f"  Initial shape: {obs.dataset_profile.row_count}x{obs.dataset_profile.col_count}, dups={obs.dataset_profile.duplicate_row_count}")
    actions = [
        {"action_type": "fix_dtype", "column": "quantity", "target_type": "int"},
        {"action_type": "fix_dtype", "column": "unit_price", "target_type": "float"},
        {"action_type": "fix_dtype", "column": "order_date", "target_type": "datetime"},
        {"action_type": "fix_dtype", "column": "customer_id", "target_type": "int"},
        {"action_type": "remove_duplicates", "subset": ["order_id"], "keep": "first"},
        {"action_type": "fill_missing", "column": "quantity", "strategy": "median"},
        {"action_type": "fill_missing", "column": "unit_price", "strategy": "median"},
        {"action_type": "submit"},
    ]
    rewards = []
    for a in actions:
        _, r, done, info = env.step(a)
        rewards.append(r)
    print(f"  Reward trajectory: {[round(r, 3) for r in rewards]}")
    print(f"  Final score: {rewards[-2]:.4f}")
    return rewards[-2]


def run_optimal_task3():
    env = DataCleanEnvironment("task3", seed=42)
    obs = env.reset()
    print(f"  Initial: {obs.dataset_profile.row_count} rows, {obs.dataset_profile.total_nulls} nulls, {obs.dataset_profile.duplicate_row_count} dups")
    actions = [
        {"action_type": "standardize_format", "column": "systolic_bp", "format_type": "strip_special"},
        {"action_type": "fix_dtype", "column": "age", "target_type": "float"},
        {"action_type": "fix_dtype", "column": "systolic_bp", "target_type": "float"},
        {"action_type": "fix_dtype", "column": "bmi", "target_type": "float"},
        {"action_type": "fix_dtype", "column": "admission_date", "target_type": "datetime", "datetime_format": "%d/%m/%Y"},
        {"action_type": "remove_duplicates", "subset": ["patient_id"], "keep": "first"},
        {"action_type": "fill_missing", "column": "age", "strategy": "median"},
        {"action_type": "fill_missing", "column": "gender", "strategy": "mode"},
        {"action_type": "fill_missing", "column": "bmi", "strategy": "median"},
        {"action_type": "fill_missing", "column": "cholesterol", "strategy": "median"},
        {"action_type": "fill_missing", "column": "glucose", "strategy": "median"},
        {"action_type": "remove_outliers", "column": "systolic_bp", "method": "iqr", "threshold": 1.5},
        {"action_type": "remove_outliers", "column": "bmi", "method": "iqr", "threshold": 1.5},
        {"action_type": "standardize_format", "column": "gender", "format_type": "titlecase"},
        {"action_type": "standardize_format", "column": "blood_type", "format_type": "strip"},
        {"action_type": "standardize_format", "column": "diagnosis_code", "format_type": "uppercase"},
        {"action_type": "submit"},
    ]
    rewards = []
    for a in actions:
        _, r, done, info = env.step(a)
        rewards.append(r)
    print(f"  Reward trajectory: {[round(r, 3) for r in rewards]}")
    print(f"  Final score: {rewards[-2]:.4f}")
    return rewards[-2]


if __name__ == "__main__":
    print("=" * 60)
    print("DataClean OpenEnv — Local Validation")
    print("=" * 60)

    print("\n[TASK 1] Fill Missing Values (Easy)")
    s1 = run_optimal_task1()

    print("\n[TASK 2] Fix Dtypes + Dedup (Medium)")
    s2 = run_optimal_task2()

    print("\n[TASK 3] Full Pipeline (Hard)")
    s3 = run_optimal_task3()

    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"  task1 (easy):   {s1:.4f}")
    print(f"  task2 (medium): {s2:.4f}")
    print(f"  task3 (hard):   {s3:.4f}")
    print(f"  average:        {(s1+s2+s3)/3:.4f}")
    print("=" * 60)

    # Validate graders
    print("\n[GRADER VALIDATION]")
    from graders.graders import grade_task1, grade_task2, grade_task3
    from environment.datasets import generate_task1_dataset, generate_task2_dataset, generate_task3_dataset

    for fn, gen, name in [
        (grade_task1, generate_task1_dataset, "task1"),
        (grade_task2, generate_task2_dataset, "task2"),
        (grade_task3, generate_task3_dataset, "task3"),
    ]:
        dirty, _ = gen(seed=42)
        r = fn(dirty, dirty, n_steps=0, max_steps=20)
        assert 0.0 <= r.total <= 1.0, f"Score out of range: {r.total}"
        r2 = fn(dirty, dirty, n_steps=0, max_steps=20)
        assert r.total == r2.total, f"Non-deterministic: {r.total} != {r2.total}"
        print(f"  ✓ {name}: score={r.total:.4f}, deterministic=True")

    print("\nAll validations passed ✓")
