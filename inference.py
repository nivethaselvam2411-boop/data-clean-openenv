"""
inference.py — DataClean OpenEnv Baseline Agent

"""

from __future__ import annotations
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
# FIX: Default to dummy_key to prevent OpenAI library validation crash
HF_TOKEN = os.environ.get("HF_TOKEN", "dummy_key")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

if not os.environ.get("HF_TOKEN"):
    print("[WARN] HF_TOKEN not set — using dummy key", flush=True)

# FIX: Wrap in try/except to prevent Phase 2 "Unhandled Exception" failure
try:
    client = OpenAI(
        api_key=HF_TOKEN if HF_TOKEN else "dummy_key",
        base_url=API_BASE_URL,
    )
except Exception as e:
    print(f"[ERROR] OpenAI init failed: {e}", flush=True)
    client = None

TASKS = ["task1", "task2", "task3"]
SESSION_ID = f"inference_{int(time.time())}"

# ---------------------------------------------------------------------------
# Logging helpers — Fixed to strict [START] / [STEP] / [END] format
# ---------------------------------------------------------------------------

def log_start(task_id: str, model: str):
    # Changed from JSON to strict Tag format
    print(f"[START] Task: {task_id} | Model: {model}", flush=True)


def log_step(task_id: str, step: int, action: Dict, reward: float, done: bool, info: Dict):
    # Changed from JSON to strict Tag format
    action_type = action.get("action_type", "unknown")
    print(f"[STEP] {step}: {action_type} | Reward: {reward:.4f} | Done: {done}", flush=True)


def log_end(task_id: str, final_reward: float, total_steps: int, success: bool):
    # Changed from JSON to strict Tag format
    print(f"[END] Task: {task_id} | Final Reward: {final_reward:.4f} | Steps: {total_steps} | Success: {success}", flush=True)


# ---------------------------------------------------------------------------
# Environment client
# ---------------------------------------------------------------------------

def env_reset(task_id: str) -> Dict:
    r = requests.post(f"{ENV_URL}/reset", json={
        "task_id": task_id,
        "seed": 42,
        "session_id": SESSION_ID,
    }, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(task_id: str, action: Dict) -> Dict:
    r = requests.post(f"{ENV_URL}/step", json={
        "task_id": task_id,
        "action": action,
        "seed": 42,
        "session_id": SESSION_ID,
    }, timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# LLM Agent (YOUR ORIGINAL LOGIC - NO CHANGES)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert data cleaning agent. You will be given a dataset profile
and must clean it step by step using specific actions.
Available actions (always return valid JSON with action_type):
1. fill_missing: Fill null values
   {"action_type": "fill_missing", "column": "col_name", "strategy": "mean|median|mode|constant|ffill|bfill|drop", "value": null}
2. fix_dtype: Fix data types
   {"action_type": "fix_dtype", "column": "col_name", "target_type": "int|float|str|bool|datetime", "datetime_format": null}
3. remove_duplicates: Remove duplicate rows
   {"action_type": "remove_duplicates", "subset": ["col1", "col2"], "keep": "first|last|none"}
4. remove_outliers: Remove outliers
   {"action_type": "remove_outliers", "column": "col_name", "method": "iqr|zscore|clip", "threshold": 1.5}
5. standardize_format: Fix string formatting
   {"action_type": "standardize_format", "column": "col_name", "format_type": "lowercase|uppercase|titlecase|strip|strip_special"}
6. filter_rows: Filter rows by condition
   {"action_type": "filter_rows", "column": "col_name", "operator": "eq|ne|gt|lt|gte|lte|isin|notin|contains", "value": "val"}
7. rename_column: Rename a column
   {"action_type": "rename_column", "old_name": "old", "new_name": "new"}
8. validate_schema: Validate schema
   {"action_type": "validate_schema", "expected_columns": ["col1"], "expected_dtypes": {"col1": "int"}}
9. submit: Submit final cleaned dataset
   {"action_type": "submit", "message": "Done cleaning"}
RULES:
- Always return ONLY a JSON object with action_type and required fields
- Be systematic: fix dtypes first, then remove duplicates, then fill missing, then handle outliers
- Read the task description and issues_remaining carefully
- Submit when all issues are resolved or you're near the step limit
"""


def build_user_prompt(obs: Dict) -> str:
    profile = obs.get("dataset_profile", {})
    issues = obs.get("issues_remaining", [])
    hints = obs.get("hints", [])
    step = obs.get("step_number", 0)
    max_steps = obs.get("max_steps", 20)
    score = obs.get("current_score", 0.0)

    cols_summary = []
    for col in profile.get("columns", []):
        cols_summary.append(
            f"  - {col['name']}: dtype={col['dtype']}, nulls={col['null_count']} ({col['null_pct']*100:.1f}%), "
            f"unique={col['unique_count']}, sample={col['sample_values'][:2]}"
            + (f", outliers={col['has_outliers']}" if col.get('has_outliers') is not None else "")
        )

    history = obs.get("action_history", [])
    recent = history[-3:] if history else []
    history_str = "\n".join(
        f"  Step {h['step']}: {h['action'].get('action_type','?')} on "
        f"{h['action'].get('column', h['action'].get('subset', '?'))} "
        f"→ {'✓' if h['success'] else '✗'} {h.get('message','')[:80]}"
        for h in recent
    ) or "  (none yet)"

    return f"""TASK: {obs.get('task_description', '')}
CURRENT STATE:
  Step: {step}/{max_steps}
  Current Score: {score:.3f}
  Rows: {profile.get('row_count', '?')} | Cols: {profile.get('col_count', '?')}
  Total Nulls: {profile.get('total_nulls', '?')} ({profile.get('total_null_pct', 0)*100:.1f}%)
  Duplicate Rows: {profile.get('duplicate_row_count', '?')}
COLUMNS:
{chr(10).join(cols_summary)}
ISSUES REMAINING:
{chr(10).join(f'  • {issue}' for issue in issues) or '  ✓ No issues detected!'}
RECENT ACTIONS:
{history_str}
HINTS:
{chr(10).join(f'  → {h}' for h in hints)}
SCHEMA REQUIREMENTS: {json.dumps(obs.get('schema_requirements', {}), indent=2)}
Return your next action as a single JSON object. If all issues are resolved, submit.
"""


def get_agent_action(obs: Dict, conversation: List[Dict]) -> Dict:
    """Call LLM to get next action."""
    if client is None:
        return {"action_type": "submit", "message": "client_not_initialized"}
        
    user_msg = build_user_prompt(obs)
    conversation.append({"role": "user", "content": user_msg})

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation,
        temperature=0.1,
        max_tokens=300,
    )

    reply = response.choices[0].message.content.strip()
    conversation.append({"role": "assistant", "content": reply})

    if "```json" in reply:
        reply = reply.split("```json")[1].split("```")[0].strip()
    elif "```" in reply:
        reply = reply.split("```")[1].split("```")[0].strip()

    try:
        action = json.loads(reply)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{[^{}]+\}', reply, re.DOTALL)
        if match:
            action = json.loads(match.group())
        else:
            print(f"[WARN] Could not parse action: {reply[:100]}", flush=True)
            action = {"action_type": "submit", "message": "parse_error"}

    return action


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task_id: str) -> Dict:
    print(f"\n{'='*60}", flush=True)
    log_start(task_id, MODEL_NAME)

    obs = env_reset(task_id)
    max_steps = obs.get("max_steps", 20)
    final_reward = 0.0
    step = 0
    done = False
    conversation: List[Dict] = []

    while not done and step < max_steps:
        try:
            action = get_agent_action(obs, conversation)
        except Exception as e:
            print(f"[WARN] LLM call failed: {e}", flush=True)
            action = {"action_type": "submit", "message": "llm_error"}

        try:
            result = env_step(task_id, action)
        except Exception as e:
            print(f"[ERROR] env_step failed: {e}", flush=True)
            break

        obs = result.get("observation", obs)
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        info = result.get("info", {})
        step += 1
        final_reward = reward

        log_step(task_id, step, action, reward, done, info)

        if done:
            break

        time.sleep(0.5)

    log_end(task_id, final_reward, step, final_reward >= 0.7)
    return {"task_id": task_id, "final_reward": final_reward, "steps": step}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"DataClean OpenEnv Inference Script", flush=True)
    print(f"Model: {MODEL_NAME} | API: {API_BASE_URL} | Env: {ENV_URL}", flush=True)
    print(f"{'='*60}", flush=True)

    # FIX: Increased attempts and added more robust health check
    for attempt in range(15):
        try:
            r = requests.get(f"{ENV_URL}/health", timeout=2)
            if r.status_code == 200:
                print(f"Environment ready at {ENV_URL}", flush=True)
                break
        except Exception:
            pass
        print(f"Waiting for environment... ({attempt+1}/15)", flush=True)
        time.sleep(2)
    else:
        print("[ERROR] Environment not reachable. Exiting.", flush=True)
        sys.exit(1)

    results = []
    for task_id in TASKS:
        try:
            result = run_task(task_id)
            results.append(result)
        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}", flush=True)
            traceback.print_exc()
            results.append({"task_id": task_id, "final_reward": 0.0, "steps": 0})

    print(f"\n{'='*60}", flush=True)
    print("FINAL RESULTS:", flush=True)
    for r in results:
        grade = "A" if r["final_reward"] >= 0.85 else "B" if r["final_reward"] >= 0.70 else "C" if r["final_reward"] >= 0.50 else "F"
        print(f"  {r['task_id']}: reward={r['final_reward']:.4f} steps={r['steps']} grade={grade}", flush=True)

    avg = sum(r["final_reward"] for r in results) / len(results) if results else 0
    print(f"  AVERAGE: {avg:.4f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
