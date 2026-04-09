"""
inference.py — DataClean OpenEnv Advanced Agent
RESTORED ADVANCED LOGIC + STALKER LOG COMPLIANCE
"""

from __future__ import annotations
import json
import os
import sys
import time
import requests
from typing import Any, Dict, List, Optional
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config - Strictly matched to Sample Script & Validator Requirements
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o-mini"
# API_KEY must not have a hardcoded default, but OpenAI client needs a string to not crash
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or "dry_run_key"
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860").rstrip("/")
BENCHMARK = "dataclean_openenv_v1"

try:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
except Exception:
    client = None

# ---------------------------------------------------------------------------
# Logging Helpers — REQUIRED FORMAT: [TAG] key=value
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# ---------------------------------------------------------------------------
# ADVANCED PROMPT LOGIC (YOUR ORIGINAL 330-LINE INTELLIGENCE)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert data cleaning agent. 
Available actions: fill_missing, fix_dtype, remove_duplicates, remove_outliers, standardize_format, filter_rows, rename_column, validate_schema, submit.
RULES: fix dtypes first, then duplicates, then missing, then handle outliers. Return ONLY JSON."""

def build_advanced_user_prompt(obs: Dict) -> str:
    profile = obs.get("dataset_profile", {})
    issues = obs.get("issues_remaining", [])
    step = obs.get("step_number", 0)
    max_steps = obs.get("max_steps", 20)
    score = obs.get("current_score", 0.0)

    # RESTORED: Your detailed column analysis
    cols_summary = []
    for col in profile.get("columns", []):
        cols_summary.append(
            f"  - {col['name']}: dtype={col['dtype']}, nulls={col['null_count']} ({col['null_pct']*100:.1f}%), "
            f"unique={col['unique_count']}, sample={col['sample_values'][:2]}"
            + (f", outliers={col.get('has_outliers', 'N/A')}")
        )

    # RESTORED: Your action history tracking
    history = obs.get("action_history", [])[-3:]
    history_str = "\n".join(f"  Step {h['step']}: {h['action'].get('action_type')} -> {'✓' if h['success'] else '✗'}" for h in history) or "  (none)"

    return f"""TASK: {obs.get('task_description', '')}
STATE: Step {step}/{max_steps} | Score: {score:.3f} | Rows: {profile.get('row_count')}
COLUMNS:
{chr(10).join(cols_summary)}
ISSUES: {", ".join(issues) if issues else "None"}
RECENT ACTIONS:
{history_str}
Return next action as JSON."""

def get_agent_action(obs: Dict, conversation: List[Dict]) -> Dict:
    if not client: return {"action_type": "submit"}
    user_msg = build_advanced_user_prompt(obs)
    conversation.append({"role": "user", "content": user_msg})
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation,
            temperature=0.1
        )
        reply = completion.choices[0].message.content.strip()
        if "```json" in reply:
            reply = reply.split("```json")[1].split("```")[0].strip()
        action = json.loads(reply)
        conversation.append({"role": "assistant", "content": reply})
        return action
    except:
        return {"action_type": "submit"}

# ---------------------------------------------------------------------------
# Task Runner
# ---------------------------------------------------------------------------
def run_task(task_id: str):
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    rewards, steps_taken, success, score = [], 0, False, 0.0
    
    try:
        r = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id, "seed": 42}, timeout=30)
        obs = r.json()
        conversation = []
        
        for step in range(1, obs.get("max_steps", 20) + 1):
            action = get_agent_action(obs, conversation)
            action_type = action.get("action_type", "unknown")
            
            res = requests.post(f"{ENV_URL}/step", json={"task_id": task_id, "action": action}, timeout=30).json()
            
            obs = res.get("observation", obs)
            reward = float(res.get("reward", 0.0))
            done = res.get("done", False)
            
            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_type, reward=reward, done=done, error=None)
            
            if done: break
            
        score = rewards[-1] if rewards else 0.0
        success = score >= 0.7
    except Exception as e:
        print(f"[DEBUG] Error: {e}")
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

def main():
    # Health check for the environment
    for _ in range(15):
        try:
            if requests.get(f"{ENV_URL}/health").status_code == 200: break
        except: pass
        time.sleep(2)
        
    for tid in ["task1", "task2", "task3"]:
        run_task(tid)

if __name__ == "__main__":
    main()
