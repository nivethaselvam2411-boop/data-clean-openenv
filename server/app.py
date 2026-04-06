import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
DataClean OpenEnv — FastAPI server.
Implements OpenEnv HTTP API: POST /reset, POST /step, GET /state, GET /tasks
"""

from __future__ import annotations
import os
import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from environment.env import DataCleanEnvironment, TASK_CONFIG
from environment.models import Action


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="DataClean OpenEnv",
    description=(
        "Real-world data cleaning environment for AI agents. "
        "Implements the OpenEnv spec with 3 progressive tasks: "
        "missing values (easy), dtype+dedup (medium), full pipeline (hard)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Per-session environment instances (keyed by session_id)
_sessions: Dict[str, DataCleanEnvironment] = {}


def _get_or_create_env(session_id: str, task_id: str, seed: int) -> DataCleanEnvironment:
    key = f"{session_id}:{task_id}"
    if key not in _sessions:
        _sessions[key] = DataCleanEnvironment(task_id=task_id, seed=seed)
    return _sessions[key]


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "task1"
    seed: int = 42
    session_id: str = "default"


class StepRequest(BaseModel):
    action: Dict[str, Any]
    task_id: str = "task1"
    seed: int = 42
    session_id: str = "default"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", tags=["health"])
def root():
    return {
        "name": "DataClean OpenEnv",
        "version": "1.0.0",
        "status": "running",
        "tasks": list(TASK_CONFIG.keys()),
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/health", "/docs"],
    }


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.get("/tasks", tags=["tasks"])
def list_tasks():
    """List all available tasks with metadata."""
    tasks = []
    for tid, cfg in TASK_CONFIG.items():
        tasks.append({
            "task_id": tid,
            "description": cfg["description"],
            "max_steps": cfg["max_steps"],
            "difficulty": {"task1": "easy", "task2": "medium", "task3": "hard"}.get(tid),
            "hints": cfg["hints"],
            "schema_requirements": cfg.get("schema_requirements"),
        })
    return {"tasks": tasks}

@app.post("/reset", tags=["openenv"])
def reset(req: Optional[ResetRequest] = None): # Make it optional
    try:
        # Use defaults if no body is provided
        if req is None:
            req = ResetRequest()

        key = f"{req.session_id}:{req.task_id}"
        _sessions[key] = DataCleanEnvironment(task_id=req.task_id, seed=req.seed)
        obs = _sessions[key].reset()
        return obs.model_dump()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/step", tags=["openenv"])
def step(req: Optional[StepRequest] = None):
    """
    Apply an action and return observation, reward, done, info.
    OpenEnv spec: POST /step
    """
    try:
        if req is None:
            # You can't take a step without an action, 
            # but we'll return a cleaner error if it's missing
            raise HTTPException(status_code=400, detail="Action required")
        env = _get_or_create_env(req.session_id, req.task_id, req.seed)
        obs, reward, done, info = env.step(req.action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"{type(e).__name__}: {str(e)}")



@app.get("/state", tags=["openenv"])
def state(
    task_id: str = Query("task1"),
    session_id: str = Query("default"),
    seed: int = Query(42),
):
    """
    Return full current environment state.
    OpenEnv spec: GET /state
    """
    try:
        env = _get_or_create_env(session_id, task_id, seed)
        return env.state()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/validate", tags=["openenv"])
def validate():
    """Validate all tasks can be instantiated and stepped."""
    results = {}
    for tid in TASK_CONFIG:
        try:
            env = DataCleanEnvironment(task_id=tid, seed=42)
            obs = env.reset()
            # Take one no-op step
            _, reward, done, info = env.step({"action_type": "submit"})
            results[tid] = {
                "status": "ok",
                "obs_keys": list(obs.model_dump().keys()),
                "reward": reward,
                "done": done,
            }
        except Exception as e:
            results[tid] = {"status": "error", "error": str(e)}
    return {"validation": results, "all_passed": all(r["status"] == "ok" for r in results.values())}
