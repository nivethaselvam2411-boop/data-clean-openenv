# 🧹 DataClean OpenEnv

> A real-world **data cleaning environment** for AI agents — built on the [OpenEnv](https://openenv.dev) specification.

DataClean simulates the daily work of data engineers and analysts: fixing messy datasets containing missing values, wrong data types, duplicate rows, statistical outliers, and inconsistent string formatting. Three progressive tasks (easy → hard) with deterministic graders and shaped reward functions guide agents through realistic data quality pipelines.

---

## 🎯 Why DataClean?

Every organization generates messy data. Data cleaning consumes [80% of a data scientist's time](https://www.ibm.com/cloud/blog/ibm-data-catalog-data-scientists-productivity). This environment enables:

- **Agent training** — RL/LLM agents learn systematic data cleaning workflows
- **Agent evaluation** — Benchmark models on realistic, deterministic cleaning tasks
- **Research** — Study how agents handle sequential multi-objective optimization

---

## 🗂️ Environment Overview

| Property | Value |
|---|---|
| Domain | Data Engineering / Analytics |
| Tasks | 3 (Easy → Medium → Hard) |
| Reward | Shaped (0.0 – 1.0), partial credit at every step |
| Episode length | 15 / 25 / 35 steps |
| State | Full dataset profile (nulls, dtypes, outliers, duplicates) |
| Actions | 9 typed action types |

---

## 📊 Tasks

### Task 1 — Fill Missing Values (Easy, 15 steps)
**Dataset:** HR Employee Records (120 rows × 7 columns)

**Objective:** Fill all missing values in `salary`, `department`, and `tenure_years`.

**What's broken:**
- 18 missing salary values (numeric → use `median`)
- 12 missing department values (categorical → use `mode`)
- 10 missing tenure_years values (numeric → use `median`)

**Success threshold:** 0.80 | **Optimal steps:** 3

---

### Task 2 — Fix Dtypes + Remove Duplicates (Medium, 25 steps)
**Dataset:** E-commerce Orders (200 rows + 15 injected duplicates × 8 columns)

**Objective:** Fix 4 wrong data types, remove duplicate orders, fill remaining nulls.

**What's broken:**
- `quantity` stored as string (should be `int`)
- `unit_price` stored as `"$12.50"` string (should be `float`)
- `order_date` stored as string (should be `datetime`)
- `customer_id` stored as string (should be `int`)
- 15 duplicate order rows (exact copies)
- 10 missing `quantity` + 8 missing `unit_price` values

**Success threshold:** 0.80 | **Optimal steps:** 7

---

### Task 3 — Full Pipeline (Hard, 35 steps)
**Dataset:** Healthcare Patient Records (300 rows + 20 duplicates × 11 columns)

**Objective:** Complete 6-stage cleaning pipeline.

| Stage | Task |
|---|---|
| 1. Type fixing | `age`, `bmi` (str→float), `admission_date` (DD/MM/YYYY→datetime), `systolic_bp` (strip "mmHg" units then float) |
| 2. Deduplication | Remove duplicate patients by `patient_id` |
| 3. Missing values | Fill `age`, `bmi`, `cholesterol`, `glucose` (median); `gender` (mode) |
| 4. Outlier removal | Remove IQR-based outliers in `systolic_bp` and `bmi` |
| 5. String formatting | `gender`→titlecase, `blood_type`→strip whitespace, `diagnosis_code`→UPPERCASE |
| 6. Schema validation | All 11 columns present with correct types |

**Success threshold:** 0.85 | **Optimal steps:** 15

---

## 🔧 Action Space

All actions are structured JSON objects with an `action_type` field:

| Action | Fields | Description |
|---|---|---|
| `fill_missing` | `column`, `strategy`, `value?` | Fill nulls: mean/median/mode/constant/ffill/bfill/drop |
| `fix_dtype` | `column`, `target_type`, `datetime_format?` | Convert dtype: int/float/str/bool/datetime |
| `remove_duplicates` | `subset?`, `keep` | Remove duplicate rows |
| `remove_outliers` | `column`, `method`, `threshold` | IQR / Z-score / clip outliers |
| `standardize_format` | `column`, `format_type` | lowercase/uppercase/titlecase/strip/strip_special |
| `filter_rows` | `column`, `operator`, `value` | eq/ne/gt/lt/gte/lte/isin/notin/contains |
| `rename_column` | `old_name`, `new_name` | Rename column |
| `validate_schema` | `expected_columns`, `expected_dtypes` | Check schema compliance |
| `submit` | `message?` | Submit cleaned dataset for grading |

### Example Actions

```json
{"action_type": "fill_missing", "column": "salary", "strategy": "median"}
{"action_type": "fix_dtype", "column": "order_date", "target_type": "datetime", "datetime_format": "%Y-%m-%d"}
{"action_type": "remove_duplicates", "subset": ["order_id"], "keep": "first"}
{"action_type": "remove_outliers", "column": "systolic_bp", "method": "iqr", "threshold": 1.5}
{"action_type": "standardize_format", "column": "gender", "format_type": "titlecase"}
{"action_type": "submit"}
```

---

## 👁️ Observation Space

Every step returns a rich observation:

```json
{
  "task_id": "task1",
  "task_description": "...",
  "step_number": 2,
  "max_steps": 15,
  "current_score": 0.698,
  "issues_remaining": [
    "Missing values: 40 nulls across ['department', 'salary', 'tenure_years']",
    "Outliers: 1 outliers in 'salary'"
  ],
  "hints": ["Use strategy='median' for salary", "..."],
  "schema_requirements": {"salary": "numeric"},
  "dataset_profile": {
    "row_count": 120,
    "col_count": 7,
    "total_nulls": 22,
    "total_null_pct": 0.026,
    "duplicate_row_count": 0,
    "columns": [
      {
        "name": "salary",
        "dtype": "float64",
        "null_count": 18,
        "null_pct": 0.15,
        "unique_count": 94,
        "sample_values": [82500.0, 63200.0, 91100.0],
        "min": 31200.0, "max": 148900.0, "mean": 74823.5,
        "has_outliers": false
      }
    ]
  },
  "action_history": [...]
}
```

---

## 🏆 Reward Function

Rewards are shaped — the agent receives meaningful signal at **every step**, not just at episode end.

| Component | Task 1 | Task 2 | Task 3 | Description |
|---|---|---|---|---|
| `null_score` | 55% | 15% | 20% | % missing values fixed |
| `dtype_score` | 15% | 35% | 20% | Fraction of correct data types |
| `duplicate_score` | 10% | 30% | 15% | Absence of duplicate rows |
| `outlier_score` | — | — | 20% | Statistical outlier removal |
| `format_score` | — | — | 10% | String format consistency |
| `schema_score` | — | — | 5% | All required columns/types present |
| `efficiency_penalty` | 20% | 20% | 10% | Penalty for wasted actions |

---

## 🚀 Setup & Usage

### Local (Python)

```bash
git clone https://huggingface.co/spaces/YOUR_USERNAME/dataclean-openenv
cd dataclean-openenv
pip install -r requirements.txt
uvicorn app:app --port 7860
```

### Docker

```bash
docker build -t dataclean-openenv .
docker run -p 7860:7860 dataclean-openenv
```

### API Usage

```python
import requests

BASE = "http://localhost:7860"

# Reset environment
obs = requests.post(f"{BASE}/reset", json={"task_id": "task1", "seed": 42}).json()

# Take a step
result = requests.post(f"{BASE}/step", json={
    "task_id": "task1",
    "action": {"action_type": "fill_missing", "column": "salary", "strategy": "median"}
}).json()

print(result["reward"])  # 0.698
print(result["done"])    # False
```

### Python Direct

```python
from environment.env import DataCleanEnvironment

env = DataCleanEnvironment(task_id="task1", seed=42)
obs = env.reset()

obs, reward, done, info = env.step({
    "action_type": "fill_missing",
    "column": "salary",
    "strategy": "median"
})
print(f"reward={reward:.3f}, done={done}")
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Environment info |
| GET | `/health` | Health check |
| GET | `/tasks` | List all tasks |
| POST | `/reset` | Reset episode |
| POST | `/step` | Take action |
| GET | `/state` | Current state snapshot |
| POST | `/validate` | Validate all tasks |
| GET | `/docs` | Swagger UI |

---

## 🤖 Baseline Inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_key_here"
export ENV_URL="http://localhost:7860"

python inference.py
```

### Baseline Scores (GPT-4o-mini)

| Task | Difficulty | Baseline Score | Steps Used |
|---|---|---|---|
| task1 | Easy | ~0.85 | 4–6 |
| task2 | Medium | ~0.72 | 8–12 |
| task3 | Hard | ~0.65 | 18–25 |

---

## 📁 Project Structure

```
dataclean-openenv/
├── app.py                    # FastAPI server (OpenEnv HTTP API)
├── inference.py              # Baseline LLM agent script
├── openenv.yaml              # OpenEnv metadata spec
├── Dockerfile                # Container definition
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── environment/
│   ├── __init__.py
│   ├── models.py             # Pydantic typed models (Observation, Action, Reward)
│   ├── datasets.py           # Synthetic dataset generators
│   └── env.py                # Main DataCleanEnvironment class
├── graders/
│   ├── __init__.py
│   └── graders.py            # Deterministic task graders
└── tests/
    └── test_environment.py   # Full test suite (8 tests)
```

---

## 🔬 Running Tests

```bash
python tests/test_environment.py
```

Expected output: `8 passed, 0 failed`

---

## 📜 License

MIT
