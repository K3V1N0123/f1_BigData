# 🏎️ F1 Race Strategy Optimization — AI Agentic Development Guide

> **Project:** Data-Driven & AI-Powered Formula 1 Race Strategy Optimization  
> **Team:** Kevin Vataliya · Tanay Shankarikopa · Ishaan Thakur  
> **Guide:** Mr. Giridhar N Shakarad  
> **Specialization:** Big Data Analytics — School of Computer Engineering  
> **Stack:** Python · FastF1 · Scikit-learn · PyTorch · Stable-Baselines3 · FastAPI · React

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Agent Persona & Behaviour Rules](#2-agent-persona--behaviour-rules)
3. [Directory Structure](#3-directory-structure)
4. [Environment & Configuration](#4-environment--configuration)
5. [Data Specification](#5-data-specification)
6. [Pipeline Architecture](#6-pipeline-architecture)
7. [Module Specifications](#7-module-specifications)
   - [7.1 Data Ingestion](#71-data-ingestion)
   - [7.2 Data Preprocessing & Feature Engineering](#72-data-preprocessing--feature-engineering)
   - [7.3 Lap Time Prediction Model](#73-lap-time-prediction-model)
   - [7.4 Tyre Degradation Model](#74-tyre-degradation-model)
   - [7.5 Pit Stop Delta Model](#75-pit-stop-delta-model)
   - [7.6 Race Simulation Engine](#76-race-simulation-engine)
   - [7.7 Monte Carlo Simulation](#77-monte-carlo-simulation)
   - [7.8 Reinforcement Learning Agent](#78-reinforcement-learning-agent)
   - [7.9 FastAPI Backend](#79-fastapi-backend)
   - [7.10 React Dashboard](#710-react-dashboard)
8. [Output Contracts](#8-output-contracts)
9. [Model Performance Targets](#9-model-performance-targets)
10. [Error Handling Rules](#10-error-handling-rules)
11. [Testing Specification](#11-testing-specification)
12. [Coding Standards](#12-coding-standards)
13. [Agent Task Queue](#13-agent-task-queue)
14. [Common Pitfalls & Guardrails](#14-common-pitfalls--guardrails)
15. [Prompt Templates for AI Agent Invocation](#15-prompt-templates-for-ai-agent-invocation)
16. [Notebook Templates](#16-notebook-templates)
17. [Interpretation & Validation Guide](#17-interpretation--validation-guide)
18. [API Contract Reference](#18-api-contract-reference)
19. [Extensions (Post-MVP)](#19-extensions-post-mvp)
20. [Git Workflow & Deliverables Checklist](#20-git-workflow--deliverables-checklist)

---

## 1. Project Overview

### What You Are Building

A modular software platform with five integrated layers:

| Layer | Description |
|-------|-------------|
| **Data Layer** | Automated ingestion and preprocessing of F1 race telemetry from the FastF1 API |
| **Modelling Layer** | ML models for lap time prediction, tyre degradation, and pit stop delta |
| **Simulation Layer** | Deterministic race engine + Monte Carlo probabilistic analysis |
| **Optimization Layer** | PPO Reinforcement Learning agent for adaptive pit stop strategy |
| **Interface Layer** | FastAPI REST backend + interactive React dashboard |

### Project Phases at a Glance

| Phase | Name | Duration | Key Deliverables |
|-------|------|----------|------------------|
| 1 | Setup & Data Ingestion | Weeks 1–2 | Environment configured, raw data collected |
| 2 | Preprocessing & EDA | Weeks 3–4 | Clean datasets, feature-engineered DataFrames, EDA notebook |
| 3 | Predictive Modelling | Weeks 5–7 | Lap time model, tyre degradation model, pit delta model |
| 4 | Race Simulation Engine | Weeks 8–9 | Deterministic simulator + Monte Carlo engine |
| 5 | Reinforcement Learning | Weeks 10–12 | Trained RL agent with strategy policy |
| 6 | API & Dashboard | Weeks 13–14 | FastAPI backend, React dashboard, live recommendations |
| 7 | Testing & Evaluation | Weeks 15–16 | Unit tests, model evaluations, benchmarks |
| 8 | Documentation & Report | Weeks 17–18 | Final report, README, slide deck, demo video |

> **Note:** All data is sourced exclusively from FastF1. No Ergast or other external APIs are used.

---

## 2. Agent Persona & Behaviour Rules

> These rules govern how an AI coding agent must behave when developing this project autonomously.

### Identity

```
You are a senior software engineer and quantitative motorsport analyst.
Your domain is F1 telemetry data, predictive ML, RL optimization, and full-stack Python.
You write clean, modular, documented, reproducible code.
You never invent F1 data — always fetch from FastF1.
You build each phase completely before moving to the next.
```

### Core Rules

```yaml
rules:
  - NEVER hardcode lap times, tyre data, or race results — always fetch from API
  - ALWAYS validate DataFrame shape and null counts after every ingestion/cleaning step
  - ALWAYS write Google-style docstrings on every function and class
  - ALWAYS save outputs to paths specified in output_contracts — never change filenames
  - NEVER split data randomly by row — always split by race/season to prevent data leakage
  - NEVER apply StandardScaler fitted on full data — fit on training split only
  - ALWAYS remove out-laps and in-laps before training any lap time model
  - ALWAYS normalize compound names at ingestion (SOFT/MEDIUM/HARD, never numeric codes)
  - ALWAYS log decisions with Python logging module — no print() statements
  - ALWAYS set random seeds (seed: 42) for all models and simulations
  - ALWAYS produce figures at 300 DPI, saved to docs/figures/
  - NEVER commit model weights or raw data to git — add to .gitignore
  - ALWAYS gate structural break tests with minimum sample size check
  - ALWAYS add max_tyre_age guard in degradation model inference
```

### Decision Rules

```yaml
on_api_failure:
  - retry: 3 times with 5s exponential backoff
  - fallback: load from data/raw/ Parquet cache if exists
  - delay: 2 seconds between FastF1 calls to avoid rate limiting
  - raise: DataIngestionError if both live and cache fail
  - log: WARNING for every retry, ERROR for fallback trigger

on_missing_data:
  - lap_time_null: drop row entirely
  - tyre_life_null: forward-fill within driver stint (max 3 consecutive)
  - compound_null: drop row — cannot be imputed safely
  - weather_null: use circuit average from config/circuit_params.yaml
  - log: all fill/drop decisions at WARNING level with row counts

on_outlier_detection:
  - upper_bound: median + 2.5 * IQR on LapTime per circuit
  - lower_bound: 1.05 * circuit fastest lap record
  - action: drop and log to data/processed/dropped_laps_log.csv

on_model_underperformance:
  - lap_time_mae_above_0.5: log WARNING, try additional hyperparameter tuning
  - r2_below_0.92: check for data leakage, re-examine feature set
  - never_proceed: do not proceed to simulation with an undertrained model

on_rl_training:
  - reward_sparse: add intermediate position-gain rewards
  - not_generalizing: increase episode randomization in reset()
  - unstable: reduce learning_rate to 1e-4, increase n_steps
```

---

## 3. Directory Structure

```
f1_strategy/
│
├── data/
│   ├── raw/                          # Raw API data — NEVER modified after save
│   │   ├── laps/                     # Hive-partitioned: season={y}/event={e}/part.parquet
│   │   │   └── season={2021,2022,2023,2024}/
│   │   │       └── event={EventName}/
│   │   │           └── part.parquet
│   │   ├── weather/                  # Same Hive-partitioned structure
│   │   ├── results/                  # Same Hive-partitioned structure
│   │   └── race_control/             # Same Hive-partitioned structure
│   ├── processed/                    # Cleaned, feature-engineered data (to be created)
│   │   ├── laps_clean_{circuit}_{year}.parquet
│   │   ├── laps_all_circuits.parquet
│   │   ├── tyre_deg_coefficients.json
│   │   └── dropped_laps_log.csv
│   └── cache/                        # FastF1 HTTP cache (gitignored)
│       └── fastf1_http_cache.sqlite
│
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── load_race_sessions.py     # FastF1 session loader + Hive-partitioned writer
│   │   ├── session_load.py           # Minimal FastF1 load test
│   │   └── collect_missing.py        # Incremental missing-session collector
│   ├── preprocessing/                # ⬜ stub — to be built
│   │   ├── __init__.py
│   │   ├── cleaner.py
│   │   └── feature_engineer.py
│   ├── models/                       # ⬜ stub — to be built
│   │   ├── __init__.py
│   │   ├── train_laptime.py
│   │   ├── tyre_degradation.py
│   │   ├── pit_delta.py
│   │   └── saved/                    # Serialized model artifacts (gitignored)
│   │       ├── laptime_model.pkl
│   │       ├── scaler.pkl
│   │       └── tyre_deg_coefficients.json
│   ├── simulation/                   # ⬜ stub — to be built
│   │   ├── __init__.py
│   │   ├── race_engine.py
│   │   ├── traffic_model.py
│   │   ├── safety_car.py
│   │   └── monte_carlo.py
│   ├── rl_agent/                     # ⬜ stub — to be built
│   │   ├── __init__.py
│   │   ├── f1_env.py
│   │   ├── train_agent.py
│   │   ├── evaluate_agent.py
│   │   └── saved/                    # Trained RL model (gitignored)
│   │       └── f1_ppo_agent.zip
│   └── api/                          # ⬜ stub — to be built
│       ├── __init__.py
│       ├── main.py
│       ├── routes/
│       │   ├── strategy.py
│       │   ├── simulation.py
│       │   └── data.py
│       └── schemas.py
│
├── dashboard/                        # ⬜ stub — React frontend (to be built)
│   ├── src/
│   │   ├── components/
│   │   │   ├── RaceControlPanel.jsx
│   │   │   ├── StrategyComparison.jsx
│   │   │   ├── LiveStrategyAdvisor.jsx
│   │   │   └── HistoricalExplorer.jsx
│   │   └── App.jsx
│   └── package.json
│
├── notebooks/
│   ├── check.ipynb                   # Data validation notebook
│   ├── 01_eda.ipynb                  # ⬜ stub — to be built
│   ├── 02_model_evaluation.ipynb     # ⬜ stub — to be built
│   └── 03_simulation_demo.ipynb      # ⬜ stub — to be built
│
├── tests/
│   ├── __init__.py
│   ├── test_session_load.py          # Session load integration test
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── test_load_race_sessions.py  # Unit tests for ingestion module
│   ├── test_cleaner.py               # ⬜ stub — to be built
│   ├── test_features.py              # ⬜ stub — to be built
│   ├── test_models.py                # ⬜ stub — to be built
│   ├── test_simulator.py             # ⬜ stub — to be built
│   └── test_api.py                   # ⬜ stub — to be built
│
├── config/
│   ├── config.yaml                   # ⬜ stub — All project parameters
│   ├── circuit_params.yaml           # ⬜ stub — Per-circuit params
│   └── pit_deltas.yaml               # ⬜ stub — Mean pit delta per circuit
│
├── docs/
│   └── figures/                      # ⬜ stub — All exported charts (PNG 300 DPI)
│
├── requirements.txt                  # ⬜ stub — Python dependencies
├── README.md
└── .gitignore
```

---

## 4. Environment & Configuration

### `requirements.txt`

```
# Data
fastf1>=3.3.0
pandas>=2.0.0
numpy>=1.24.0
pyarrow>=14.0.0
requests>=2.31.0               # FastF1 dependency

# ML
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.1.0
joblib>=1.3.0

# Deep Learning
torch>=2.1.0
torchvision>=0.16.0

# Reinforcement Learning
stable-baselines3>=2.2.0
gymnasium>=0.29.0

# Simulation
scipy>=1.11.0

# API
fastapi>=0.104.0
uvicorn>=0.24.0
pydantic>=2.5.0

# Visualisation
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.15.0

# Experiment Tracking
mlflow>=2.8.0

# Dev & Testing
jupyter>=1.0.0
pytest>=7.4.0
pytest-cov>=4.1.0
httpx>=0.25.0
pyyaml>=6.0
python-dotenv>=1.0.0
tqdm>=4.65.0
```

### `config.yaml`

```yaml
project:
  name: "F1 Race Strategy Optimization"
  version: "1.0.0"
  random_seed: 42

data:
  seasons: [2021, 2022, 2023, 2024]
  session_type: "R"                  # R=Race, Q=Qualifying
  cache_path: "data/cache"
  raw_path: "data/raw"
  raw_tables: [laps, weather, results, race_control]
  processed_path: "data/processed"
  api_delay_seconds: 2

fuel:
  starting_kg: 105
  burn_rate_kg_per_lap: 1.6
  time_cost_per_kg: 0.035            # seconds per kg

tyre:
  compounds:
    SOFT: 0
    MEDIUM: 1
    HARD: 2
    INTERMEDIATE: 3
    WET: 4
  max_stint_laps:
    SOFT: 30
    MEDIUM: 40
    HARD: 55
    INTERMEDIATE: 999
    WET: 999

cleaning:
  iqr_multiplier: 2.5
  fastest_lap_margin: 1.05
  max_forward_fill_laps: 3

models:
  train_test_split_by: "race"        # never "random"
  train_ratio: 0.80
  val_ratio: 0.10
  test_ratio: 0.10
  cv_folds: 5
  laptime_target_mae: 0.5
  laptime_target_r2: 0.92
  tyre_deg_target_rmse: 0.3

simulation:
  fuel_burn_rate: 1.6
  traffic_threshold_sec: 1.0         # gap below which traffic penalty applies
  traffic_penalty_min: 0.3
  traffic_penalty_max: 0.8
  sc_lap_time_multiplier: 1.5
  vsc_lap_time_multiplier: 1.2
  monte_carlo_n_simulations: 5000
  lap_time_noise_sigma: 0.15
  pit_duration_mean: 2.5
  pit_duration_std: 0.3

rl:
  algorithm: "PPO"
  policy: "MlpPolicy"
  learning_rate: 3.0e-4
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  total_timesteps: 500000
  eval_freq: 50000
  n_eval_episodes: 200

api:
  host: "0.0.0.0"
  port: 8000

outputs:
  figures_path: "docs/figures"
  dpi: 300
```

### `config/circuit_params.yaml`

```yaml
circuits:
  Bahrain:
    total_laps: 57
    sc_probability_per_lap: 0.015
    overtaking_difficulty: 0.3        # 0=easy, 1=impossible
    avg_air_temp: 28
    avg_track_temp: 38
  Monaco:
    total_laps: 78
    sc_probability_per_lap: 0.045
    overtaking_difficulty: 0.95
    avg_air_temp: 22
    avg_track_temp: 32
  Silverstone:
    total_laps: 52
    sc_probability_per_lap: 0.020
    overtaking_difficulty: 0.4
    avg_air_temp: 18
    avg_track_temp: 28
  Monza:
    total_laps: 53
    sc_probability_per_lap: 0.030
    overtaking_difficulty: 0.35
    avg_air_temp: 24
    avg_track_temp: 34
  # Add all remaining circuits following same structure
```

---

## 5. Data Specification

### Raw Schema — FastF1 Laps DataFrame (from Hive-partitioned Parquet)

**Partition columns** (added at ingestion by `load_race_sessions.py`):

| Column | Type | Description |
|--------|------|-------------|
| `season` | `int` | Season year (2021-2024) |
| `event_name` | `str` | Original event name (e.g. "Bahrain Grand Prix") |
| `round_number` | `int` | Round number within season |
| `session_name` | `str` | Session name (e.g. "Race") |
| `session_date` | `str` | Date string (e.g. "2023-03-05") |

**Core FastF1 columns** (from `session.laps`):

| Column | Type | Description |
|--------|------|-------------|
| `LapNumber` | `int` | Lap number within race (1 to total laps) |
| `LapTime` | `timedelta` | Lap duration — convert to float seconds |
| `Sector1Time` | `timedelta` | Sector 1 time |
| `Sector2Time` | `timedelta` | Sector 2 time |
| `Sector3Time` | `timedelta` | Sector 3 time |
| `Compound` | `str` | Tyre compound: SOFT / MEDIUM / HARD / INTERMEDIATE / WET |
| `TyreLife` | `int` | Age of tyre in laps at that lap |
| `Driver` | `str` | Three-letter driver code (VER, HAM, etc.) |
| `DriverNumber` | `str` | Driver number |
| `Team` | `str` | Constructor name |
| `Stint` | `int` | Stint number |
| `PitInTime` | `timedelta` | Time driver entered pit lane (NaT if no pit) |
| `PitOutTime` | `timedelta` | Time driver exited pit lane (NaT if no pit) |
| `FreshTyre` | `bool` | Whether tyre is fresh (start of stint) |
| `LapStartTime` | `timedelta` | Absolute lap start time |
| `Position` | `int` | Position during/at end of lap |
| `TrackStatus` | `str` | Track status flag |
| `IsAccurate` | `bool` | FastF1 accuracy flag |
| `SpeedI1` `SpeedI2` `SpeedFL` `SpeedST` | `float` | Speed trap measurements |

### Processed Schema — Feature-Engineered DataFrame

| Column | Type | Notes |
|--------|------|-------|
| `LapTimeSec` | `float64` | `LapTime.total_seconds()` |
| `TyreAge` | `int` | Lap number within current stint |
| `TyreCompoundEncoded` | `int` | SOFT=0, MEDIUM=1, HARD=2, INTER=3, WET=4 |
| `StartingTyreAge` | `int` | Tyre age at stint start |
| `RelativeTyrePerformance` | `float64` | `LapTimeSec - stint_mean` |
| `FuelLoad` | `float64` | `105 - 1.6 * LapNumber` |
| `FuelEffect` | `float64` | `FuelLoad * 0.035` |
| `RaceProgress` | `float64` | `LapNumber / TotalLaps` |
| `IsPitLap` | `bool` | True if PitInTime is not NaT |
| `IsOutLap` | `bool` | True if PitOutTime is not NaT |
| `CircuitEncoded` | `int` | Label-encoded circuit name |
| `DriverEncoded` | `int` | Label-encoded driver code |
| `AirTemp` | `float64` | Merged from weather data |
| `TrackTemp` | `float64` | Merged from weather data |
| `Humidity` | `float64` | Merged from weather data |
| `IsRainingLap` | `bool` | `Rainfall > 0` |
| `IsInTraffic` | `bool` | `GapToCarAhead < 1.0` |
| `GapToCarAhead` | `float64` | Estimated gap in seconds |
| `DeltaFromBase` | `float64` | `LapTimeSec - min(first 3 laps of stint)` — target for degradation model |

### Validation Rules

```python
# Agent must assert all of these after preprocessing
assert processed_df.isnull().sum().sum() == 0, "NaN values found in processed data"
assert (processed_df['LapTimeSec'] > 0).all(), "Negative lap times detected"
assert (processed_df['LapTimeSec'] < 200).all(), "Implausible lap times (>200s) detected"
assert 'IsPitLap' in processed_df.columns, "Pit lap flag missing"
assert processed_df['TyreCompoundEncoded'].isin([0,1,2,3,4]).all(), "Unknown compound encoded"
assert not processed_df[processed_df['IsOutLap'] | processed_df['IsPitLap']].empty or True  # pit laps exist
```

---

## 6. Pipeline Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        F1 STRATEGY PIPELINE                            │
│                                                                        │
│  ┌──────────────────────────────────────┐                             │
│  │         PHASE 1: DATA LAYER          │                             │
│  │                                      │                             │
│  │  [load_race_sessions.py]             │                             │
│  │       FastF1 API → Hive-partitioned  │                             │
│  │       Parquet (laps, weather,        │                             │
│  │        results, race_control)        │                             │
│  │  [collect_missing.py]                │                             │
│  │       Incremental backfill           │                             │
│  └───────────────┬──────────────────────┘                             │
│                  │                                                     │
│  ┌───────────────▼──────────────────────┐                             │
│  │       PHASE 2: PREPROCESSING         │                             │
│  │                                      │                             │
│  │  [cleaner.py]                        │                             │
│  │       Null handling → Outlier        │                             │
│  │       removal → Time conversion      │                             │
│  │  [feature_engineer.py]               │                             │
│  │       Tyre + Fuel + Traffic          │                             │
│  │       + Weather features             │                             │
│  └───────────────┬──────────────────────┘                             │
│                  │                                                     │
│  ┌───────────────▼──────────────────────┐                             │
│  │       PHASE 3: MODELS                │                             │
│  │                                      │                             │
│  │  [train_laptime.py]    → laptime_model.pkl                        │
│  │  [tyre_degradation.py] → tyre_deg_coefficients.json              │
│  │  [pit_delta.py]        → pit_deltas.yaml                         │
│  └───────────────┬──────────────────────┘                             │
│                  │                                                     │
│  ┌───────────────▼──────────────────────┐                             │
│  │       PHASE 4: SIMULATION            │                             │
│  │                                      │                             │
│  │  [race_engine.py]   deterministic    │                             │
│  │  [traffic_model.py] position gaps    │                             │
│  │  [safety_car.py]    SC/VSC events    │                             │
│  │  [monte_carlo.py]   5000 iterations  │                             │
│  └───────────────┬──────────────────────┘                             │
│                  │                                                     │
│  ┌───────────────▼──────────────────────┐                             │
│  │       PHASE 5: RL AGENT              │                             │
│  │                                      │                             │
│  │  [f1_env.py]        Gym environment  │                             │
│  │  [train_agent.py]   PPO 500k steps   │                             │
│  │  [evaluate_agent.py] 200 eval eps    │                             │
│  └───────────────┬──────────────────────┘                             │
│                  │                                                     │
│  ┌───────────────▼──────────────────────┐                             │
│  │       PHASE 6: INTERFACE             │                             │
│  │                                      │                             │
│  │  [FastAPI backend]  REST endpoints   │                             │
│  │  [React dashboard]  4 page views     │                             │
│  └──────────────────────────────────────┘                             │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Module Specifications

### 7.1 Data Ingestion

**File:** `src/ingestion/load_race_sessions.py`

```python
import re
import unicodedata
from pathlib import Path

import fastf1
import pandas as pd


def normalize_event_name(event_name: str) -> str:
    """
    Normalise event name to ASCII-safe, underscore-separated string.

    Strips accents, replaces non-alphanumeric sequences with '_',
    and removes leading/trailing underscores. Used to build safe
    Hive-partitioned directory names.

    Args:
        event_name: Raw event name (e.g. 'São Paulo Grand Prix')

    Returns:
        Normalised string suitable for filesystem paths
    """
    normalized = unicodedata.normalize("NFKD", event_name) \
                             .encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Za-z0-9]+", "_", normalized).strip("_")


def build_output_path(
    raw_root: Path, table_name: str, season: int, event_name: str
) -> Path:
    """
    Build Hive-partitioned Parquet output path.

    Pattern: {raw_root}/{table_name}/season={season}/event={norm}/part.parquet

    Args:
        raw_root: Root directory for raw data
        table_name: Table name (laps, weather, results, race_control)
        season: Season year
        event_name: Raw event name (will be normalised)

    Returns:
        Full path to write Parquet file
    """
    normalized_event = normalize_event_name(event_name)
    return raw_root / table_name / f"season={season}" \
           / f"event={normalized_event}" / "part.parquet"


def add_session_metadata(
    frame: pd.DataFrame, season: int, event_name: str,
    round_number: int, session_name: str, session_date: str,
) -> pd.DataFrame:
    """
    Enrich a DataFrame with session-level partition columns.

    Columns added: season, event_name, round_number, session_name, session_date

    Args:
        frame: Raw DataFrame from FastF1
        season: Season year
        event_name: Raw event name
        round_number: Round number within season
        session_name: Session type string (e.g. 'Race')
        session_date: ISO date string

    Returns:
        Copy of frame with metadata columns appended
    """
    result = frame.copy()
    result["season"] = season
    result["event_name"] = event_name
    result["round_number"] = round_number
    result["session_name"] = session_name
    result["session_date"] = session_date
    return result


def extract_session_tables(session) -> dict[str, pd.DataFrame]:
    """
    Extract all four table types from a loaded FastF1 session.

    Args:
        session: Loaded FastF1 session object

    Returns:
        Dict mapping table name to DataFrame:
        - 'laps': session.laps
        - 'weather': session.weather_data
        - 'results': session.results
        - 'race_control': session.race_control_messages
    """
    return {
        "laps": session.laps.copy(),
        "weather": session.weather_data.copy(),
        "results": session.results.copy(),
        "race_control": session.race_control_messages.copy(),
    }


def write_table(frame: pd.DataFrame, output_path: Path) -> None:
    """
    Write a DataFrame to a Hive-partitioned Parquet file.

    Creates parent directories if they do not exist.

    Args:
        frame: DataFrame to write
        output_path: Full path ending in part.parquet
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)


def ingest_loaded_session(
    session, raw_root: Path, season: int,
    event_name: str, round_number: int
) -> None:
    """
    Extract and persist all tables from a loaded session.

    Writes to: {raw_root}/{table}/season={season}/event={norm}/part.parquet

    Args:
        session: Loaded FastF1 session object
        raw_root: Root directory for raw data
        season: Season year
        event_name: Raw event name
        round_number: Round number within season
    """
    session_name = str(session.name)
    session_date = str(session.date)[:10]
    tables = extract_session_tables(session)

    for table_name, frame in tables.items():
        enriched = add_session_metadata(
            frame=frame, season=season,
            event_name=event_name, round_number=round_number,
            session_name=session_name, session_date=session_date,
        )
        output_path = build_output_path(raw_root, table_name, season, event_name)
        write_table(enriched, output_path)


def ingest_seasons(
    seasons: list[int],
    raw_root: Path,
    get_event_schedule=fastf1.get_event_schedule,
    get_session=fastf1.get_session,
) -> dict[str, int]:
    """
    Batch-ingest all race sessions for given seasons.

    Iterates through the event schedule for each season, loads each
    race session via FastF1, and persists all four table types as
    Hive-partitioned Parquet files.

    Args:
        seasons: List of season years (e.g. [2021, 2022, 2023, 2024])
        raw_root: Root directory for raw data
        get_event_schedule: Callable for schedule (injectable for testing)
        get_session: Callable for session loading (injectable for testing)

    Returns:
        Summary dict with keys: attempted, succeeded, failed
    """
    summary = {"attempted": 0, "succeeded": 0, "failed": 0}

    for season in seasons:
        schedule = get_event_schedule(season)
        for _, event_row in schedule.iterrows():
            round_number = int(event_row["RoundNumber"])
            event_name = str(event_row["EventName"])
            summary["attempted"] += 1
            try:
                session = get_session(season, round_number, "R")
                session.load()
                ingest_loaded_session(session, raw_root, season, event_name, round_number)
                summary["succeeded"] += 1
            except Exception:
                summary["failed"] += 1

    return summary
```

**File:** `src/ingestion/session_load.py`

```python
from pathlib import Path

import fastf1


PROJECT_ROOT = Path(__file__).resolve().parents[2]
fastf1.Cache.enable_cache(str(PROJECT_ROOT / 'data' / 'cache'))
session = fastf1.get_session(2023, 'Bahrain', 'R')
session.load()
laps = session.laps          # Main laps DataFrame
weather = session.weather_data  # Weather DataFrame
```

**File:** `src/ingestion/collect_missing.py`

```python
from pathlib import Path

import fastf1
import pandas as pd


def normalize_event_name(event_name: str) -> str:
    """ASCII-safe, underscore-separated normalisation."""
    ...


def is_event_collected(raw_root: Path, season: int, event_name: str) -> bool:
    """
    Check whether all four tables exist for a given event.

    Args:
        raw_root: Root directory for raw data
        season: Season year
        event_name: Raw event name

    Returns:
        True if all four table partitions exist
    """
    norm = normalize_event_name(event_name)
    tables = ["laps", "race_control", "results", "weather"]
    return all(
        (raw_root / t / f"season={season}" / f"event={norm}" / "part.parquet").exists()
        for t in tables
    )



---

### 7.2 Data Preprocessing & Feature Engineering

**File:** `src/preprocessing/cleaner.py`

```python
def clean_laps(
    laps_df: pd.DataFrame,
    circuit: str,
    config: dict
) -> pd.DataFrame:
    """
    Apply full cleaning pipeline to raw laps DataFrame.

    Steps in order:
      1. Drop rows where LapTime is null
      2. Flag pit laps: is_pit_lap = PitInTime.notna()
      3. Flag out laps: is_out_lap = PitOutTime.notna()
      4. Forward-fill TyreLife within driver stint (max 3 laps)
      5. Drop rows where Compound is null
      6. Convert LapTime and all SectorTimes to float seconds via .total_seconds()
      7. Remove outliers: LapTimeSec > (median + 2.5 * IQR) per circuit
      8. Remove outliers: LapTimeSec < 1.05 * circuit fastest lap record
      9. Log all dropped rows to data/processed/dropped_laps_log.csv
      10. Normalize compound names: uppercase, strip whitespace

    Args:
        laps_df: Raw laps DataFrame from FastF1
        circuit: Circuit name (for outlier threshold lookup)
        config: Parsed config.yaml

    Returns:
        Cleaned DataFrame

    Note:
        NEVER remove pit laps from the DataFrame — mark them with is_pit_lap=True
        and exclude from model training only via filter at training time.
    """
    pass


def convert_timedeltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert all timedelta columns to float seconds.

    Columns to convert: LapTime, Sector1Time, Sector2Time, Sector3Time

    Returns:
        DataFrame with float columns LapTimeSec, S1Sec, S2Sec, S3Sec
    """
    pass
```

**File:** `src/preprocessing/feature_engineer.py`

```python
def engineer_all_features(
    laps_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    config: dict
) -> pd.DataFrame:
    """
    Derive all predictive features from cleaned laps and weather data.

    Calls all sub-functions in sequence and merges results.

    Returns:
        Feature-complete DataFrame ready for model training
    """
    pass


def add_tyre_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add tyre-related features.

    Features added:
      - TyreAge: lap number within current stint (reset to 1 at each pit)
      - TyreCompoundEncoded: SOFT=0, MEDIUM=1, HARD=2, INTER=3, WET=4
      - StartingTyreAge: tyre age at stint start (from TyreLife at first stint lap)
      - RelativeTyrePerformance: LapTimeSec - mean(LapTimeSec in stint)
      - DeltaFromBase: LapTimeSec - min(LapTimeSec of first 3 laps in stint)

    Note:
        TyreAge must be computed per driver, per stint.
        A stint ends whenever is_pit_lap is True.
    """
    pass


def add_fuel_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add fuel load and fuel effect features.

    Formulas:
      - FuelLoad = config['fuel']['starting_kg'] - config['fuel']['burn_rate_kg_per_lap'] * LapNumber
      - FuelEffect = FuelLoad * config['fuel']['time_cost_per_kg']

    Returns DataFrame with FuelLoad and FuelEffect columns added.
    """
    pass


def add_race_progression_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add race progression features.

    Features:
      - RaceProgress: LapNumber / max(LapNumber) per race session
      - PositionInRace: from session position data if available

    Returns DataFrame with progression features added.
    """
    pass


def merge_weather_features(
    laps_df: pd.DataFrame,
    weather_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge weather data onto laps by nearest timestamp.

    Strategy:
      - Use pd.merge_asof on Time column (nearest match)
      - Add IsRainingLap boolean: Rainfall > 0

    If weather_df is empty, fill with circuit average values from config.
    """
    pass


def add_traffic_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Estimate traffic features from positional data.

    GapToCarAhead: estimated from sector time deltas between
    consecutive drivers at same lap (approximation).

    IsInTraffic: True if GapToCarAhead < config['simulation']['traffic_threshold_sec']

    Note:
        This is an approximation — true gap requires position telemetry
        which is not always available. Document this limitation clearly.
    """
    pass
```

---

### 7.3 Lap Time Prediction Model

**File:** `src/models/train_laptime.py`

```python
FEATURE_COLUMNS = [
    'TyreCompoundEncoded', 'TyreAge', 'StartingTyreAge',
    'FuelLoad', 'FuelEffect',
    'RaceProgress', 'LapNumber',
    'AirTemp', 'TrackTemp', 'Humidity',
    'IsInTraffic', 'GapToCarAhead',
    'CircuitEncoded', 'DriverEncoded'
]

TARGET_COLUMN = 'LapTimeSec'


def split_by_race(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by race (year + circuit combination), NOT by row.

    CRITICAL: Random row splits cause data leakage because laps from
    the same race appear in both train and test sets.

    Strategy:
      1. Get unique (year, circuit) combinations
      2. Shuffle with seed=42
      3. Assign first 80% to train, next 10% to val, last 10% to test
      4. Filter df by assigned races

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    pass


def train_all_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: dict
) -> dict:
    """
    Train and evaluate all candidate models on training data.

    Models to train:
      1. LinearRegression (baseline)
      2. XGBRegressor (expected best)
      3. LGBMRegressor
      4. RandomForestRegressor
      5. MLP (PyTorch, architecture: 256 → 128 → 64 → 1)

    For each model:
      - Fit StandardScaler on train features (NEVER on full data)
      - Tune hyperparameters via 5-fold CV on train set
      - Evaluate on val_df using MAE, RMSE, R2
      - Log metrics to MLflow

    Returns:
        Dict mapping model_name → {'model': fitted_model, 'mae': float, 'r2': float}
    """
    pass


def select_and_save_best_model(
    models: dict,
    test_df: pd.DataFrame,
    scaler,
    save_dir: str
) -> None:
    """
    Evaluate all models on test_df, select best by MAE, save artifacts.

    Saves:
      - src/models/saved/laptime_model.pkl  (best model via joblib)
      - src/models/saved/scaler.pkl         (fitted scaler via joblib)
      - outputs/tables/model_comparison.csv (all model metrics)

    Raises:
        ModelPerformanceError: if best model MAE > 0.5 or R2 < 0.92
    """
    pass
```

**PyTorch MLP Architecture:**

```python
import torch
import torch.nn as nn

class LapTimeMLP(nn.Module):
    """
    Multi-layer perceptron for lap time regression.

    Architecture:
        Input → Linear(256) → ReLU → Dropout(0.2)
               → Linear(128) → ReLU → Dropout(0.2)
               → Linear(64)  → ReLU
               → Linear(1)

    Args:
        input_dim: Number of input features (len(FEATURE_COLUMNS))
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)
```

---

### 7.4 Tyre Degradation Model

**File:** `src/models/tyre_degradation.py`

```python
def fit_degradation_curves(
    laps_df: pd.DataFrame,
    degree: int = 2
) -> dict:
    """
    Fit polynomial degradation curves per (circuit, compound) pair.

    Method:
      1. Filter to non-pit, non-outlap laps only
      2. Group by (CircuitName, Compound)
      3. Compute DeltaFromBase = LapTimeSec - min(first 3 laps of stint)
      4. Fit numpy.polyfit(TyreAge, DeltaFromBase, deg=degree) per group
      5. Store coefficients in dict keyed by (circuit, compound) tuple

    Args:
        laps_df: Processed laps DataFrame with DeltaFromBase column
        degree: Polynomial degree (default 2 — quadratic)

    Returns:
        Dict: {(circuit, compound): np.ndarray of coefficients}

    Side effects:
        Saves dict to data/processed/tyre_deg_coefficients.json
    """
    pass


def predict_degradation_delta(
    circuit: str,
    compound: str,
    tyre_age: int,
    coefficients: dict,
    max_tyre_age: int = 55
) -> float:
    """
    Predict lap time penalty from tyre degradation at given age.

    Formula: delta = coeff[0]*age^2 + coeff[1]*age + coeff[2]

    Args:
        circuit: Circuit name
        compound: Tyre compound string (SOFT/MEDIUM/HARD)
        tyre_age: Current tyre age in laps
        coefficients: Dict from fit_degradation_curves()
        max_tyre_age: Guard value — raise ValueError if tyre_age > max_tyre_age

    Returns:
        Predicted time penalty in seconds (always >= 0)

    Raises:
        ValueError: if tyre_age > max_tyre_age (extrapolation guard)
        KeyError: if (circuit, compound) not in coefficients
    """
    pass
```

---

### 7.5 Pit Stop Delta Model

**File:** `src/models/pit_delta.py`

```python
def compute_pit_deltas(
    laps_df: pd.DataFrame,
    config: dict
) -> dict:
    """
    Compute mean pit stop time loss per circuit from historical data.

    Method:
      1. Filter to laps where IsPitLap is True
      2. Compute pit_duration = PitOutTime - PitInTime (in seconds)
      3. Subtract actual_stop_time (approximated as 2.5s)
      4. pit_delta = pit_duration - expected_race_lap_time_for_that_lap
      5. Compute mean and std pit_delta per circuit

    Returns:
        Dict: {circuit_name: {'mean': float, 'std': float}}

    Side effects:
        Saves to config/pit_deltas.yaml
    """
    pass
```

---

### 7.6 Race Simulation Engine

**File:** `src/simulation/race_engine.py`

```python
from dataclasses import dataclass
from typing import List

@dataclass
class Stint:
    compound: str      # SOFT / MEDIUM / HARD / INTERMEDIATE / WET
    start_lap: int     # Lap this stint begins
    end_lap: int       # Lap this stint ends (pit lap)


@dataclass
class Strategy:
    stints: List[Stint]
    driver: str
    circuit: str


def simulate_race(
    strategy: Strategy,
    lap_time_model,
    scaler,
    deg_coefficients: dict,
    pit_deltas: dict,
    circuit_params: dict,
    config: dict,
    noise: bool = False,
    random_seed: int = 42
) -> dict:
    """
    Run deterministic race simulation for a given strategy.

    Algorithm (per lap 1 to total_laps):
      1. Determine current compound and tyre_age from strategy stints
      2. Compute FuelLoad and FuelEffect
      3. Predict base lap time via lap_time_model.predict()
      4. Add degradation delta from predict_degradation_delta()
      5. Add traffic_delta from traffic_model (uses previous lap positions)
      6. If current lap is pit lap: add pit_delta[circuit]['mean'], reset tyre_age
      7. If noise=True: add Gaussian noise N(0, config['simulation']['lap_time_noise_sigma'])
      8. Accumulate total_race_time

    Args:
        strategy: Strategy dataclass defining stints
        lap_time_model: Fitted sklearn/XGB model with .predict() interface
        scaler: Fitted StandardScaler
        deg_coefficients: Dict from tyre_degradation module
        pit_deltas: Dict of mean pit deltas per circuit
        circuit_params: Dict with total_laps, sc_probability etc.
        config: Parsed config.yaml
        noise: If True, add stochastic noise (for Monte Carlo)
        random_seed: Random seed for reproducibility

    Returns:
        {
            'total_time_sec': float,
            'lap_times': list[float],
            'pit_laps': list[int],
            'compounds_used': list[str],
            'n_stops': int
        }

    Raises:
        InvalidStrategyError: if stints don't cover all laps
        InvalidStrategyError: if fewer than 2 distinct compounds used
                              (F1 rules require 2 compound types minimum)
    """
    pass


def compare_strategies(
    strategies: List[Strategy],
    **kwargs
) -> pd.DataFrame:
    """
    Run simulate_race() for each strategy and rank by total race time.

    Returns:
        DataFrame with columns: strategy_id, n_stops, compounds, total_time_sec
        Sorted by total_time_sec ascending (best strategy first)
    """
    pass


def validate_strategy(strategy: Strategy, circuit_params: dict) -> None:
    """
    Validate a strategy before simulation.

    Rules:
      - Stints must be contiguous and cover all laps (1 to total_laps)
      - At least 2 distinct compounds must be used (FIA regulation)
      - No compound may be used beyond its max_stint_laps from config

    Raises:
        InvalidStrategyError with descriptive message
    """
    pass
```

**File:** `src/simulation/traffic_model.py`

```python
def compute_traffic_delta(
    gap_to_car_ahead: float,
    circuit: str,
    circuit_params: dict,
    config: dict
) -> float:
    """
    Compute time penalty from dirty air / traffic.

    Logic:
      - gap < 1.0s: penalty = overtaking_difficulty * random.uniform(0.3, 0.8)
      - gap 1.0–3.0s: partial penalty, linearly scaled
      - gap > 3.0s: penalty = 0.0

    Args:
        gap_to_car_ahead: Current gap in seconds
        circuit: Circuit name for overtaking difficulty lookup
        circuit_params: Circuit config dict
        config: Main config dict

    Returns:
        Time penalty in seconds to add to lap time
    """
    pass
```

**File:** `src/simulation/safety_car.py`

```python
def sample_safety_car(
    lap: int,
    circuit: str,
    circuit_params: dict,
    rng: np.random.Generator
) -> str:
    """
    Probabilistically determine safety car status for a given lap.

    Returns:
        'NONE' | 'SC' | 'VSC'

    Probabilities:
      - P(SC) = circuit_params[circuit]['sc_probability_per_lap']
      - P(VSC) = 0.5 * P(SC)  [VSC is more common than full SC]
      - P(NONE) = 1 - P(SC) - P(VSC)
    """
    pass


def apply_safety_car_lap_time(
    base_lap_time: float,
    sc_status: str,
    config: dict
) -> float:
    """
    Adjust lap time for safety car / VSC conditions.

    SC:  lap_time = base_lap_time * config['simulation']['sc_lap_time_multiplier']
    VSC: lap_time = base_lap_time * config['simulation']['vsc_lap_time_multiplier']

    Returns:
        Adjusted lap time in seconds
    """
    pass
```

---

### 7.7 Monte Carlo Simulation

**File:** `src/simulation/monte_carlo.py`

```python
def run_monte_carlo(
    strategy: Strategy,
    n_simulations: int,
    lap_time_model,
    scaler,
    deg_coefficients: dict,
    pit_deltas: dict,
    circuit_params: dict,
    config: dict
) -> dict:
    """
    Run n_simulations stochastic race simulations for a single strategy.

    Stochastic variables per simulation:
      - Lap time noise: N(0, config['simulation']['lap_time_noise_sigma']) per lap
      - Pit duration: N(mean, std) from config per pit stop
      - Safety car: Bernoulli per lap using circuit SC probability
      - Tyre degradation noise: ±10% variation on deg coefficients

    Args:
        strategy: Strategy to simulate
        n_simulations: Number of iterations (default 5000)
        ... (other model artifacts)

    Returns:
        {
            'race_times': list[float],   # n_simulations values
            'mean': float,
            'median': float,
            'std': float,
            'p10': float,                # 10th percentile
            'p90': float,                # 90th percentile
        }

    Performance target: complete 5000 simulations in < 500ms
    (Optimize with numpy vectorization where possible)
    """
    pass


def compare_strategies_monte_carlo(
    strategies: List[Strategy],
    n_simulations: int,
    **kwargs
) -> pd.DataFrame:
    """
    Run Monte Carlo for each strategy and compute win probabilities.

    Win probability: fraction of simulations where strategy A has
    lower total_time than all other strategies.

    Returns:
        DataFrame with: strategy_id, mean_time, p10, p90, win_probability_pct
        Sorted by win_probability_pct descending
    """
    pass
```

---

### 7.8 Reinforcement Learning Agent

**File:** `src/rl_agent/f1_env.py`

```python
import gymnasium as gym
import numpy as np

class F1RaceEnv(gym.Env):
    """
    Custom Gymnasium environment for F1 race strategy optimization.

    Observation Space (12 continuous values, all normalized to [0, 1]):
        [current_lap_norm, tyre_age_norm, compound_encoded,
         position_norm, gap_to_leader_norm, gap_to_car_ahead_norm,
         tyre_degradation_rate, fuel_load_norm,
         track_temp_norm, safety_car_active,
         laps_remaining_norm, pit_stops_made_norm]

    Action Space (Discrete 5):
        0: Stay out
        1: Pit for SOFT
        2: Pit for MEDIUM
        3: Pit for HARD
        4: Pit for INTERMEDIATE

    Reward:
        Per-lap: +0.5 per position gained, -0.5 per position lost
        Pit stop: -pit_delta / 100
        Final: 100 - (finishing_position - 1) * 5
        Illegal action (pit on same lap as previous pit): -50, done=True
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, config: dict, circuit: str = None):
        super().__init__()
        self.config = config
        self.circuit = circuit

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(12,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(5)

    def reset(self, seed: int = None, options: dict = None):
        """
        Reset to a new episode.

        CRITICAL: Randomize circuit, starting position, and initial
        tyre compound in each reset so agent cannot memorize one race.

        Returns:
            Tuple of (observation: np.ndarray, info: dict)
        """
        pass

    def step(self, action: int):
        """
        Advance simulation by one lap given agent action.

        Args:
            action: Integer 0–4 (see Action Space above)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        pass

    def _get_observation(self) -> np.ndarray:
        """Build normalized observation vector from current race state."""
        pass

    def _compute_reward(
        self,
        action: int,
        prev_position: int,
        curr_position: int,
        is_final_lap: bool
    ) -> float:
        """Compute reward using reward function defined in class docstring."""
        pass
```

**File:** `src/rl_agent/train_agent.py`

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

def train_ppo_agent(config: dict) -> None:
    """
    Train PPO agent on F1RaceEnv for config['rl']['total_timesteps'] steps.

    Training setup:
      - Wrap F1RaceEnv with DummyVecEnv (required by SB3)
      - PPO hyperparameters from config['rl']
      - EvalCallback every config['rl']['eval_freq'] steps
      - TensorBoard logging to runs/ directory
      - MLflow experiment logging

    Saves best model to src/rl_agent/saved/f1_ppo_agent.zip

    Args:
        config: Parsed config.yaml

    Example:
        >>> train_ppo_agent(config)
        Training PPO... 500000/500000 steps
        Best model saved.
    """
    pass
```

**File:** `src/rl_agent/evaluate_agent.py`

```python
def evaluate_agent(
    model_path: str,
    config: dict,
    n_episodes: int = 200
) -> dict:
    """
    Evaluate trained RL agent over n_episodes with deterministic=True.

    Metrics computed:
      - mean_finishing_position
      - std_finishing_position
      - win_rate (finished P1)
      - points_scored (F1 points system)
      - comparison vs best static strategy from Monte Carlo

    Returns:
        Dict of evaluation metrics

    Side effects:
        Saves evaluation results to outputs/tables/rl_evaluation.csv
    """
    pass


def visualize_agent_strategy(
    model_path: str,
    circuit: str,
    config: dict
) -> None:
    """
    Run one deterministic episode and visualize the agent's decisions.

    Plot: Tyre compound vs lap number (what compound was used and when)
    Plot: Cumulative position change across race
    Save both to docs/figures/rl_strategy_{circuit}.png
    """
    pass
```

---

### 7.9 FastAPI Backend

**File:** `src/api/main.py`

```python
from fastapi import FastAPI
from src.api.routes import strategy, simulation, data

app = FastAPI(
    title="F1 Strategy Optimizer API",
    description="AI-powered race strategy recommendations for Formula 1",
    version="1.0.0"
)

app.include_router(strategy.router, prefix="/strategy", tags=["Strategy"])
app.include_router(simulation.router, prefix="/simulation", tags=["Simulation"])
app.include_router(data.router, prefix="/data", tags=["Data"])

@app.on_event("startup")
async def load_models():
    """Load all ML model artifacts into memory at startup."""
    pass
```

**File:** `src/api/schemas.py`

```python
from pydantic import BaseModel, Field
from typing import List

class StintInput(BaseModel):
    compound: str = Field(..., pattern="^(SOFT|MEDIUM|HARD|INTERMEDIATE|WET)$")
    start_lap: int = Field(..., ge=1)
    end_lap: int = Field(..., ge=1)

class SimulationRequest(BaseModel):
    driver: str
    circuit: str
    stints: List[StintInput]

class RaceStateInput(BaseModel):
    circuit: str
    lap_number: int
    position: int
    tyre_compound: str
    tyre_age: int
    gap_to_leader: float
    safety_car_active: bool = False

class MonteCarloRequest(BaseModel):
    circuit: str
    stints: List[StintInput]
    n_simulations: int = Field(default=5000, le=10000)

class SimulationResult(BaseModel):
    total_time_sec: float
    lap_times: List[float]
    pit_laps: List[int]
    n_stops: int
    compounds_used: List[str]

class StrategyRecommendation(BaseModel):
    rank: int
    stints: List[StintInput]
    predicted_time_sec: float
    win_probability_pct: float
    time_advantage_vs_baseline_sec: float
```

**File:** `src/api/routes/strategy.py`

```python
@router.post("/recommend", response_model=List[StrategyRecommendation])
async def recommend_strategy(race_state: RaceStateInput):
    """
    Given current race state, return top 3 recommended strategies.

    Logic:
      1. Generate candidate strategies: all valid 1-stop and 2-stop combos
         for remaining laps given current compound and position
      2. Run simulate_race() for each candidate
      3. Run compare_strategies_monte_carlo() to get win probabilities
      4. Return top 3 by win_probability_pct

    Response time target: < 200ms P95
    """
    pass

@router.get("/presets/{circuit}")
async def get_strategy_presets(circuit: str):
    """
    Return historically common strategies for given circuit.
    Sourced from historical FastF1 data.
    """
    pass
```

---

### 7.10 React Dashboard

**Component: `RaceControlPanel.jsx`**

```jsx
// Props: none (self-contained)
// State: selectedCircuit, selectedDriver, stints[], simulationResult
// API calls: POST /simulation/run

// UI elements:
// - Circuit selector dropdown (GET /data/circuits)
// - Driver selector with team colour indicator
// - Stint builder: Add Stint button → compound dropdown + lap range slider
// - Validate Strategy button (check FIA 2-compound rule client-side)
// - Simulate Race button → loading spinner → result panel
// - Result: total race time, position timeline (Recharts LineChart)
```

**Component: `StrategyComparison.jsx`**

```jsx
// Props: none
// State: strategies[], monteCarloResults
// API calls: POST /simulation/monte-carlo for each strategy

// UI elements:
// - Add Strategy panel (reuse stint builder)
// - Run Comparison button
// - Recharts HistogramChart: overlapping distributions per strategy
// - Win probability table: strategy vs win% vs mean time vs P10-P90
// - Best strategy highlighted with green border
// - Tyre compound colour coding: Red=Soft, Yellow=Medium, White=Hard, Green=Inter
```

**Component: `LiveStrategyAdvisor.jsx`**

```jsx
// Props: none
// State: raceState{}, recommendations[]
// API calls: POST /strategy/recommend

// UI elements:
// - Current lap number input
// - Current position selector (1-20)
// - Current tyre: compound + age inputs
// - Gap to leader input
// - Safety car toggle
// - Get Recommendation button
// - Recommendation cards: top 3 strategies with predicted advantage
// - Colour-coded urgency: red if tyre age critical, green if optimal window
```

**Component: `HistoricalExplorer.jsx`**

```jsx
// Props: none
// State: selectedCircuit, selectedYear, lapTimeData, degradationData
// API calls: GET /data/lap-times/{circuit}/{year}

// UI elements:
// - Circuit + year selector
// - Lap time distribution: Recharts BoxPlot per compound
// - Tyre degradation curves: Recharts LineChart per compound
//   (x-axis: TyreAge, y-axis: DeltaFromBase seconds)
// - Pit stop heatmap: lap number vs frequency (Recharts BarChart)
```

---

## 8. Output Contracts

> Every module MUST save its outputs to the exact paths below. Filenames must not change.

| Output | Path | Format | Produced By |
|--------|------|--------|-------------|
| Raw laps (Hive-partitioned) | `data/raw/laps/season={y}/event={e}/part.parquet` | Parquet | `load_race_sessions.py` |
| Raw weather (Hive-partitioned) | `data/raw/weather/season={y}/event={e}/part.parquet` | Parquet | `load_race_sessions.py` |
| Raw results (Hive-partitioned) | `data/raw/results/season={y}/event={e}/part.parquet` | Parquet | `load_race_sessions.py` |
| Raw race control (Hive-partitioned) | `data/raw/race_control/season={y}/event={e}/part.parquet` | Parquet | `load_race_sessions.py` |
| Cleaned laps | `data/processed/laps_clean_{circuit}_{year}.parquet` | Parquet | `cleaner.py` |
| All circuits combined | `data/processed/laps_all_circuits.parquet` | Parquet | `cleaner.py` |
| Dropped laps log | `data/processed/dropped_laps_log.csv` | CSV | `cleaner.py` |
| Tyre deg coefficients | `data/processed/tyre_deg_coefficients.json` | JSON | `tyre_degradation.py` |
| Pit deltas | `config/pit_deltas.yaml` | YAML | `pit_delta.py` |
| Best lap time model | `src/models/saved/laptime_model.pkl` | Pickle | `train_laptime.py` |
| Feature scaler | `src/models/saved/scaler.pkl` | Pickle | `train_laptime.py` |
| Model comparison | `outputs/tables/model_comparison.csv` | CSV | `train_laptime.py` |
| RL trained agent | `src/rl_agent/saved/f1_ppo_agent.zip` | ZIP | `train_agent.py` |
| RL evaluation results | `outputs/tables/rl_evaluation.csv` | CSV | `evaluate_agent.py` |
| EDA figures | `docs/figures/eda_*.png` | PNG 300dpi | `notebooks/01_eda.ipynb` |
| Model eval figures | `docs/figures/model_*.png` | PNG 300dpi | `notebooks/02_model_evaluation.ipynb` |
| RL strategy figures | `docs/figures/rl_strategy_{circuit}.png` | PNG 300dpi | `evaluate_agent.py` |
| Simulation demo figures | `docs/figures/simulation_*.png` | PNG 300dpi | `notebooks/03_simulation_demo.ipynb` |

---

## 9. Model Performance Targets

| Metric | Target | Measurement | Action if Missed |
|--------|--------|-------------|-----------------|
| Lap Time MAE | < 0.5 seconds | Mean Absolute Error on test races | Tune XGBoost, add circuit-specific features |
| Lap Time R² | > 0.92 | Coefficient of determination | Check data leakage, re-examine feature set |
| Tyre Deg RMSE | < 0.3 seconds | RMSE of DeltaFromBase | Increase polynomial degree to 3 |
| Pit Delta Error | < 0.5 seconds | Mean error vs actual pit durations | Stratify by team (pit crew speed varies) |
| RL Mean Position | < 8.0 | Average finishing position over 200 episodes | Tune reward scaling, increase training steps |
| Simulation Speed | < 500ms | Time for 5000 Monte Carlo runs | Vectorize with NumPy, profile bottlenecks |
| API P95 Latency | < 200ms | 95th percentile for /strategy/recommend | Add in-memory model caching at startup |
| Strategy Validation | 3 of 5 races correct | Simulated winner matches actual winner | Review traffic and SC modelling |

---

## 10. Error Handling Rules

```python
# src/exceptions.py — define all custom exceptions here

class DataIngestionError(Exception):
    """Raised when API call fails after all retries."""
    pass

class InvalidStrategyError(Exception):
    """Raised when strategy violates FIA rules or simulation constraints."""
    pass

class ModelPerformanceError(Exception):
    """Raised when trained model fails to meet performance targets."""
    pass

class InsufficientDataError(Exception):
    """Raised when a circuit has too few laps for reliable modelling."""
    pass

class TyreAgeExtrapolationError(Exception):
    """Raised when tyre degradation model is asked to extrapolate beyond training data."""
    pass
```

### Retry Wrapper

```python
import time, logging
from functools import wraps

logger = logging.getLogger(__name__)

def with_retry(retries: int = 3, delay: float = 5.0, backoff: float = 2.0):
    """
    Decorator for retrying functions that may fail transiently.

    Args:
        retries: Number of retry attempts
        delay: Initial delay in seconds
        backoff: Multiplier applied to delay each retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(1, retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"[{func.__name__}] Attempt {attempt}/{retries} failed: {e}")
                    if attempt == retries:
                        logger.error(f"[{func.__name__}] All retries exhausted. Raising.")
                        raise
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator

# Usage:
@with_retry(retries=3, delay=5.0)
def load_session(...):
    ...
```

---

## 11. Testing Specification

### `tests/ingestion/test_load_race_sessions.py`

```python
def test_normalize_event_name_replaces_spaces_and_strips_symbols():
    """'Bahrain Grand Prix' → 'Bahrain_Grand_Prix', 'São Paulo Grand Prix' → 'Sao_Paulo_Grand_Prix'."""

def test_build_output_path_uses_table_season_and_event_partitions():
    """Path format: data/raw/{table}/season={y}/event={e}/part.parquet."""

def test_add_session_metadata_appends_required_columns():
    """season, event_name, round_number, session_name, session_date columns added."""

def test_extract_session_tables_returns_expected_frames():
    """Returns dict with keys: laps, weather, results, race_control."""

def test_write_table_creates_partitioned_parquet_file():
    """Parquet file is written and round-trips correctly."""

def test_ingest_loaded_session_writes_all_expected_tables():
    """All four table partitions exist after ingest_loaded_session()."""

def test_ingest_seasons_continues_when_one_session_fails():
    """Failed session does not halt the entire batch; summary reflects failures."""

def test_parse_args_defaults_to_2021_through_2024():
    """Default year range covers 2021–2024."""

def test_parse_args_rejects_removed_data_root_option():
    """--data-root flag is no longer accepted."""
```

### `tests/test_session_load.py`

```python
def test_session_load_runs_from_script_directory():
    """session_load.py executes without error."""
```

### `tests/test_cleaner.py`

```python
def test_null_laptimes_are_dropped():
    """Rows with null LapTime must be removed."""

def test_outliers_are_removed_and_logged():
    """Laps above (median + 2.5 * IQR) must be removed; log file must exist."""

def test_timedelta_conversion_to_seconds():
    """LapTime timedelta of 1min 30sec must become 90.0 float."""

def test_compound_normalization():
    """'soft', 'SOFT', '1' must all become 'SOFT' after cleaning."""

def test_pit_laps_preserved():
    """IsPitLap=True rows must remain in DataFrame after cleaning."""

def test_no_nulls_in_output():
    """Cleaned DataFrame must have zero NaN values."""
```

### `tests/test_features.py`

```python
def test_tyre_age_resets_at_pit():
    """TyreAge must reset to 1 the lap after IsPitLap=True."""

def test_fuel_load_decreases_per_lap():
    """FuelLoad must decrease monotonically with LapNumber."""

def test_fuel_effect_positive():
    """FuelEffect must be positive for all laps."""

def test_race_progress_bounded():
    """RaceProgress must be in [0.0, 1.0] for all laps."""

def test_compound_encoding_exhaustive():
    """All compound strings must map to an integer 0–4."""
```

### `tests/test_simulator.py`

```python
def test_deterministic_seed_produces_same_result():
    """Two runs with same seed and noise=True must produce identical race times."""

def test_invalid_strategy_raises_error():
    """Strategy with gap in stints must raise InvalidStrategyError."""

def test_single_compound_strategy_raises_error():
    """Strategy using only one compound must raise InvalidStrategyError (FIA rule)."""

def test_pit_lap_adds_pit_delta():
    """Total race time must be greater than sum of clean lap times by at least pit_delta."""

def test_strategy_comparison_sorted():
    """compare_strategies() output must be sorted by total_time_sec ascending."""
```

### `tests/test_api.py`

```python
from fastapi.testclient import TestClient
from src.api.main import app
client = TestClient(app)

def test_recommend_endpoint_returns_three_strategies():
    """POST /strategy/recommend must return exactly 3 recommendations."""

def test_simulation_run_returns_lap_times():
    """POST /simulation/run must return list of lap times equal to total_laps."""

def test_monte_carlo_returns_distribution():
    """POST /simulation/monte-carlo must return mean, p10, p90 keys."""

def test_circuits_endpoint_returns_list():
    """GET /data/circuits must return non-empty list."""

def test_invalid_compound_returns_422():
    """POST /simulation/run with compound='INVALID' must return HTTP 422."""
```

---

## 12. Coding Standards

```yaml
style:
  formatter: black
  max_line_length: 88
  import_sorter: isort
  type_hints: required on all public functions
  docstrings: Google style (Args, Returns, Raises, Example, Note)

logging:
  module: logging (NEVER print)
  format: "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
  levels:
    INFO: routine steps (session loaded, model trained, shape confirmed)
    WARNING: data quality issues (fills, drops, API retry)
    ERROR: recoverable failures (fallback to cache, model below target)
    CRITICAL: unrecoverable — raise exception then log

notebooks:
  first_cell: imports + config load + logging setup
  each_section_header: markdown cell explaining what follows + what to look for
  save_all_figures: every plot saved to docs/figures/ via plt.savefig()
  no_hardcoded_paths: always reference config values
  clear_outputs_before_commit: strip notebook outputs from git

api:
  all_endpoints_documented: OpenAPI docstrings required
  all_inputs_validated: Pydantic schemas required
  no_model_loading_in_routes: load at startup via lifespan event
  cors_enabled: allow localhost:3000 for React dev

react:
  state_management: React useState and useReducer only
  data_fetching: React Query (useQuery, useMutation)
  error_states: every API call must show error UI if request fails
  loading_states: every async operation must show spinner
  tyre_colours:
    SOFT: "#E8002D"         # F1 red
    MEDIUM: "#FFF200"       # F1 yellow
    HARD: "#FFFFFF"         # F1 white
    INTERMEDIATE: "#39B54A" # F1 green
    WET: "#0067FF"          # F1 blue

git:
  commit_convention: "type(scope): message"
  types: [feat, fix, data, model, sim, rl, api, dash, docs, test, refactor]
  examples:
    - "feat(ingestion): implement bulk season collector with retry"
    - "fix(cleaner): correct compound normalization for numeric codes"
    - "model(laptime): add XGBoost with 5-fold CV, MAE=0.42"
    - "rl(env): implement reward function with position-gain bonus"

gitignore_must_include:
  - f1_venv/                       # Project virtualenv
  - f1_strategy/data/              # Raw + processed + cache (large binary data)
  - src/models/saved/              # Serialized model weights
  - src/rl_agent/saved/            # Trained RL agent weights
  - __pycache__/
  - .ipynb_checkpoints/
  - .pytest_cache/
  - runs/                          # TensorBoard logs
  - mlruns/                        # MLflow runs
```

---

## 13. Agent Task Queue

> Ordered execution plan. Each task has strict dependencies — do not start a task until all dependencies are marked complete.

```yaml
tasks:

  - id: T01
    name: "Project scaffold"
    file: "setup script / manual"
    depends_on: []
    steps:
      - Create all directories per section 3
      - Create requirements.txt from section 4
      - Create config.yaml, circuit_params.yaml, pit_deltas.yaml stubs
      - Create all __init__.py files
      - Initialise git repo with .gitignore
    acceptance:
      - All directories exist
      - config.yaml loads without YAML errors
      - git status shows clean working tree after initial commit

  - id: T02
    name: "FastF1 session loader + bulk collection"
    file: "src/ingestion/load_race_sessions.py, collect_missing.py"
    depends_on: [T01]
    steps:
      - Implement ingest_seasons() with Hive-partitioned Parquet output
      - Implement collect_missing() for incremental backfill
      - Verify all four tables (laps, weather, results, race_control) are written
      - Verify partition columns (season, event_name, round_number, session_name, session_date)
    acceptance:
      - Data exists for all 2021-2024 seasons in data/raw/{table}/season={y}/event={e}/part.parquet
      - laps table has LapTime, Compound, TyreLife, Driver, DriverNumber, Team columns
      - All four tables load via pd.read_parquet() without errors

  - id: T03
    name: "Data cleaning"
    file: "src/preprocessing/cleaner.py"
    depends_on: [T02]
    steps:
      - Implement clean_laps() with all steps from spec
      - Implement convert_timedeltas()
      - Verify no nulls in output
    acceptance:
      - Output DataFrame has zero NaN
      - dropped_laps_log.csv exists
      - LapTimeSec is float64 column in [50, 200] range
      - IsPitLap column exists as boolean

  - id: T04
    name: "Feature engineering"
    file: "src/preprocessing/feature_engineer.py"
    depends_on: [T03]
    steps:
      - Implement all add_*_features() functions
      - Implement merge_weather_features() with fallback
      - Validate output with assert block from section 5
    acceptance:
      - All 20 feature columns present in output
      - TyreAge resets correctly at pit boundaries
      - FuelLoad decreases with LapNumber
      - RaceProgress bounded [0, 1]

  - id: T05
    name: "EDA notebook"
    file: "notebooks/01_eda.ipynb"
    depends_on: [T04]
    steps:
      - All 6 EDA analyses from spec (distributions, degradation, pit timing, correlation, weather, circuit comparison)
      - Save all plots to docs/figures/eda_*.png at 300 DPI
    acceptance:
      - All figures exist in docs/figures/
      - Notebook runs top-to-bottom without errors

  - id: T06
    name: "Lap time model"
    file: "src/models/train_laptime.py"
    depends_on: [T04]
    steps:
      - Implement split_by_race() — NOT random split
      - Implement train_all_models() for all 5 models
      - Implement select_and_save_best_model()
      - Log to MLflow
    acceptance:
      - Best model MAE < 0.5 on test set
      - Best model R² > 0.92 on test set
      - laptime_model.pkl and scaler.pkl saved
      - model_comparison.csv has 5 rows

  - id: T07
    name: "Tyre degradation model"
    file: "src/models/tyre_degradation.py"
    depends_on: [T04]
    steps:
      - Implement fit_degradation_curves() for all circuit/compound pairs
      - Implement predict_degradation_delta() with max_tyre_age guard
    acceptance:
      - tyre_deg_coefficients.json has entries for all circuits × compounds
      - Degradation delta is non-negative for TyreAge > 5
      - TyreAgeExtrapolationError raised beyond max_tyre_age

  - id: T08
    name: "Pit delta model"
    file: "src/models/pit_delta.py"
    depends_on: [T03]
    steps:
      - Implement compute_pit_deltas()
      - Save results to config/pit_deltas.yaml
    acceptance:
      - All circuits have mean pit delta between 15s and 35s
      - pit_deltas.yaml is valid YAML

  - id: T09
    name: "Deterministic race engine"
    file: "src/simulation/race_engine.py"
    depends_on: [T06, T07, T08]
    steps:
      - Implement simulate_race() with all steps from spec
      - Implement validate_strategy() with FIA 2-compound check
      - Implement compare_strategies()
    acceptance:
      - Simulate Bahrain 2023 with actual Mercedes strategy — output within ±30s of real result
      - Single-compound strategy raises InvalidStrategyError
      - compare_strategies output sorted ascending

  - id: T10
    name: "Traffic, SC, Monte Carlo"
    file: "src/simulation/traffic_model.py, safety_car.py, monte_carlo.py"
    depends_on: [T09]
    steps:
      - Implement compute_traffic_delta()
      - Implement sample_safety_car() and apply_safety_car_lap_time()
      - Implement run_monte_carlo() with all stochastic variables
      - Implement compare_strategies_monte_carlo()
    acceptance:
      - 5000 simulations complete in < 500ms
      - Output has mean, median, p10, p90 keys
      - Win probabilities across all strategies sum to ~100%

  - id: T11
    name: "RL environment"
    file: "src/rl_agent/f1_env.py"
    depends_on: [T09]
    steps:
      - Implement F1RaceEnv (reset, step, _get_observation, _compute_reward)
      - Verify gym.utils.env_checker(env) passes
      - Randomize circuit/position in reset()
    acceptance:
      - env_checker passes with no errors
      - reset() returns observation shape (12,)
      - step() returns (obs, reward, terminated, truncated, info)
      - Illegal action (double pit) returns done=True with -50 reward

  - id: T12
    name: "RL training"
    file: "src/rl_agent/train_agent.py"
    depends_on: [T11]
    steps:
      - Implement train_ppo_agent() with EvalCallback
      - Train for 500,000 timesteps
      - Save best model
    acceptance:
      - f1_ppo_agent.zip saved and loadable
      - Mean reward in final 50k steps positive
      - TensorBoard logs exist in runs/

  - id: T13
    name: "RL evaluation"
    file: "src/rl_agent/evaluate_agent.py"
    depends_on: [T12]
    steps:
      - Implement evaluate_agent() for 200 episodes
      - Implement visualize_agent_strategy()
      - Compare vs best static strategy
    acceptance:
      - Mean finishing position < 8.0
      - rl_evaluation.csv saved with all metrics
      - Strategy visualization figure saved per circuit

  - id: T14
    name: "FastAPI backend"
    file: "src/api/"
    depends_on: [T10, T13]
    steps:
      - Implement main.py with model loading at startup
      - Implement all 5 endpoints from spec
      - Implement Pydantic schemas
    acceptance:
      - All endpoints return correct HTTP 200 responses
      - Invalid compound input returns HTTP 422
      - /docs Swagger page loads correctly
      - P95 latency < 200ms for /strategy/recommend

  - id: T15
    name: "React dashboard"
    file: "dashboard/"
    depends_on: [T14]
    steps:
      - Implement all 4 components from spec
      - Connect all API calls via React Query
      - Apply tyre colour coding
    acceptance:
      - All 4 pages load without console errors
      - Strategy comparison chart renders with overlapping histograms
      - Live advisor returns 3 recommendations

  - id: T16
    name: "Unit tests"
    file: "tests/"
    depends_on: [T03, T04, T09, T14]
    steps:
      - Implement all tests from section 11
    acceptance:
      - pytest tests/ -v returns 0 failures
      - Coverage > 70% for src/ modules

  - id: T17
    name: "Model evaluation notebook + strategy validation"
    file: "notebooks/02_model_evaluation.ipynb, 03_simulation_demo.ipynb"
    depends_on: [T06, T10, T13]
    steps:
      - Learning curves, residual plots, feature importance
      - Validate 5 historical 2023 races
    acceptance:
      - Simulated winner matches actual winner in ≥3 of 5 races
      - All figures saved to docs/figures/
```

---

## 14. Common Pitfalls & Guardrails

### Data Pitfalls

| Pitfall | Detection | Fix |
|---------|-----------|-----|
| Random row split causing data leakage | Model shows unrealistically high R² (>0.99) | Always split by race using split_by_race() |
| Out-lap / in-lap contaminating model | MAE spikes; slow laps pulling predictions down | Filter `IsPitLap=False & IsOutLap=False` before training |
| Compound numeric codes ('1', '2', '3') | TyreCompoundEncoded has values >4 | Normalize at ingestion: map numeric → string → encoded int |
| Missing weather data | NaN in AirTemp, TrackTemp after merge | Fallback to circuit averages from circuit_params.yaml |
| FastF1 version incompatibility | AttributeError on session attributes | Pin fastf1>=3.3.0, test with fresh cache |

### Modelling Pitfalls

| Pitfall | Detection | Fix |
|---------|-----------|-----|
| Scaler fitted on full dataset | Subtle inflation of test metrics | Assert scaler.fit() only called on train split |
| Single-circuit overfitting | Poor performance on held-out circuit | Evaluate per-circuit; ensure ≥5 different circuits in test set |
| Tyre age extrapolation | Negative or implausible degradation delta | max_tyre_age guard raises TyreAgeExtrapolationError |
| Feature importance misread | Assuming CircuitEncoded is causal | It captures track-specific pace — interpret as baseline, not causal |

### Simulation Pitfalls

| Pitfall | Detection | Fix |
|---------|-----------|-----|
| Circular dependency in traffic | Position update uses current-lap times not yet computed | Use previous lap positions to compute current traffic delta |
| Strategy space explosion | Runtime > 10s for compare_strategies | Limit to 1-stop and 2-stop strategies; 3-stop on request only |
| FIA regulation violation | Agent pits on lap 1 or uses 1 compound | validate_strategy() must gate all simulate_race() calls |
| Monte Carlo too slow | 5000 sims > 500ms | Vectorize noise sampling with np.random.Generator batch calls |

### RL Pitfalls

| Pitfall | Detection | Fix |
|---------|-----------|-----|
| Sparse reward → no learning | Mean reward stays near 0 | Add intermediate position-gain reward per lap |
| Reward scale imbalance | Agent ignores per-lap signals | Scale all rewards to similar magnitude (all in [-1, 1] + terminal bonus) |
| Memorizing one race | Perfect train performance, poor generalization | Randomize circuit, starting position, tyre compound in reset() |
| PPO not converging | Loss oscillates wildly | Reduce learning_rate to 1e-4, increase n_steps to 4096 |
| Invalid action exploitation | Agent always pits to maximize short-term reward | Mask actions at environment level using ActionMasks |

---

## 15. Prompt Templates for AI Agent Invocation

### Master Bootstrap Prompt

```
You are a senior Python engineer building an F1 race strategy optimization platform.

Before writing any code:
1. Read F1_STRATEGY_AGENT.md in full
2. Confirm you understand the 5-layer architecture (Data → Models → Simulation → RL → API/Dashboard)
3. Confirm the Agent Behaviour Rules from section 2
4. State which task from the Task Queue (T01–T17) you are starting

Non-negotiable rules:
- NEVER split data by random row — always by race
- NEVER use print() — always logging module
- NEVER hardcode F1 data — always fetch from FastF1
- ALWAYS write docstrings on every public function
- ALWAYS use config.yaml for all parameters
- ALWAYS save outputs to the exact paths in section 8
```

### Module Prompts

**T03 — Data Cleaning**
```
Implement src/preprocessing/cleaner.py per section 7.2 of F1_STRATEGY_AGENT.md.

Critical requirements:
- IsPitLap rows must be PRESERVED, only flagged — never dropped
- Outlier threshold: median + 2.5*IQR per circuit (not global)
- Compound normalization: 'soft'/'SOFT'/'1' all → 'SOFT'
- Save dropped rows log to data/processed/dropped_laps_log.csv
- Zero NaN allowed in final output

Test your implementation on Bahrain 2023 Race session.
Show shape before and after each cleaning step.
```

**T06 — Lap Time Model**
```
Implement src/models/train_laptime.py per section 7.3.

Critical requirements:
- split_by_race() splits on (year, circuit) groups — NOT random rows
- StandardScaler fitted on train split ONLY
- Train 5 models: LinearRegression, XGBoost, LightGBM, RandomForest, MLP
- MLP architecture: 256→128→64→1 with ReLU and Dropout(0.2)
- Log all experiments to MLflow
- Select best model by MAE on val set
- Target: MAE < 0.5, R² > 0.92 on test set
- Raise ModelPerformanceError if targets not met

Save laptime_model.pkl, scaler.pkl, model_comparison.csv.
```

**T09 — Race Simulation Engine**
```
Implement src/simulation/race_engine.py per section 7.6.

Critical requirements:
- validate_strategy() must check: continuous stints, all laps covered,
  ≥2 distinct compounds (FIA rule), no compound beyond max_stint_laps
- simulate_race() processes laps 1 to total_laps sequentially
- Traffic delta computed from PREVIOUS lap positions (avoid circular dependency)
- When noise=True, add N(0, 0.15) per lap time for Monte Carlo use
- Pit lap adds pit_delta from config/pit_deltas.yaml

Validate output against Bahrain 2023 Mercedes actual strategy.
Simulated total time must be within ±30 seconds of actual race time.
```

**T11 — RL Environment**
```
Implement src/rl_agent/f1_env.py as a Gymnasium environment per section 7.8.

Critical requirements:
- Inherit from gymnasium.Env exactly
- Observation: 12 floats, all normalized to [0, 1]
- Actions: Discrete(5) — 0=Stay out, 1=Pit SOFT, 2=Pit MEDIUM, 3=Pit HARD, 4=Pit INTER
- Reward: position delta per lap + pit penalty + terminal finishing position reward
- Illegal action (double pit same lap): -50 reward, terminated=True
- reset() MUST randomize: circuit, starting_position, initial_compound
  (otherwise agent memorizes one scenario)

Verify with: gymnasium.utils.env_checker.check_env(env)
The checker must pass with 0 errors before proceeding to T12 (RL training).
```

---

## 16. Notebook Templates

### `01_eda.ipynb` Cell Structure

```python
# Cell 1 — Setup
import yaml, pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
import sys; sys.path.append('..')
from src.preprocessing.cleaner import clean_laps
from src.preprocessing.feature_engineer import engineer_all_features

with open('../config/config.yaml') as f:
    config = yaml.safe_load(f)

plt.rcParams['figure.dpi'] = 150
SAVE_PATH = '../docs/figures/'
```

```python
# Cell 2 — Load raw data from Hive-partitioned Parquet
import pyarrow.parquet as pq

raw_root = Path('../data/raw')
seasons = [2021, 2022, 2023, 2024]
dfs = []
for season in seasons:
    laps_dir = raw_root / 'laps' / f'season={season}'
    if laps_dir.exists():
        for event_dir in sorted(laps_dir.iterdir()):
            p = event_dir / 'part.parquet'
            if p.exists():
                dfs.append(pd.read_parquet(p))
df = pd.concat(dfs, ignore_index=True)
print(f"Shape: {df.shape}")
print(f"Seasons: {df['season'].unique()}")
print(f"Events: {df['event_name'].nunique()}")
```

```python
# Cell 3 — Load processed data (once available)
df = pd.read_parquet('../data/processed/laps_all_circuits.parquet')
print(f"Shape: {df.shape}")
print(f"Circuits: {df['CircuitName'].nunique()}")
print(f"Seasons: {df['Year'].unique()}")
print(f"Null count: {df.isnull().sum().sum()}")
```

```python
# Cell 3 — Lap time distribution per circuit per compound
fig, axes = plt.subplots(4, 4, figsize=(20, 16))
for ax, circuit in zip(axes.flat, df['CircuitName'].unique()[:16]):
    subset = df[df['CircuitName'] == circuit]
    for compound, color in [('SOFT','red'),('MEDIUM','gold'),('HARD','gray')]:
        data = subset[subset['Compound']==compound]['LapTimeSec'].dropna()
        if len(data) > 0:
            ax.hist(data, bins=30, alpha=0.6, color=color, label=compound)
    ax.set_title(circuit, fontsize=9)
    ax.set_xlabel('Lap Time (s)')
plt.tight_layout()
plt.savefig(f'{SAVE_PATH}eda_laptimes_by_compound.png', dpi=300, bbox_inches='tight')
```

```python
# Cell 4 — Tyre degradation curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
circuits_to_plot = ['Bahrain', 'Silverstone', 'Monza']
for ax, circuit in zip(axes, circuits_to_plot):
    subset = df[(df['CircuitName'] == circuit) & (~df['IsPitLap']) & (~df['IsOutLap'])]
    for compound, color in [('SOFT','red'),('MEDIUM','gold'),('HARD','gray')]:
        data = subset[subset['Compound']==compound].groupby('TyreAge')['DeltaFromBase'].mean()
        if len(data) > 0:
            ax.plot(data.index, data.values, color=color, label=compound, linewidth=2)
    ax.set_title(f'{circuit} Tyre Degradation')
    ax.set_xlabel('Tyre Age (laps)')
    ax.set_ylabel('Time Delta vs Fresh Tyre (s)')
    ax.legend()
plt.tight_layout()
plt.savefig(f'{SAVE_PATH}eda_tyre_degradation.png', dpi=300, bbox_inches='tight')
```

```python
# Cell 5 — Correlation heatmap of features vs LapTimeSec
feature_cols = ['TyreAge','FuelLoad','FuelEffect','RaceProgress','AirTemp','TrackTemp']
corr = df[feature_cols + ['LapTimeSec']].corr()['LapTimeSec'].drop('LapTimeSec').sort_values()
fig, ax = plt.subplots(figsize=(10, 5))
corr.plot(kind='barh', ax=ax, color=['red' if x < 0 else 'steelblue' for x in corr])
ax.set_title('Feature Correlation with Lap Time')
ax.axvline(0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig(f'{SAVE_PATH}eda_feature_correlation.png', dpi=300, bbox_inches='tight')
```

---

### `03_simulation_demo.ipynb` Cell Structure

```python
# Cell 1 — Load simulation components
from src.simulation.race_engine import simulate_race, compare_strategies, Strategy, Stint
from src.simulation.monte_carlo import compare_strategies_monte_carlo
import joblib, json, yaml

laptime_model = joblib.load('../src/models/saved/laptime_model.pkl')
scaler = joblib.load('../src/models/saved/scaler.pkl')
with open('../data/processed/tyre_deg_coefficients.json') as f:
    deg_coeff = json.load(f)
with open('../config/pit_deltas.yaml') as f:
    pit_deltas = yaml.safe_load(f)
with open('../config/circuit_params.yaml') as f:
    circuit_params = yaml.safe_load(f)
with open('../config/config.yaml') as f:
    config = yaml.safe_load(f)
```

```python
# Cell 2 — Define candidate strategies for Bahrain
strategies = [
    Strategy(stints=[Stint('SOFT', 1, 18), Stint('MEDIUM', 19, 40), Stint('HARD', 41, 57)],
             driver='HAM', circuit='Bahrain'),
    Strategy(stints=[Stint('MEDIUM', 1, 25), Stint('HARD', 26, 57)],
             driver='HAM', circuit='Bahrain'),
    Strategy(stints=[Stint('SOFT', 1, 12), Stint('MEDIUM', 13, 35), Stint('SOFT', 36, 57)],
             driver='HAM', circuit='Bahrain'),
]
```

```python
# Cell 3 — Deterministic comparison
args = dict(lap_time_model=laptime_model, scaler=scaler,
            deg_coefficients=deg_coeff, pit_deltas=pit_deltas,
            circuit_params=circuit_params, config=config)

det_results = compare_strategies(strategies, **args)
print(det_results[['n_stops','compounds_used','total_time_sec']])
```

```python
# Cell 4 — Monte Carlo comparison with visualisation
mc_results = compare_strategies_monte_carlo(strategies, n_simulations=5000, **args)
print(mc_results[['strategy_id','mean_time','p10','p90','win_probability_pct']])

# Histogram plot
import plotly.graph_objects as go
fig = go.Figure()
colours = ['red','blue','green']
for i, (_, row) in enumerate(mc_results.iterrows()):
    fig.add_trace(go.Histogram(x=row['race_times'], name=f"Strategy {i+1}",
                               opacity=0.6, nbinsx=100, marker_color=colours[i]))
fig.update_layout(barmode='overlay', title='Monte Carlo Strategy Distribution — Bahrain')
fig.write_image('../docs/figures/simulation_monte_carlo_bahrain.png', scale=3)
fig.show()
```

---

## 17. Interpretation & Validation Guide

### Strategy Validation Against Real Races

```python
# Validation template — fill in after running simulation
VALIDATION_RACES = [
    {"year": 2023, "circuit": "Bahrain",    "actual_winner": "VER", "actual_strategy": "2-stop: M-H-M"},
    {"year": 2023, "circuit": "Monaco",     "actual_winner": "ALO", "actual_strategy": "1-stop: M-H"},
    {"year": 2023, "circuit": "Silverstone","actual_winner": "HAM", "actual_strategy": "2-stop: H-M-S"},
    {"year": 2023, "circuit": "Monza",      "actual_winner": "SAI", "actual_strategy": "2-stop: S-M-H"},
    {"year": 2023, "circuit": "Abu Dhabi",  "actual_winner": "VER", "actual_strategy": "1-stop: M-H"},
]
# Target: simulated winner matches actual winner in >= 3 of 5 races
```

### Report Narrative Templates

**Lap Time Model:**
```
The gradient boosting model (XGBoost) achieved the lowest test MAE of [X] seconds
(R² = [X]), outperforming the linear regression baseline (MAE = [X]) by [X]%.

Feature importance analysis reveals that TyreAge and FuelEffect are the
dominant predictors, followed by CircuitEncoded — confirming that track-specific
pace differentials are a primary driver of lap time variance.

The neural network (MLP) achieved comparable accuracy (MAE = [X]) but required
significantly longer training time, making the gradient boosting model preferred
for deployment.
```

**Monte Carlo Results:**
```
For [circuit], the [N]-stop [compound] strategy achieved a win probability
of [X]% across 5,000 simulations, compared to [Y]% for the alternative
[N]-stop strategy. The P10–P90 range of [X]–[X] seconds indicates
[low/moderate/high] uncertainty in the outcome.

The overlap region of [X]% in the distribution suggests that under [X]%
of race scenarios, the lower-ranked strategy would outperform — primarily
driven by safety car occurrences at laps [X]–[X] where an early pit
becomes advantageous.
```

**RL Agent:**
```
After 500,000 training timesteps, the PPO agent achieved a mean finishing
position of [X] across 200 evaluation episodes, compared to [X] for the
best static strategy identified by Monte Carlo simulation.

The agent demonstrates adaptive behaviour under safety car conditions:
in [X]% of episodes with a safety car event, the agent correctly brought
forward its pit stop by [X] laps compared to its baseline policy.

Strategy profiles show the agent predominantly uses [compound] as a
starting tyre at [circuit], consistent with historical team strategies,
validating that the reward function correctly captures race dynamics.
```

---

## 18. API Contract Reference

### POST /strategy/recommend

**Request:**
```json
{
  "circuit": "Bahrain",
  "lap_number": 23,
  "position": 4,
  "tyre_compound": "MEDIUM",
  "tyre_age": 18,
  "gap_to_leader": 12.4,
  "safety_car_active": false
}
```

**Response:**
```json
[
  {
    "rank": 1,
    "stints": [
      {"compound": "MEDIUM", "start_lap": 23, "end_lap": 23},
      {"compound": "HARD", "start_lap": 24, "end_lap": 57}
    ],
    "predicted_time_sec": 5847.3,
    "win_probability_pct": 42.1,
    "time_advantage_vs_baseline_sec": 3.7
  }
]
```

### POST /simulation/monte-carlo

**Request:**
```json
{
  "circuit": "Silverstone",
  "stints": [
    {"compound": "SOFT", "start_lap": 1, "end_lap": 16},
    {"compound": "MEDIUM", "start_lap": 17, "end_lap": 52}
  ],
  "n_simulations": 5000
}
```

**Response:**
```json
{
  "mean": 5923.4,
  "median": 5921.1,
  "std": 18.7,
  "p10": 5901.2,
  "p90": 5948.6,
  "win_probability_vs_alternatives_pct": null
}
```

---

## 19. Extensions (Post-MVP)

### Extension 1 — Live Race Integration

```yaml
description: >
  Connect to official F1 Live Timing data stream (available via FastF1
  livetiming module) to update strategy recommendations in real-time
  during an active race.
effort: High
resume_impact: Very High — production-grade live system
```

### Extension 2 — Multi-Car Strategy (Team Optimization)

```yaml
description: >
  Optimize strategy for both cars simultaneously. Car A pitting influences
  Car B's traffic situation. Requires multi-agent RL (MAPPO or QMIX).
effort: High
resume_impact: Very High — genuinely novel academic contribution
```

### Extension 3 — Weather-Adaptive Strategy

```yaml
description: >
  Integrate Met Office weather forecast API to predict rainfall probability
  during a race. RL agent learns to pre-empt wet weather with early
  intermediate tyre pit.
effort: Medium
resume_impact: High — realistic feature, easily demonstrable
```

### Extension 4 — Spark Pipeline for Historical Processing

```yaml
description: >
  Replace pandas bulk collection with PySpark pipeline for processing
  all historical seasons in parallel. Demonstrates Big Data skills
  directly aligned with your specialization.
effort: Medium
resume_impact: High for Big Data Analytics programme specifically
libraries: pyspark>=3.5.0
```

### Extension 5 — MLflow Model Registry + Docker Deployment

```yaml
description: >
  Register trained models in MLflow Model Registry. Package API in Docker.
  Deploy to cloud (GCP Cloud Run or AWS ECS). Accessible via public URL.
effort: Medium
resume_impact: Very High — deployed project is 10x more impressive than local
```

---

## 20. Git Workflow & Deliverables Checklist

### Branch Strategy

```bash
main          ← stable, submission-ready code only
dev           ← integration branch, must pass all tests
feature/*     ← individual module branches

# Example workflow for T09 (race engine)
git checkout -b feature/T09-race-engine
# implement src/simulation/race_engine.py + tests
git add src/simulation/race_engine.py tests/test_simulator.py
git commit -m "feat(sim): implement deterministic race engine with FIA validation"
git checkout dev
git merge feature/T09-race-engine
git push origin dev
```

### Commit Examples

```
feat(ingestion): implement FastF1 bulk season collector with exponential retry
fix(cleaner): normalize numeric compound codes to string at ingestion
data(features): add TyreAge reset logic per driver per stint
model(laptime): XGBoost achieves MAE=0.43s, R2=0.94 on test races
sim(monte-carlo): vectorize noise sampling, 5000 sims in 320ms
rl(env): implement position-gain reward with safety car adaptive bonus
api(strategy): add /recommend endpoint with top-3 Monte Carlo ranking
dash(comparison): add overlapping Monte Carlo histogram with Recharts
test(simulator): add FIA 2-compound violation and strategy gap tests
docs(readme): add architecture diagram and demo screenshots
```

### Final Deliverables Checklist

**Code Repository**
- [ ] All source code committed with meaningful messages
- [ ] README.md with setup instructions, architecture diagram, demo screenshots
- [ ] requirements.txt with pinned versions
- [ ] All tests passing: `pytest tests/ -v --cov=src` → 0 failures, ≥70% coverage
- [ ] API running, all endpoints correct
- [ ] React dashboard 4 pages working

**Models & Data**
- [ ] `laptime_model.pkl` saved and loads correctly
- [ ] `scaler.pkl` saved and loads correctly
- [ ] `tyre_deg_coefficients.json` covers all circuits × compounds
- [ ] `f1_ppo_agent.zip` saved and evaluates at mean position < 8.0
- [ ] MLflow experiment runs recorded with training curves

**Documentation**
- [ ] `notebooks/01_eda.ipynb` — all 6 analyses complete, all figures saved
- [ ] `notebooks/02_model_evaluation.ipynb` — learning curves, residuals, feature importance
- [ ] `notebooks/03_simulation_demo.ipynb` — strategy comparison + Monte Carlo
- [ ] Strategy validation: ≥3/5 historical races correct
- [ ] API docs at `/docs` endpoint correct and complete

**Presentation**
- [ ] 8–10 slide deck: problem → architecture → results → demo
- [ ] 3–5 minute demo video: pipeline → predictions → simulation → dashboard
- [ ] Results table: RL agent vs static strategies vs actual race outcomes

---

### Resume Bullet

> *"Built end-to-end F1 race strategy optimization platform using FastF1 telemetry, XGBoost lap time prediction (MAE < 0.5s), Monte Carlo simulation (5,000 iterations), and a PPO reinforcement learning agent (SB3/Gymnasium); deployed via FastAPI + React dashboard with real-time strategy recommendations."*

---

*F1 Strategy Agent Guide | Sections: 20 | Tasks: 18 | Modules: 10 | API Endpoints: 5*  
*Big Data Analytics Minor · School of Computer Engineering*
