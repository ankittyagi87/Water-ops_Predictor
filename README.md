# Firefighting Risk Prediction System

## Overview
This system ingests telemetry from pumps, nozzles, and hydrants, combines it with risk labels, and trains XGBoost models to predict firefighting risks. It is organized into three phases:

1. **Ingestion Pipeline (ingest.py)**
2. **Training Pipeline (train.py)**
3. **Inference Service (serve.py)**

---

## Data Sources
- **Pump Sensors** (~2 Hz)
- **Nozzle Sensors** (~1 Hz)
- **Hydrant Sensors** (~0.2 Hz)
- **Risk Labels** (manual/auto)

Data is stored in `data_raw/`:
- `incident_000/pump.jsonl` → `{"ts_ms": 1000, "pump_id": "pump_1", "rpm": 1850}`
- `incident_000/nozzle.jsonl` → `{"ts_ms": 1000, "nozzle_id": "n1", "flow_gpm": 250}`
- `incident_000/hydrant.jsonl` → `{"ts_ms": 1000, "hydrant_id": "h1", "psi": 40.5}`
- `labels.csv` → `incident_id, ts_ms, risk_in_next_120s`

**Total:** 10 incidents × 3 streams = 30 JSONL files + 1 CSV  
**Size:** ~8,280 pump events, ~4,163 nozzle events, ~834 hydrant events

---

## Phase A: Ingestion Pipeline (`ingest.py`)

### Steps
1. **Load JSONL Files**
   - Read pump/nozzle/hydrant streams
   - Parse JSON lines, handle malformed/out-of-order events

2. **Canonicalize Field Names**
   - Example mappings:
     - `engine_rpm` → `pump.engine_rpm`
     - `intake_pressure` → `pump.intake_pressure_psi`
     - `nozzle_psi` → `nozzle.pressure_psi`
     - `residual_psi` → `hydrant.residual_pressure_psi`

3. **Transform to Long Schema**
   | incident_id | ts_ms | source | metric                  | value  |
   |-------------|-------|--------|-------------------------|--------|
   | incident_000| 1000  | pump_1 | pump.engine_rpm         | 1850.0 |
   | incident_000| 1000  | nozzle | nozzle.flow_gpm         | 250.0  |

   Benefits:
   - Efficient storage
   - Easy to add new metrics
   - Natural time-series representation

4. **Deduplicate**
   - Remove duplicates using `event_key = incident + source_id + ts_ms + metric`

5. **Separate Numeric vs Categorical**
   - Numeric: `pump.rpm`, `nozzle.flow_gpm`, etc.
   - Categorical: `valve_state`, `pattern`, `priming_active`

6. **Write Parquet Files**
   - `telemetry_long.parquet` (54,636 rows)
   - `telemetry_categorical.parquet` (20,723 rows)
   - `labels.parquet` (4,101 rows)

---

## Phase B: Training Pipeline (`train.py`)

### Steps
1. **Load Parquet Files**
   - `telemetry_long.parquet`
   - `telemetry_categorical.parquet`
   - `labels.parquet`

2. **Pivot Long → Wide Format**
   - Group by `(incident_id, ts_ms)`
   - Pivot metrics into columns

3. **Generate Features (60s Window)**
   - For each label at timestamp T:
     - Extract min, max, mean, last, count
   - Label window: `[T, T+120s]`
   - Result: 4,099 samples × 396 features

4. **Train/Test Split (Incident-Based)**
   - Train: incidents [0–6] → 2,922 samples
   - Test: incidents [7–9] → 1,177 samples
   - Prevents temporal leakage

5. **Train Two Models**
   - **Baseline Model (XGBoost, threshold=0.5)**
     - Balanced precision/recall
   - **Sentinel Model (XGBoost, threshold=0.0002)**
     - Recall-optimized (≥0.75 recall)
     - Accepts more false positives

   **Threshold Tradeoff Example:**
   | Threshold | Precision | Recall | FP  | FN |
   |-----------|-----------|--------|-----|----|
   | 0.01      | 0.398     | 1.000  | 445 | 0  |
   | 0.05      | 0.374     | 0.537  | 264 | 136|
   | 0.50      | N/A       | 0.000  | 0   | 294|

6. **Save Artifacts**
   - `baseline.xgb`
   - `sentinel.xgb`
   - `metadata.json`
   - `feature_names.json` (396 features)
   - `label_encoders.json`

---

## Phase C: Inference (`serve.py` / `predict.py`)

### Input
POST `/predict`

Added a representative file X_test_last.json in artifacts folder when training code runs.

## Instructions

Run the following commands to set up and start the service:

First, git clone the required repository
``` git
git clone https://github.com/ankittyagi87/Water-ops_Predictor
```
Then add data_raw/ data folder in the root directory

``` bash
uv sync
uv run python -m waterops.ingest --input ./data_raw --output ./data_curated
uv run python -m waterops.train --data ./data_curated --model-out ./artifacts
python -m waterops.serve
```

### Powershell
``` ps
curl.exe -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d @artifacts/X_test_last.json
```
### Linux
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d @artifacts/X_test_last.json
```
