# Firefighting Risk Prediction System

## Overview
This system ingests telemetry from pumps, nozzles, and hydrants, combines it with risk labels, and trains XGBoost models to predict firefighting risks. 

## Problem Statement
Build a system that predicts a Water Supply Risk score: the probability
that water delivery will become unstable within the next 2 minutes, based on the last N seconds of
telemetry.

Water instability manifests as:

- Loss of suction or cavitation at the pump
- Collapse of hydrant pressure under load
- Sudden loss of nozzle performance despite the pump running normally
- Rapid depletion of onboard tank while high flow demand continues

### 1. Cavitation Risk
   
#### Relevant Telemetry
   
##### Pump
- intake_pressure_psi
- engine_rpm
- throttle_pct
- valve_state

Intake pressure reflects the availability of water at the pump suction.

When intake pressure drops too low:

- Water begins to vaporize inside the pump
- Vapor bubbles collapse violently → cavitation
- Pump efficiency collapses and hardware damage occurs

### 2. Hydrant Residual Pressure Crash
   
#### Relevant Telemetry

##### Supply (Hydrant)
- static_pressure_psi
- residual_pressure_psi

Static pressure = pressure with no flow

Residual pressure = pressure while flowing

A large drop from static → residual means the water distribution system cannot support the demanded flow.

### 3. Sudden Nozzle Pressure / Flow Drop While Pump Discharge Is Steady

#### Relevant Telemetry

##### Pump
- discharge_pressure_psi

##### Nozzle
- nozzle_pressure_psi
- flow_gpm
- nozzle_setting_gpm
- pattern

If pump discharge pressure remains stable but nozzle pressure or flow drops, the problem is downstream of the pump leading to

- Hose kink or collapse
- Valve partially closing
- Nozzle pattern change increasing friction loss

#### Risk Pattern

- Discharge pressure steady
- Nozzle pressure ↓ suddenly
- Flow ↓ while demand setting unchanged

### 4. Tank Emptying Soon While Flow Demand Remains High

#### Relevant Telemetry

##### Pump
- tank_level_gal
- discharge_pressure_psi

##### Nozzle
- flow_gpm
  
##### Supply
- Intake pressure / residual pressure

Rapid tank depletion indicates demand is exceeding sustainable supply.

This typically occurs when:

- Hydrant supply is inadequate or delayed
- Multiple high-flow lines are opened
- Pump operator relies too long on tank water

The code is organized into three phases:

1. **Ingestion Pipeline (ingest.py)**
2. **Training Pipeline (train.py)**
3. **Inference Service (serve.py)**

---
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

## Phase C: Inference (`serve.py`)

### Input
POST `/predict`

Added a representative file X_test_last.json in artifacts folder when training code runs.

---

## Future Steps
We can add more physics related features to help identify the model about risk.
### Feature Engineering
| Feature name          | Formulae          | Physical Meaning  |
|-----------------------|-------------------|-------------------|
| hydrant_pressure_drop | static − residual |Network head loss  |
| intake_pressure_volatility | std(intake_pump_pressure) |Unstable suction / cavitation|
| pressure_loss_between_pump_nozzle | discharge_pressure_mean − nozzle_pressure_mean |Identifies hose kinks, valve restriction, or downstream collapse|
|cavitation_risk_index | (engine_rpm_mean × throttle_pct_mean) / max(intake_pressure_mean, small value) | Detects pump working harder while suction collapses|

This is only a representative set.




