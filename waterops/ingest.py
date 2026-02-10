import os
import json
import argparse
import pandas as pd
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field


# Pydantic Base Field Definition

class MetricSpec(BaseModel):
    canonical_name: str = Field(..., description="Canonical metric name with unit embedded")
    
class StreamSpec(BaseModel):
    source_type: Literal["pump", "nozzle", "hydrant"]
    source_id: str
    metrics: Dict[str, MetricSpec]
    
class TelemetryPydanticConfig(BaseModel):
    streams: Dict[str, StreamSpec]
    
# Define the metric configuration for each stream (numeric only)

PUMP_METRICS = {
    "engine_rpm": MetricSpec(canonical_name="pump.engine_rpm"),
    "throttle_pct": MetricSpec(canonical_name="pump.throttle_pct"),
    "intake_pressure_psi": MetricSpec(canonical_name="pump.intake_pressure_psi"),
    "discharge_pressure_psi": MetricSpec(canonical_name="pump.discharge_pressure_psi"),
    "tank_level_gal": MetricSpec(canonical_name="pump.tank_level_gal"),
}

NOZZLE_METRICS = {
    "nozzle_pressure_psi": MetricSpec(canonical_name="nozzle.pressure_psi"),
    "flow_gpm": MetricSpec(canonical_name="nozzle.flow_gpm"),
    "nozzle_setting_gpm": MetricSpec(canonical_name="nozzle.setting_gpm"),
}

HYDRANT_METRICS = {
    "static_pressure_psi": MetricSpec(canonical_name="hydrant.static_pressure_psi"),
    "residual_pressure_psi": MetricSpec(canonical_name="hydrant.residual_pressure_psi"),
}

TELEMETRY_Pydantic = TelemetryPydanticConfig(
    streams={
        "pump.jsonl": StreamSpec(
            source_type="pump",
            source_id="pump_id",
            metrics=PUMP_METRICS,
        ),
        "nozzle.jsonl": StreamSpec(
            source_type="nozzle",
            source_id="nozzle_id",
            metrics=NOZZLE_METRICS,
        ),
        "hydrant.jsonl": StreamSpec(
            source_type="hydrant",
            source_id="hydrant_id",
            metrics=HYDRANT_METRICS,
        ),
    }
)

# Categorical Fields

CATEGORICAL_FIELDS = {
    "pump": [
        "valve_state",
        "priming_active",
    ],
    "nozzle": [
        "pattern",
    ],
}

# Load Labels

def load_labels(base_path: str) -> pd.DataFrame:
    labels_path = os.path.join(base_path, "labels.csv")
    
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found at {labels_path}")
    
    labels = pd.read_csv(labels_path)
    labels.columns = [c.strip().lower() for c in labels.columns]
    
    required = {"incident_id", "ts_ms", "risk_in_next_120s"}
    if not required.issubset(labels.columns):
        raise ValueError(f"Labels file is missing required columns: {required}")
    
    labels["ts_ms"] = labels["ts_ms"].astype("int64")
    labels["risk_in_next_120s"] = labels["risk_in_next_120s"].astype("int8")

    return labels[["incident_id", "ts_ms", "risk_in_next_120s"]]

# Dataset Loader

class LoadJSONDataset:
    def __init__(self, base_path: str):
        self.base_path =  base_path
        self.incidents = [
            d for d in os.listdir(base_path)
            if d.startswith("incident_")
            and os.path.isdir(os.path.join(base_path, d))
        ]
        
        self.data: Dict[str, Dict[str, List[Dict]]] = {}
        
    def _load_jsonl(self, file_path: str) -> List[Dict]:
        records = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except FileNotFoundError:
            pass
        
        # Sort by event time to handle out-of-order events
        return sorted(records, key=lambda x: x.get('ts_ms', 0))
    
    def load_all(self):
        for incident in self.incidents:
            self.data[incident] = {}
            incident_path = os.path.join(self.base_path, incident)
            for file in TELEMETRY_Pydantic.streams:
                file_path = os.path.join(incident_path, file)
                self.data[incident][file] = self._load_jsonl(file_path)
                
    def to_long_schema(self, incident: str) -> pd.DataFrame:
        rows = []
        
        for file, stream in TELEMETRY_Pydantic.streams.items():
            records = self.data[incident].get(file, [])
            
            for rec in records:
                ts_ms = rec.get("ts_ms")
                source_id = rec.get(stream.source_id)
                if ts_ms is None or source_id is None:
                    continue
                
                for raw_field, spec in stream.metrics.items():
                    if raw_field in rec and rec[raw_field] is not None:
                        rows.append({
                            "incident_id": incident,
                            "ts_ms": int(ts_ms),
                            "source_type": stream.source_type,
                            "source_id": source_id,
                            "metric": spec.canonical_name,
                            "value": float(rec[raw_field]),
                            "event_key": f"{incident}_{source_id}_{ts_ms}_{spec.canonical_name}",
                        })
                        
        return pd.DataFrame(rows)
    
    # Categorical long schema
    
    def to_long_schema_categorical(self, incident: str) -> pd.DataFrame:
        rows = []
        
        for file, stream in TELEMETRY_Pydantic.streams.items():
            fields = CATEGORICAL_FIELDS.get(stream.source_type, [])
            if not fields:
                continue
            
            records = self.data[incident].get(file, [])
            
            for rec in records:
                ts_ms = rec.get("ts_ms")
                source_id = rec.get(stream.source_id)
                if ts_ms is None or source_id is None:
                    continue
                
                for cat_field in fields:
                    if cat_field in rec and rec[cat_field] is not None:
                        rows.append({
                            "incident_id": incident,
                            "ts_ms": int(ts_ms),
                            "source_type": stream.source_type,
                            "source_id": source_id,
                            "metric": f"{stream.source_type}.{cat_field}",
                            "value": str(rec[cat_field]),
                            "event_key": f"{incident}_{source_id}_{ts_ms}_{stream.source_type}_{cat_field}",
                        })
                        
        return pd.DataFrame(rows)
    
    # Write Output
    
    def write_outputs(self, out_path: str):
        os.makedirs(out_path, exist_ok=True)
        
        numeric_frames = []
        categorical_frames = []
        
        for incident in self.incidents:
            df_numeric = self.to_long_schema(incident)
            df_categorical = self.to_long_schema_categorical(incident)
            
            if not df_numeric.empty:
                numeric_frames.append(df_numeric)
                
            if not df_categorical.empty:
                categorical_frames.append(df_categorical)
                
        telemetry_long = (
            pd.concat(numeric_frames, ignore_index=True)
            if numeric_frames else
            pd.DataFrame(columns=[
                "incident_id", "ts_ms", "source_type",
                "source_id", "metric", "value", "event_key"
            ])
        )

        telemetry_cat = (
            pd.concat(categorical_frames, ignore_index=True)
            if categorical_frames else
            pd.DataFrame(columns=[
                "incident_id", "ts_ms", "source_type",
                "source_id", "field", "value", "event_key"
            ])
        )

        labels = load_labels(self.base_path)
        print(telemetry_long)
        telemetry_long.to_parquet(os.path.join(out_path, "telemetry_long.parquet"), index=False)
        telemetry_cat.to_parquet(os.path.join(out_path, "telemetry_categorical.parquet"), index=False)
        labels.to_parquet(os.path.join(out_path, "labels.parquet"), index=False)

# Main Function

def main():
    parser = argparse.ArgumentParser(description="Ingest raw JSONL telemetry into Parquet")
    parser.add_argument("--input", required=True, help="Path to input directory containing incident folders")
    parser.add_argument("--output", required=True, help="Path to curated output directory")
    args = parser.parse_args()

    dataset = LoadJSONDataset(base_path=args.input)
    dataset.load_all()
    dataset.write_outputs(out_path=args.output)
    
    print("Loaded incidents:", dataset.incidents)
    
    print("Data ingestion done!")
    
if __name__ == "__main__":
    main()        

    
        