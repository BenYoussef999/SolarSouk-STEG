from datetime import datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from backend.tunisovereign_core.ml_detector import AnomalyDetector
from backend.tunisovereign_core.ml_synthetic import generate_dataset
from agent.skills.base import BaseAgentSkill


class Reading(BaseModel):
    timestamp: datetime
    moisture: float = Field(ge=0, le=100)
    flow: float = Field(ge=0)
    temp: float


class AnomalyInput(BaseModel):
    node_code: str
    readings: list[Reading] = Field(min_length=13)  # >= 2h of 10-min samples


class AnomalyOutput(BaseModel):
    node_code: str
    is_anomaly: bool
    severity: Literal["NONE", "LOW", "MEDIUM", "CRITICAL"]
    anomaly_type: str
    isolation_forest_score: float
    z_score: float
    forecasted_deficit_24h: float


class SoilTelemetryAnomalySkill(BaseAgentSkill):
    name = "soil_telemetry_anomaly"
    description = "Detects leaks/theft/heat stress on a plot and forecasts the 24h moisture deficit."
    input_model, output_model = AnomalyInput, AnomalyOutput

    def __init__(self, detector: AnomalyDetector | None = None):
        self.detector = detector or AnomalyDetector().fit(generate_dataset(seed=42))

    def execute(self, data: AnomalyInput) -> AnomalyOutput:
        df = pd.DataFrame([r.model_dump() for r in data.readings]).sort_values("timestamp")
        df = df.reset_index(drop=True)
        scored = self.detector.score(df)
        last = scored.iloc[-1]

        if last["rule_flag"]:
            severity = "CRITICAL"
            heat = last["moisture"] < self.detector.crit_moisture and last["temp"] > self.detector.crit_temp
            kind = "HEAT_STRESS_LOW_MOISTURE" if heat else "RAPID_MOISTURE_DROP_LEAK_OR_THEFT"
        elif last["if_flag"]:
            severity, kind = "MEDIUM", "STATISTICAL_OUTLIER"
        elif last["z_flag"]:
            severity, kind = "LOW", "MOISTURE_DEVIATION"
        else:
            severity, kind = "NONE", "NORMAL"

        return AnomalyOutput(
            node_code=data.node_code, is_anomaly=severity != "NONE", severity=severity,
            anomaly_type=kind, isolation_forest_score=float(last["if_score"]),
            z_score=float(last["z_score"]),
            forecasted_deficit_24h=self.detector.forecast_deficit_24h(df))
