"""Django glue: run the anomaly skill on a node's recent readings and store an
AnomalyAndForecastEvent. Call from a management command / periodic task / the gRPC server
after a BATCH of readings (not on every single save)."""
from datetime import timedelta

import pandas as pd
from django.utils import timezone

APP_LABEL = "tunisovereign_core"  # change if your Django app label differs


def _models():
    from django.apps import apps
    return (apps.get_model(APP_LABEL, "TelemetryReading"),
            apps.get_model(APP_LABEL, "AnomalyAndForecastEvent"))


def recent_dataframe(node, hours: int = 4) -> pd.DataFrame:
    """Pivot MOISTURE/FLOW/TEMP rows into one 10-minute-resolution frame."""
    telemetry, _ = _models()
    since = timezone.now() - timedelta(hours=hours)
    rows = telemetry.objects.filter(node=node, timestamp__gte=since).values(
        "timestamp", "metric_type", "value")
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    wide = df.pivot_table(index="timestamp", columns="metric_type", values="value", aggfunc="mean")
    wide = wide.rename(columns={"MOISTURE": "moisture", "FLOW": "flow", "TEMP": "temp"})
    if not {"moisture", "flow", "temp"} <= set(wide.columns):
        return pd.DataFrame()
    wide = wide.resample("10min").mean().ffill().dropna()
    return wide.reset_index()


def evaluate_node(node, skill=None):
    """Returns the created AnomalyAndForecastEvent, or None if normal / not enough data."""
    from agent.skills.soil_anomaly import SoilTelemetryAnomalySkill
    _, event_model = _models()
    skill = skill or SoilTelemetryAnomalySkill()
    df = recent_dataframe(node)
    if len(df) < 13:
        return None
    result = skill.run({"node_code": node.node_code,
                        "readings": df[["timestamp", "moisture", "flow", "temp"]].to_dict("records")})
    if not result["is_anomaly"]:
        return None
    return event_model.objects.create(
        node=node, anomaly_type=result["anomaly_type"], severity=result["severity"],
        isolation_forest_score=result["isolation_forest_score"],
        forecasted_deficit_24h=result["forecasted_deficit_24h"])
