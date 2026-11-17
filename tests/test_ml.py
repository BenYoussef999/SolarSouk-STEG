from backend.tunisovereign_core.ml_anomaly import TunisianTelemetryAnomalyEngine
from backend.tunisovereign_core.ml_detector import AnomalyDetector
from backend.tunisovereign_core.ml_synthetic import generate_dataset


def test_dataset_has_labeled_anomalies():
    df = generate_dataset()
    assert df["label"].sum() > 0 and set(df["event_type"]) >= {"pipe_burst", "heat_stress"}


def test_dataset_is_reproducible():
    assert generate_dataset(seed=1).equals(generate_dataset(seed=1))


def test_rules_catch_critical_conditions():
    df = generate_dataset(seed=3)
    scored = AnomalyDetector().fit(df).score(df)
    assert scored.loc[df["event_type"] == "heat_stress", "rule_flag"].mean() > 0.9


def test_metrics_are_reasonable():
    m = TunisianTelemetryAnomalyEngine().evaluate_all_methods()["Combined (IF OR Rules)"]
    assert m["precision"] > 0.5 and m["recall"] > 0.5 and m["f1_score"] > 0.5


def test_dashboard_compatible_flat_metrics():
    m = TunisianTelemetryAnomalyEngine().fit_and_evaluate_synthetic_benchmark()
    assert {"samples", "injected_anomalies", "detected_anomalies",
            "precision", "recall", "f1_score"} <= set(m)


def test_forecast_deficit_on_falling_moisture():
    df = generate_dataset(seed=1).iloc[:100].copy()
    df["moisture"] = [60 - 0.2 * i for i in range(100)]
    assert AnomalyDetector.forecast_deficit_24h(df) > 0
