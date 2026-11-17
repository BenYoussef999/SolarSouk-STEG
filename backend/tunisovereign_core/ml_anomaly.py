"""
Dual ML anomaly engine (IsolationForest + rolling z-score + Tunisian threshold rules)
with a reproducible Precision / Recall / F1 harness.

Run:  python backend/tunisovereign_core/ml_anomaly.py
"""
import os
import sys

from sklearn.metrics import precision_recall_fscore_support

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.tunisovereign_core.ml_detector import AnomalyDetector
from backend.tunisovereign_core.ml_synthetic import generate_dataset

METHODS = [("IsolationForest", "if_flag"), ("Rolling Z-Score", "z_flag"),
           ("Threshold Rules", "rule_flag"), ("Combined (IF OR Rules)", "is_anomaly")]


class TunisianTelemetryAnomalyEngine:
    """Train on one synthetic series (seed 42), evaluate on a different one (seed 7)."""

    def __init__(self, contamination: float = 0.07, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state

    def _run(self):
        train = generate_dataset(seed=42)
        test = generate_dataset(seed=7)
        det = AnomalyDetector(contamination=self.contamination, random_state=self.random_state)
        return test, det.fit(train).score(test)

    def evaluate_all_methods(self) -> dict:
        test, scored = self._run()
        out = {}
        for name, col in METHODS:
            p, r, f, _ = precision_recall_fscore_support(
                test["label"], scored[col], average="binary", zero_division=0)
            out[name] = {"precision": round(float(p), 4), "recall": round(float(r), 4),
                         "f1_score": round(float(f), 4)}
        return out

    def fit_and_evaluate_synthetic_benchmark(self) -> dict:
        """Flat dict (kept compatible with the Streamlit dashboard): the combined detector."""
        test, scored = self._run()
        p, r, f, _ = precision_recall_fscore_support(
            test["label"], scored["is_anomaly"], average="binary", zero_division=0)
        return {"samples": int(len(test)), "injected_anomalies": int(test["label"].sum()),
                "detected_anomalies": int(scored["is_anomaly"].sum()),
                "precision": round(float(p), 4), "recall": round(float(r), 4),
                "f1_score": round(float(f), 4)}


if __name__ == "__main__":
    engine = TunisianTelemetryAnomalyEngine()
    print(f"{'Method':<24}{'Precision':>10}{'Recall':>9}{'F1':>8}")
    for name, m in engine.evaluate_all_methods().items():
        print(f"{name:<24}{m['precision']:>10}{m['recall']:>9}{m['f1_score']:>8}")
    print("Combined metrics:", engine.fit_and_evaluate_synthetic_benchmark())
