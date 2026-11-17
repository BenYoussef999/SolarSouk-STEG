"""IsolationForest + rolling z-score + Tunisian threshold rules + naive 24h deficit forecast."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["moisture", "flow", "temp", "drop_2h"]


class AnomalyDetector:
    def __init__(self, contamination: float = 0.07, random_state: int = 42,
                 z_window: int = 144, z_threshold: float = 3.0, steps_per_2h: int = 12,
                 drop_limit: float = 12.0, crit_moisture: float = 33.0, crit_temp: float = 35.0):
        self.z_window, self.z_threshold = z_window, z_threshold
        self.steps_per_2h = steps_per_2h
        self.drop_limit, self.crit_moisture, self.crit_temp = drop_limit, crit_moisture, crit_temp
        self.model = Pipeline([
            ("scale", StandardScaler()),
            ("iforest", IsolationForest(n_estimators=200, contamination=contamination,
                                        random_state=random_state)),
        ])
        self.fitted = False

    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["drop_2h"] = (out["moisture"].shift(self.steps_per_2h) - out["moisture"]).fillna(0.0)
        return out

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        self.model.fit(self._features(df)[FEATURES])
        self.fitted = True
        return self

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("AnomalyDetector must be fitted before scoring")
        out = self._features(df)
        X = out[FEATURES]
        out["if_score"] = -self.model.decision_function(X)  # higher = more anomalous
        out["if_flag"] = self.model.predict(X) == -1

        roll = out["moisture"].rolling(self.z_window, min_periods=36)
        std = roll.std().replace(0, np.nan)
        out["z_score"] = ((out["moisture"] - roll.mean()) / std).fillna(0.0)
        out["z_flag"] = out["z_score"].abs() > self.z_threshold

        out["rule_flag"] = (out["drop_2h"] > self.drop_limit) | (
            (out["moisture"] < self.crit_moisture) & (out["temp"] > self.crit_temp))
        out["is_anomaly"] = out["if_flag"] | out["rule_flag"]
        return out

    @staticmethod
    def forecast_deficit_24h(df: pd.DataFrame, lookback: int = 36, step_min: int = 10,
                             target: float = 45.0) -> float:
        """Linear-trend baseline: project moisture 24h ahead; deficit = points below the 45% floor."""
        y = df["moisture"].to_numpy()[-lookback:]
        if len(y) < 3:
            return 0.0
        slope = np.polyfit(np.arange(len(y)), y, 1)[0]
        steps_24h = 24 * 60 // step_min
        projected = float(np.clip(y[-1] + slope * steps_24h, 0, 100))
        return round(max(0.0, target - projected), 2)
