"""Synthetic Kasserine/Sidi Bouzid plot telemetry with labeled anomalies (10-min sampling)."""
import numpy as np
import pandas as pd


def generate_dataset(days: int = 14, seed: int = 42, n_events: int = 6) -> pd.DataFrame:
    """Columns: timestamp, moisture (%), flow (L/min), temp (C), label (0/1), event_type."""
    rng = np.random.default_rng(seed)
    n = days * 144
    ts = pd.date_range("2026-06-01", periods=n, freq="10min")
    hour = ts.hour.values + ts.minute.values / 60

    moisture = 58 + 6 * np.sin(2 * np.pi * (hour - 8) / 24) + rng.normal(0, 0.8, n)
    temp = 27 + 8 * np.sin(2 * np.pi * (hour - 9) / 24) + rng.normal(0, 0.8, n)
    pump_on = ((hour >= 6) & (hour < 10)) | ((hour >= 17) & (hour < 20))
    flow = 15 + 25 * pump_on + rng.normal(0, 2, n)

    label = np.zeros(n, dtype=int)
    event_type = np.array([""] * n, dtype=object)
    starts = np.linspace(200, n - 80, n_events).astype(int) + rng.integers(-20, 20, n_events)

    for i, s in enumerate(starts):
        sl = slice(s, s + 24)
        if i % 2 == 0:  # pipe burst / theft: >12 point moisture drop within 2h, flow surge
            ramp = np.concatenate([np.linspace(0, 16, 12), np.full(12, 16.0)])
            moisture[sl] -= ramp
            flow[sl] += 50
            kind = "pipe_burst"
        else:  # Sirocco heat stress: low moisture under >35C
            moisture[sl] = 30 + rng.normal(0, 1, 24)
            temp[sl] = 38 + rng.normal(0, 0.8, 24)
            kind = "heat_stress"
        label[sl] = 1
        event_type[sl] = kind

    return pd.DataFrame({"timestamp": ts, "moisture": moisture, "flow": flow,
                         "temp": temp, "label": label, "event_type": event_type})
