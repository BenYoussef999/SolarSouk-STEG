import pytest
from pydantic import ValidationError

from backend.tunisovereign_core.ml_synthetic import generate_dataset
from agent.skills.soil_anomaly import SoilTelemetryAnomalySkill
from agent.skills.zk_water_compliance import (ZKWaterComplianceProverSkill, compute_commitment,
                                        verify_commitment)


def _window(df, end, n=30):
    w = df.iloc[end - n:end]
    return [{"timestamp": r.timestamp.isoformat(), "moisture": r.moisture, "flow": r.flow,
             "temp": r.temp} for r in w.itertuples()]


@pytest.fixture(scope="module")
def skill():
    return SoilTelemetryAnomalySkill()


def test_anomaly_skill_flags_heat_stress(skill):
    df = generate_dataset(seed=7)
    idx = df.index[df["event_type"] == "heat_stress"][10]
    out = skill.run({"node_code": "KSR-001", "readings": _window(df, idx + 1)})
    assert out["is_anomaly"] and out["severity"] == "CRITICAL"


def test_anomaly_skill_normal_window(skill):
    df = generate_dataset(seed=7)
    out = skill.run({"node_code": "KSR-001", "readings": _window(df, 150)})
    assert out["severity"] == "NONE" and out["is_anomaly"] is False


def test_anomaly_skill_rejects_short_input(skill):
    with pytest.raises(ValidationError):
        skill.run({"node_code": "X", "readings": []})


def test_zk_compliant_draw_produces_commitment():
    out = ZKWaterComplianceProverSkill().run(
        {"node_id": "KSR-001", "water_draw": 3000, "quota_limit": 4200})
    assert out["compliant"] and len(out["commitment_hash"]) == 66
    assert verify_commitment("KSR-001", 3000, 4200, out["salt"], out["commitment_hash"])


def test_zk_breach_is_never_committed():
    out = ZKWaterComplianceProverSkill().run(
        {"node_id": "KSR-001", "water_draw": 5000, "quota_limit": 4200})
    assert out["compliant"] is False and out["commitment_hash"] is None


def test_zk_commitment_hides_and_binds():
    a = compute_commitment("N1", 100, 200, "salt1")
    assert a != compute_commitment("N1", 100, 200, "salt2")
    assert a != compute_commitment("N1", 101, 200, "salt1")


def test_zk_same_salt_is_deterministic():
    s = ZKWaterComplianceProverSkill()
    p = {"node_id": "N1", "water_draw": 10, "quota_limit": 20, "salt": "abc"}
    assert s.run(p)["commitment_hash"] == s.run(p)["commitment_hash"]
