import asyncio

import pytest

from mcp_server import tunisovereign_mcp as srv


def test_budget_is_limited_by_et_need():
    r = srv.calculate_evapotranspiration_budget("Kasserine", 28.4, 37.2, 40.0, 3200.0)
    assert r["approved_units"] == pytest.approx(2357.3, abs=0.01)
    assert r["capped_by"] == "ET_BUDGET" and r["below_critical_threshold"] is True


def test_budget_never_exceeds_governorate_quota():
    r = srv.calculate_evapotranspiration_budget("Kasserine", 10.0, 45.0, 10.0, 99999.0)
    assert r["approved_units"] == r["max_daily_quota"] == 4200.0
    assert r["capped_by"] == "QUOTA" and r["within_quota"] is True


def test_budget_limited_by_request():
    r = srv.calculate_evapotranspiration_budget("Gafsa", 28.0, 37.0, 40.0, 500.0)
    assert r["approved_units"] == 500.0 and r["capped_by"] == "REQUEST"


def test_no_water_needed_when_soil_is_wet_and_cool():
    r = srv.calculate_evapotranspiration_budget("Tozeur", 60.0, 25.0, 50.0, 1800.0)
    assert r["approved_units"] == 0.0


@pytest.mark.parametrize("kwargs", [
    {"current_moisture_pct": 120.0}, {"current_moisture_pct": -1.0},
    {"current_moisture_pct": 30.0, "humidity_pct": 150.0},
    {"current_moisture_pct": 30.0, "requested_units": -5.0},
    {"current_moisture_pct": float("nan")},
])
def test_budget_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        srv.calculate_evapotranspiration_budget("Kasserine", **kwargs)


def test_unknown_governorate_uses_default_policy_and_says_so():
    r = srv.calculate_evapotranspiration_budget("Atlantis", 30.0)
    assert r["policy_source"] == "default" and r["max_daily_quota"] == 4000.0


def test_governorate_name_is_normalized():
    assert srv.get_aquifer_quota("sidi bouzid")["max_daily_quota"] == 4500.0


def test_resources():
    q = srv.get_aquifer_quota("Kasserine")
    assert q["max_daily_quota"] == 4200.0 and q["critical_threshold"] == 35.0
    assert "06:00-10:00" in srv.get_irrigation_schedule("Kasserine")["pump_windows_local_time"]


def test_quota_compliance_tool():
    assert srv.check_quota_compliance("Kasserine", 1000, 500)["compliant"] is True
    r = srv.check_quota_compliance("Kasserine", 4000, 500)
    assert r["compliant"] is False and r["remaining_units"] == 3700.0


def test_server_really_speaks_mcp():
    """Needs the real SDK (mcp>=1.9,<2). Fails loudly if the silent shim is in use."""
    pytest.importorskip("mcp.server.fastmcp")
    assert hasattr(srv.mcp, "list_tools"), "FastMCP shim in use - install 'mcp<2'"
    tools = {t.name for t in asyncio.run(srv.mcp.list_tools())}
    assert tools == {"calculate_evapotranspiration_budget", "check_quota_compliance"}
    templates = {t.uriTemplate for t in asyncio.run(srv.mcp.list_resource_templates())}
    assert templates == {"water://governorate/{governorate}/aquifer-quota",
                         "water://governorate/{governorate}/irrigation-schedule"}
    result = asyncio.run(srv.mcp.call_tool(
        "check_quota_compliance", {"governorate": "Gafsa", "planned_units": 100}))
    content = result[0] if isinstance(result, tuple) else result  # shape differs across SDK versions
    assert "compliant" in content[0].text
