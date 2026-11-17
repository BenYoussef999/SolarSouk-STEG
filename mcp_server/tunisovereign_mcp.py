"""
TuniWaterMCP - custom FastMCP server for Track 1 (TunicChain-Sovereign).

Resources:  water://governorate/{governorate}/aquifer-quota
            water://governorate/{governorate}/irrigation-schedule
Tools:      calculate_evapotranspiration_budget(...)
            check_quota_compliance(...)
Prompt:     sovereign_auditor_system_prompt(...)

Run (stdio):   python mcp_server/tunisovereign_mcp.py
Inspect:       npx @modelcontextprotocol/inspector python mcp_server/tunisovereign_mcp.py

Requires the official SDK v1 line:  mcp>=1.9,<2   (v2 renamed FastMCP to MCPServer).
"""
import sys

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - loud fallback so nobody believes a fake server is real
    print("WARNING: 'mcp.server.fastmcp' not importable (install 'mcp>=1.9,<2'). "
          "Using a NON-FUNCTIONAL shim: tools can be called in-process but nothing is served.",
          file=sys.stderr)

    class FastMCP:
        def __init__(self, name: str):
            self.name = name

        def _passthrough(self, *_args, **_kwargs):
            return lambda fn: fn

        resource = tool = prompt = _passthrough

        def run(self):
            raise RuntimeError("Install 'mcp>=1.9,<2' to actually serve MCP.")

mcp = FastMCP("TuniWaterMCP")

GOVERNORATE_POLICIES = {
    "Kasserine": {"max_daily_quota": 4200.0, "critical_threshold": 35.0, "priority_tier": "Priority-A (Interior Aquifer)"},
    "Sidi_Bouzid": {"max_daily_quota": 4500.0, "critical_threshold": 33.0, "priority_tier": "Priority-A (Agricultural Basin)"},
    "Gafsa": {"max_daily_quota": 3800.0, "critical_threshold": 36.0, "priority_tier": "Priority-A (Mining & Oasis Basin)"},
    "Kairouan": {"max_daily_quota": 4400.0, "critical_threshold": 34.0, "priority_tier": "Priority-A (Central Plain)"},
    "Tozeur": {"max_daily_quota": 4000.0, "critical_threshold": 35.0, "priority_tier": "Priority-A (Southern Oasis)"},
    "Tunis": {"max_daily_quota": 5000.0, "critical_threshold": 30.0, "priority_tier": "Standard-Urban"},
}
DEFAULT_POLICY = {"max_daily_quota": 4000.0, "critical_threshold": 35.0, "priority_tier": "Standard"}
PUMP_WINDOWS = ["06:00-10:00", "17:00-20:00"]  # same windows the simulator uses


def _normalize(governorate: str) -> str:
    return governorate.strip().title().replace(" ", "_")


def _policy_for(governorate: str) -> tuple[str, dict, str]:
    gov = _normalize(governorate)
    if gov in GOVERNORATE_POLICIES:
        return gov, GOVERNORATE_POLICIES[gov], "official"
    return gov, DEFAULT_POLICY, "default"


def _check_range(name: str, value: float, lo: float, hi: float) -> None:
    if not (lo <= value <= hi):  # also rejects NaN
        raise ValueError(f"{name} must be between {lo} and {hi}, got {value}")


@mcp.resource("water://governorate/{governorate}/aquifer-quota")
def get_aquifer_quota(governorate: str) -> dict:
    """Official daily aquifer quota, critical moisture threshold and priority tier."""
    gov, policy, source = _policy_for(governorate)
    return {"governorate": gov, "policy_source": source, **policy}


@mcp.resource("water://governorate/{governorate}/irrigation-schedule")
def get_irrigation_schedule(governorate: str) -> dict:
    """Pumping windows in which allocated water may be drawn."""
    gov, _, source = _policy_for(governorate)
    return {"governorate": gov, "policy_source": source, "pump_windows_local_time": PUMP_WINDOWS}


@mcp.tool()
def calculate_evapotranspiration_budget(
    governorate: str,
    current_moisture_pct: float,
    ambient_temp_c: float = 31.0,
    humidity_pct: float = 40.0,
    requested_units: float = 1200.0,
) -> dict:
    """Deterministic water budget: approved = min(requested, ET need, governorate quota)."""
    _check_range("current_moisture_pct", current_moisture_pct, 0, 100)
    _check_range("ambient_temp_c", ambient_temp_c, -10, 60)
    _check_range("humidity_pct", humidity_pct, 0, 100)
    _check_range("requested_units", requested_units, 0, 1_000_000)

    gov, policy, source = _policy_for(governorate)
    max_quota = policy["max_daily_quota"]

    deficit = max(0.0, (55.0 - current_moisture_pct) * 65.0) + max(0.0, ambient_temp_c - 28.0) * 45.0
    humidity_factor = 1.0 + (60.0 - min(60.0, humidity_pct)) / 200.0
    et_need = round(deficit * humidity_factor, 2)

    approved = round(min(requested_units, et_need, max_quota), 2)
    if approved == max_quota:
        capped_by = "QUOTA"
    elif approved == et_need:
        capped_by = "ET_BUDGET"
    else:
        capped_by = "REQUEST"

    return {
        "governorate": gov,
        "policy_source": source,
        "priority_tier": policy["priority_tier"],
        "below_critical_threshold": current_moisture_pct < policy["critical_threshold"],
        "et_budget_units": et_need,
        "requested_units": requested_units,
        "approved_units": approved,
        "max_daily_quota": max_quota,
        "within_quota": approved <= max_quota,
        "capped_by": capped_by,
        "utilization_ratio": round(approved / max(1.0, max_quota), 4),
    }


@mcp.tool()
def check_quota_compliance(governorate: str, planned_units: float, already_used_today: float = 0.0) -> dict:
    """Checks whether a planned draw fits in what is left of today's governorate quota."""
    _check_range("planned_units", planned_units, 0, 1_000_000)
    _check_range("already_used_today", already_used_today, 0, 1_000_000)
    gov, policy, source = _policy_for(governorate)
    remaining = max(0.0, policy["max_daily_quota"] - already_used_today)
    return {
        "governorate": gov,
        "policy_source": source,
        "compliant": planned_units <= remaining,
        "planned_units": planned_units,
        "remaining_units": round(remaining, 2),
        "max_daily_quota": policy["max_daily_quota"],
    }


@mcp.prompt()
def sovereign_auditor_system_prompt(track_name: str, governorate: str) -> str:
    """System prompt template for the TuniSovereign LangGraph agent."""
    return (
        f"You are the autonomous TuniSovereign-AI auditor for {track_name} in {governorate}, Tunisia. "
        "Always verify telemetry anomalies via ML, check governorate quotas via MCP tools, and anchor "
        "privacy-preserving ZK commitments before authorizing physical actuation or fund release."
    )


if __name__ == "__main__":
    mcp.run()
