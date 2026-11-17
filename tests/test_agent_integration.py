import pytest

from agent.langgraph_agent import run_sovereign_agent_loop


@pytest.fixture(autouse=True)
def no_real_groq(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)


def test_agent_runs_end_to_end_with_new_skill_and_mcp_tool():
    state = {
        "thread_id": "t-1", "node_code": "KAS-SBEITLA-PLOT-04",
        "darja_input": "Ya khouya fi Kasserine el sonda habtet 28% wel skhana 37 darja, lazmenna maa urgent!",
        "moisture_pct": 28.4, "temp_c": 37.2, "flow_l_min": 182.0,
        "parsed_intent": {}, "anomaly_report": {}, "mcp_allocation": {}, "zk_receipt": {},
        "skills_trace": [], "final_explanation": "",
    }
    out = run_sovereign_agent_loop(state)
    assert out["parsed_intent"]["governorate"] == "Kasserine"
    assert 0 < out["mcp_allocation"]["approved_units"] <= out["mcp_allocation"]["max_daily_quota"]
    assert "darja_dialect_parser" in out["skills_trace"]
    assert out["zk_receipt"]["zk_commitment_hash"].startswith("0x")
