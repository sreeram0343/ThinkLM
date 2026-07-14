import pytest
from src.agents.master import MasterAgent
from src.agents.planner import PlannerAgent
from src.agents.executor import ExecutorAgent
from src.agents.writer import WriterAgent

def test_master_agent_init():
    agent = MasterAgent()
    assert agent.model_name == "Qwen/Qwen2.5-7B-Instruct"
    assert agent.temperature == 0.1
    
    complexity = agent.analyze_complexity("Hello world")
    assert complexity in ["WRITER_ONLY", "EXECUTOR_INCLUSIVE", "PLANNER_ENHANCED"]

def test_planner_agent_init():
    agent = PlannerAgent()
    assert agent.mcp_client is None
    
    # Test restrict_boundary with mock tools list
    tools = [
        {"name": "tool_1", "description": "fetches weather documentation"},
        {"name": "tool_2", "description": "calculates dynamic math structures"}
    ]
    restricted = agent.restrict_boundary("weather data check", tools)
    assert len(restricted) <= 2

def test_executor_agent_init():
    agent = ExecutorAgent()
    assert agent.sandbox_env is None
    assert "web_search" in agent.tool_clusters
    assert "calculator" in agent.tool_clusters

def test_writer_agent_init():
    agent = WriterAgent()
    assert agent is not None
    
    # Test base synthesis response
    results = {
        "T1": {
            "description": "Mock query",
            "output": {"result": "success"}
        }
    }
    response = agent.synthesize_response("Mock original query", results)
    assert "ThinkLM-Lite" in response
    assert "Mock original query" in response
