import pytest
from src.agents.master import MasterAgent

def test_agent_routing_complexity_tiers():
    agent = MasterAgent()
    
    # 1. Factual Lookup: routes to WRITER_ONLY
    query_factual = "Who was Emperor Han-Wu?"
    assert agent.analyze_complexity(query_factual) == "WRITER_ONLY"
    
    # 2. Tool Lookup: routes to EXECUTOR_INCLUSIVE
    query_tool = "What is the weather in Beijing today?"
    assert agent.analyze_complexity(query_tool) == "EXECUTOR_INCLUSIVE"
    
    # 3. Context-complex query: routes to PLANNER_ENHANCED
    query_complex = "Calculate the age difference between Emperor Han-Wu and Julius Caesar"
    assert agent.analyze_complexity(query_complex) == "PLANNER_ENHANCED"

def test_master_agent_run_collaborative_loop():
    agent = MasterAgent()
    
    # Verify Factual Lookup trajectory context
    res_factual = agent.run_collaborative_loop("Who was Emperor Han-Wu?")
    assert res_factual["complexity_tier"] == "WRITER_ONLY"
    assert res_factual["status"] == "SUCCESS"
    assert "final_answer" in res_factual
    
    # Verify Tool Lookup trajectory context
    res_tool = agent.run_collaborative_loop("What is the weather in Beijing today?")
    assert res_tool["complexity_tier"] == "EXECUTOR_INCLUSIVE"
    assert res_tool["status"] == "SUCCESS"
    assert "final_answer" in res_tool
    
    # Verify Context-complex query trajectory context
    res_complex = agent.run_collaborative_loop("Calculate the age difference between Emperor Han-Wu and Julius Caesar")
    assert res_complex["complexity_tier"] == "PLANNER_ENHANCED"
    assert res_complex["status"] == "SUCCESS"
    assert "final_answer" in res_complex
