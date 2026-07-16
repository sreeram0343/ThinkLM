import pytest
from unittest.mock import MagicMock
from src.agents.master import MasterAgent
from src.utils.dispatcher import REPLDispatcher
from src.memory.memory import DualProcessMemory
from src.utils.config import ConfigManager

def test_semantic_routing_detailed_tiers():
    agent = MasterAgent()
    
    # 1. Semantic Writer-Only lookup
    res = agent.analyze_complexity("Could you tell me the birth name of Emperor Han-Wu?")
    assert res == "WRITER_ONLY"
    
    # 2. Semantic Executor-Inclusive query
    res = agent.analyze_complexity("Retrieve the current price of bitcoin online")
    assert res == "EXECUTOR_INCLUSIVE"
    
    # 3. Semantic Planner-Enhanced query
    res = agent.analyze_complexity("Explain how to compare career statistics of MS Dhoni and Virat Kohli")
    assert res == "PLANNER_ENHANCED"

def test_dispatcher_bypasses_llm_entirely():
    memory = DualProcessMemory(episodic_limit=5)
    config_manager = ConfigManager()
    
    # Mock MasterAgent to track calls
    master_agent = MasterAgent()
    master_agent.run_collaborative_loop = MagicMock(return_value={"status": "SUCCESS", "final_answer": "mocked"})
    master_agent.analyze_complexity = MagicMock(return_value="WRITER_ONLY")
    
    dispatcher = REPLDispatcher(memory, master_agent, config_manager)
    
    # Send a slash command
    response = dispatcher.dispatch("/mode")
    assert "Plan Mode" in response
    
    # Verify that MasterAgent methods were NOT called for slash command
    master_agent.run_collaborative_loop.assert_not_called()
    master_agent.analyze_complexity.assert_not_called()
    
    # Send a regular query
    response_query = dispatcher.dispatch("What is the birth name of Han-Wu?")
    assert response_query == "mocked"
    
    # Verify MasterAgent WAS called for regular query
    master_agent.run_collaborative_loop.assert_called_once()
