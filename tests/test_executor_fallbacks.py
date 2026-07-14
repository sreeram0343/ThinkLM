import pytest
from src.agents.executor import ExecutorAgent, ConnectionTimeout

def test_executor_fallback_default_switch():
    # Instantiate ExecutorAgent
    agent = ExecutorAgent()
    
    # The default web_search cluster is ["baidu_search_api", "google_search_api", "wikipedia_bm25_fallback"]
    # "baidu_search_api" is hardcoded to raise ConnectionTimeout in the mock implementation.
    # Therefore, it should successfully fallback to "google_search_api".
    params = {"query": "Emperor Han-Wu"}
    
    # Invoke execution
    res = agent.execute_tool_with_fallback("web_search", params)
    
    # Assert successful retrieval from the fallback tool (google_search_api)
    assert res == {"birth_year": -156, "era": "BC", "name": "Emperor Han-Wu"}

def test_executor_fallback_direct_to_wikipedia():
    # Instantiate ExecutorAgent and configure it to fallback from baidu_search_api directly to wikipedia_bm25_fallback
    agent = ExecutorAgent()
    agent.tool_clusters["web_search"] = ["baidu_search_api", "wikipedia_bm25_fallback"]
    
    params = {"query": "Julius Caesar"}
    res = agent.execute_tool_with_fallback("web_search", params)
    
    # Assert successful retrieval from the wikipedia fallback tool
    assert res == {"birth_year": -100, "era": "BC", "name": "Julius Caesar"}

def test_executor_all_servers_exhausted():
    # Instantiate ExecutorAgent and configure a cluster to only contain broken APIs
    agent = ExecutorAgent()
    agent.tool_clusters["broken_cluster"] = ["baidu_search_api"]
    
    params = {"query": "Will fail"}
    
    # Verify that it raises RuntimeError when everything is exhausted
    with pytest.raises(RuntimeError) as exc_info:
        agent.execute_tool_with_fallback("broken_cluster", params)
        
    assert "All MCP servers in cluster" in str(exc_info.value)
