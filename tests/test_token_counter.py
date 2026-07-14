import os
import json
import pytest
from src.utils.token_counter import TokenAuditor

def test_token_auditor_init():
    auditor = TokenAuditor()
    assert auditor is not None
    assert auditor.count_tokens("") == 0
    assert auditor.count_tokens("hello world") > 0

def test_format_tools_section():
    auditor = TokenAuditor()
    tools = [
        {"name": "tool_1", "description": "This is tool 1", "parameters": {}},
        {"name": "tool_2", "description": "This is tool 2", "parameters": {}}
    ]
    formatted = auditor.format_tools_section(tools)
    assert "tool_1" in formatted
    assert "tool_2" in formatted
    
    empty_formatted = auditor.format_tools_section([])
    assert "No tools available." in empty_formatted

def test_assemble_prompt():
    auditor = TokenAuditor()
    query = "Find the weather in Beijing"
    tools = [{"name": "weather_api", "description": "Gets current weather"}]
    prompt = auditor.assemble_prompt(query, tools)
    
    assert "weather_api" in prompt
    assert query in prompt
    assert "SAFETY OVERLAY" in prompt

def test_audit():
    auditor = TokenAuditor()
    query = "Find the weather in Beijing"
    
    all_tools = [
        {"name": "weather_api", "description": "Gets current weather"},
        {"name": "search_api", "description": "Performs general web search"},
        {"name": "calc_api", "description": "Calculates math equations"},
    ]
    
    pruned_tools = [
        {"name": "weather_api", "description": "Gets current weather"},
    ]
    
    metrics = auditor.audit(query, all_tools, pruned_tools)
    
    assert "monolithic_tokens" in metrics
    assert "pruned_tokens" in metrics
    assert "reduction" in metrics
    assert "reduction_percentage" in metrics
    
    assert metrics["monolithic_tokens"] > metrics["pruned_tokens"]
    assert metrics["reduction"] > 0
    assert metrics["reduction_percentage"] > 0.0
