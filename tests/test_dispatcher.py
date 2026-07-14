import pytest
import time
from src.memory.memory import DualProcessMemory
from src.agents.master import MasterAgent
from src.utils.config import ConfigManager
from src.utils.dispatcher import REPLDispatcher

@pytest.fixture
def setup_repl_env():
    memory = DualProcessMemory(episodic_limit=5)
    master_agent = MasterAgent()
    config_manager = ConfigManager()
    dispatcher = REPLDispatcher(memory, master_agent, config_manager)
    return memory, master_agent, config_manager, dispatcher

def test_dispatcher_mode_toggle(setup_repl_env):
    memory, master_agent, config_manager, dispatcher = setup_repl_env
    
    assert config_manager.mode == "Normal Mode"
    
    # Toggle to Plan Mode
    response = dispatcher.dispatch("/mode")
    assert "Plan Mode" in response
    assert config_manager.mode == "Plan Mode"
    assert config_manager.pending_plan_mode is True
    
    # Toggle back to Normal Mode
    response = dispatcher.dispatch("/mode")
    assert "Normal Mode" in response
    assert config_manager.mode == "Normal Mode"
    assert config_manager.pending_plan_mode is False

def test_dispatcher_clear_episodic_buffer(setup_repl_env):
    memory, master_agent, config_manager, dispatcher = setup_repl_env
    
    # Add messages
    memory.add_episodic_message("user", "Hello 1", time.time())
    memory.add_episodic_message("assistant", "Hi 1", time.time())
    assert len(memory.episodic_buffer) == 2
    
    # Clear buffer via command
    response = dispatcher.dispatch("/clear")
    assert "cleared" in response.lower()
    assert len(memory.episodic_buffer) == 0

def test_dispatcher_undo_state_snapshot(setup_repl_env):
    memory, master_agent, config_manager, dispatcher = setup_repl_env
    
    # Initial state
    assert config_manager.mode == "Normal Mode"
    assert len(memory.episodic_buffer) == 0
    
    # Run a plain query (mutates memory and saves snapshot)
    ans = dispatcher.dispatch("Emperor Han-Wu birth name")
    assert "劉徹" in ans
    assert len(memory.episodic_buffer) == 2  # contains user query and assistant answer
    
    # Undo it
    response = dispatcher.dispatch("/undo")
    assert "Reverted" in response
    assert len(memory.episodic_buffer) == 0
    
    # Test undo mode toggle
    dispatcher.dispatch("/mode")
    assert config_manager.mode == "Plan Mode"
    
    dispatcher.dispatch("/undo")
    assert config_manager.mode == "Normal Mode"

def test_dispatcher_mcp_commands(setup_repl_env):
    memory, master_agent, config_manager, dispatcher = setup_repl_env
    
    # List MCP tools
    list_response = dispatcher.dispatch("/mcp list")
    assert "web_search" in list_response
    assert "calculator" in list_response
    
    # Register new MCP tool
    reg_response = dispatcher.dispatch("/mcp register custom_tool standard Python JSON lookup")
    assert "Successfully registered" in reg_response
    assert "custom_tool" in dispatcher.dispatch("/mcp list")
    
    # Evaluate MCP tool
    eval_response = dispatcher.dispatch("/mcp evaluate custom_tool")
    assert "Evaluating MCP tool 'custom_tool'" in eval_response
    assert "standard Python JSON lookup" in eval_response
    
    # Non-existent tool evaluation
    eval_fail = dispatcher.dispatch("/mcp evaluate non_existent")
    assert "not found" in eval_fail

def test_dispatcher_query_routing(setup_repl_env):
    memory, master_agent, config_manager, dispatcher = setup_repl_env
    
    # Verify standard query routing
    ans = dispatcher.dispatch("What is the weather in Beijing today?")
    assert "Beijing's weather today" in ans
    assert len(memory.episodic_buffer) == 2
    assert memory.episodic_buffer[0]["content"] == "What is the weather in Beijing today?"
    assert memory.episodic_buffer[1]["role"] == "assistant"
