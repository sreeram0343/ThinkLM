import pytest
import time
import threading
from src.agents.master import MasterAgent, MasterScaffolding, MasterExecutionHarness
from src.memory.memory import DualProcessMemory

def test_eager_scaffolding_isolation():
    scaffolding = MasterScaffolding()
    
    # Assert components are eagerly initialized
    assert scaffolding.planner is not None
    assert scaffolding.executor is not None
    assert scaffolding.writer is not None
    
    # Assert prompts and tool schemas are compiled
    assert "SAFETY OVERLAY" in scaffolding.system_prompt
    assert "web_search" in scaffolding.tool_schemas
    assert "calculator" in scaffolding.tool_schemas
    assert "sandbox" in scaffolding.tool_schemas

def test_six_phase_react_loop():
    agent = MasterAgent()
    memory = DualProcessMemory(episodic_limit=5)
    
    result = agent.run_collaborative_loop("What is the weather in Beijing today?", memory)
    assert result["status"] == "SUCCESS"
    assert "Beijing's weather today" in result["final_answer"]
    
    # Verify trajectory steps for the six phases
    steps = result["trajectory_steps"]
    assert any("Phase 1: Pre-check & Compaction" in step for step in steps)
    assert any("Phase 2: Thinking" in step for step in steps)
    assert any("Phase 3: Self-Critique" in step for step in steps)
    assert any("Phase 4: Action" in step for step in steps)
    assert any("Phase 5: Tool Execution" in step for step in steps)
    assert any("Phase 6: Post-processing" in step for step in steps)

def test_thread_safe_cancellation():
    agent = MasterAgent()
    memory = DualProcessMemory(episodic_limit=5)
    
    # Initialize the harness manually to inspect and test the thread-safe queue
    harness = MasterExecutionHarness(agent.scaffolding, "Calculate age difference of Han-Wu and Julius Caesar", memory)
    
    # Ingest cancel signal prior to or during execution
    harness.ingest_message({"action": "cancel"})
    
    # Run the loop and assert cancellation behavior
    result = harness.run_loop()
    assert result["status"] == "CANCELLED"
    assert "cancelled" in result["final_answer"].lower()
    assert any("Cancellation signal received" in step for step in result["trajectory_steps"])

def test_thread_safe_followup_message():
    agent = MasterAgent()
    memory = DualProcessMemory(episodic_limit=5)
    
    # Initialize the harness
    harness = MasterExecutionHarness(agent.scaffolding, "Calculate age difference of Han-Wu and Julius Caesar", memory)
    
    # Ingest a user follow-up message
    harness.ingest_message({"type": "user_message", "content": "Also consider MS Dhoni"})
    
    result = harness.run_loop()
    # Verify query updated dynamically in context
    assert "Also consider MS Dhoni" in result["query"]
    assert any("Received follow-up query: 'Also consider MS Dhoni'" in step for step in result["trajectory_steps"])
