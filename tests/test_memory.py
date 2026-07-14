import pytest
import time
from src.memory.memory import DualProcessMemory

def test_dual_process_memory_init():
    memory = DualProcessMemory(episodic_limit=5, decay_rate=0.02, spreading_factor=0.7)
    assert memory.episodic_limit == 5
    assert memory.decay_rate == 0.02
    assert memory.spreading_factor == 0.7
    assert len(memory.episodic_buffer) == 0
    assert len(memory.semantic_graph) == 0

def test_add_episodic_message():
    memory = DualProcessMemory(episodic_limit=2)
    memory.add_episodic_message("user", "Hello first", time.time())
    assert len(memory.episodic_buffer) == 1
    
    memory.add_episodic_message("assistant", "Hello response", time.time())
    assert len(memory.episodic_buffer) == 2
    assert memory.episodic_buffer[0]["content"] == "Hello first"

def test_add_semantic_fact_and_inquiry():
    memory = DualProcessMemory()
    memory.add_semantic_fact("Han-Wu", "Wu of Han", "alias", weight=1.0)
    assert memory.semantic_graph.has_node("Han-Wu")
    assert memory.semantic_graph.has_node("Wu of Han")
    assert memory.semantic_graph.has_edge("Han-Wu", "Wu of Han")
