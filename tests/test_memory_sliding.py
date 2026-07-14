import pytest
import time
from src.memory.memory import DualProcessMemory

def test_episodic_buffer_fifo_eviction():
    # 1. Instantiate the DualProcessMemory with default episodic limit (W=10)
    memory = DualProcessMemory(episodic_limit=10)
    
    # 2. Append 10 messages and verify that the buffer holds exactly 10
    for i in range(1, 11):
        memory.add_episodic_message(
            role="user" if i % 2 != 0 else "assistant",
            content=f"Message number {i}",
            timestamp=float(1000 + i)
        )
    
    assert len(memory.episodic_buffer) == 10
    # First message should still be the oldest (timestamp 1001.0, content "Message number 1")
    assert memory.episodic_buffer[0]["content"] == "Message number 1"
    assert memory.episodic_buffer[-1]["content"] == "Message number 10"
    
    # 3. Append the 11th message. This should trigger eviction of the 1st message (FIFO)
    memory.add_episodic_message(
        role="user",
        content="Message number 11",
        timestamp=1011.0
    )
    
    # Check that buffer size is strictly capped at W=10
    assert len(memory.episodic_buffer) == 10
    
    # Check that the 1st message has been evicted (index 0 is now the 2nd message, content "Message number 2")
    assert memory.episodic_buffer[0]["content"] == "Message number 2"
    assert memory.episodic_buffer[-1]["content"] == "Message number 11"
    
    # 4. Append another (the 12th message), causing the 2nd message to be evicted
    memory.add_episodic_message(
        role="assistant",
        content="Message number 12",
        timestamp=1012.0
    )
    
    assert len(memory.episodic_buffer) == 10
    assert memory.episodic_buffer[0]["content"] == "Message number 3"
    assert memory.episodic_buffer[-1]["content"] == "Message number 12"
