import pytest
from thinklm.training.loop import (
    AlternatingTrainingLoop,
    ActiveSamplingGuard,
    CoEvolveConfig,
    PhaseState,
)

def test_phase_state_transitions():
    cfg = CoEvolveConfig(K_steps=10)
    loop_mgr = AlternatingTrainingLoop(config=cfg)
    
    # Steps 0-9: POLICY_TRAINING
    for s in range(10):
        assert loop_mgr.determine_phase(s) == PhaseState.POLICY_TRAINING
        
    # Steps 10-19: RUBRIC_TRAINING
    for s in range(10, 20):
        assert loop_mgr.determine_phase(s) == PhaseState.RUBRIC_TRAINING
        
    # Step 20: Back to POLICY_TRAINING
    assert loop_mgr.determine_phase(20) == PhaseState.POLICY_TRAINING

def test_active_sampling_guard():
    guard = ActiveSamplingGuard()
    assert guard.is_informative_batch([0.5, 0.5, 0.5]) is False
    assert guard.is_informative_batch([0.1, 0.9, 0.5]) is True

def test_async_batch_enqueue():
    cfg = CoEvolveConfig(K_steps=5, async_steps=4)
    loop_mgr = AlternatingTrainingLoop(config=cfg)
    
    rej = loop_mgr.enqueue_async_batch({"rewards": [0.5, 0.5]})
    acc = loop_mgr.enqueue_async_batch({"rewards": [0.1, 0.8]})
    
    assert rej is False
    assert acc is True
    assert loop_mgr.dequeue_async_batch() == {"rewards": [0.1, 0.8]}

def test_step_execution_metrics():
    cfg = CoEvolveConfig(K_steps=2)
    loop_mgr = AlternatingTrainingLoop(config=cfg)
    
    s0 = loop_mgr.step()
    s1 = loop_mgr.step()
    s2 = loop_mgr.step()
    
    assert s0["phase"] == "POLICY_TRAINING"
    assert s1["phase"] == "POLICY_TRAINING"
    assert s2["phase"] == "RUBRIC_TRAINING"
