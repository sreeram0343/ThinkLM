import pytest
from thinklm.rubric.preference import PreferencePairEngine, InsufficientHistoryError

def test_buffer_capacity_and_thread_safety():
    engine = PreferencePairEngine(buffer_capacity=5)
    for i in range(10):
        engine.add_experience(step_id=i, question="Q", answer=f"A_{i}")
    
    assert engine.get_buffer_size() == 5

def test_temporal_contrast_valid_sampling():
    engine = PreferencePairEngine(min_age_gap=20, max_age_gap=100)
    engine.add_experience(step_id=10, question="Math Q", answer="Ans_10")
    engine.add_experience(step_id=50, question="Math Q", answer="Ans_50")
    
    # At step 100, step 50 has age gap 50 (valid in [20, 100])
    pos, neg = engine.generate_temporal_contrast("Math Q", "Ans_100", current_step=100)
    assert pos == "Ans_100"
    assert neg == "Ans_50"

def test_temporal_contrast_insufficient_history_error():
    engine = PreferencePairEngine(min_age_gap=20, max_age_gap=100)
    with pytest.raises(InsufficientHistoryError):
        engine.generate_temporal_contrast("Non-existent Q", "Current Ans", current_step=50, allow_broader_gap_fallback=False)

def test_inferred_question_pipeline():
    engine = PreferencePairEngine()
    mock_inf_llm = lambda sys_prompt, ans: "Inferred Q"
    mock_pol_llm = lambda q: f"Response for {q}"
    
    pos, neg = engine.generate_inferred_question("Current Ans", mock_inf_llm, mock_pol_llm)
    assert pos == "Current Ans"
    assert "Inferred Q" in neg

def test_rubric_conditioned_pipeline():
    engine = PreferencePairEngine()
    mock_pol_llm = lambda sys_prompt, q: "With Rubric" if sys_prompt else "Without Rubric"
    
    pos, neg = engine.generate_rubric_conditioned("Target Q", "Rule 1", mock_pol_llm)
    assert pos == "With Rubric"
    assert neg == "Without Rubric"

def test_master_sample_preference_pair():
    engine = PreferencePairEngine(min_age_gap=5, max_age_gap=50)
    engine.add_experience(step_id=10, question="Q", answer="History_A")
    
    mock_inf_llm = lambda sys, ans: "q_hat"
    mock_pol_llm = lambda sys, q=None: "Ans"
    
    pos, neg, strat = engine.sample_preference_pair(
        question="Q",
        current_answer="Curr_A",
        current_step=30,
        rubric="R",
        policy_llm_call=mock_pol_llm,
        inference_llm_call=mock_inf_llm
    )
    assert pos is not None
    assert neg is not None
    assert strat in ["temporal_contrast", "inferred_question", "rubric_conditioned"]
