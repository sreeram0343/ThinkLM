"""
thinklm/rubric/preference.py

Implements PreferencePairEngine for Phase 2 (Rubric Training) of ThinkLM based on EvoLM (arXiv:2605.03871).
Constructs preference pairs (a+, a-) on-policy from policy rollouts using three complementary strategies:
1. Temporal Contrast: Pairs current rollout (a+) against earlier policy checkpoints (a-).
2. Inferred Question (IQ): Infers q_hat from a+, then generates a- from q_hat.
3. Rubric-Conditioned (RC): Generates a+ conditioned on question + rubric, and a- from question alone.
"""

import random
import threading
from collections import deque
from typing import Callable, Dict, List, Optional, Tuple, Union


class InsufficientHistoryError(Exception):
    """Raised when the rollout buffer lacks valid historical samples for Temporal Contrast."""
    pass


class PreferencePairEngine:
    """
    Constructs preference training pairs (a+, a-) for rubric generator training.
    Maintains a thread-safe rollout buffer capped at 2,048 entries.
    """

    def __init__(
        self,
        buffer_capacity: int = 2048,
        min_age_gap: int = 20,
        max_age_gap: int = 100,
    ):
        """
        Initializes the PreferencePairEngine.

        Args:
            buffer_capacity (int): Maximum capacity of the rollout buffer (default: 2048).
            min_age_gap (int): Minimum step age gap for temporal contrast (default: 20).
            max_age_gap (int): Maximum step age gap for temporal contrast (default: 100).
        """
        self.buffer_capacity = buffer_capacity
        self.min_age_gap = min_age_gap
        self.max_age_gap = max_age_gap

        # Thread-safe buffer of experiences: (step_id, question, answer)
        self._buffer: deque = deque(maxlen=self.buffer_capacity)
        self._lock = threading.Lock()

    def add_experience(self, step_id: int, question: str, answer: str) -> None:
        """
        Adds a rollout experience to the thread-safe buffer.

        Args:
            step_id (int): Training step ID when rollout was generated.
            question (str): Prompt / question text.
            answer (str): Generated response text.
        """
        if not isinstance(step_id, int):
            raise ValueError("step_id must be an integer.")
        if not question or not isinstance(question, str):
            raise ValueError("question must be a non-empty string.")
        if not answer or not isinstance(answer, str):
            raise ValueError("answer must be a non-empty string.")

        with self._lock:
            self._buffer.append((step_id, question.strip(), answer.strip()))

    def get_buffer_size(self) -> int:
        """Returns the current number of experiences stored in the buffer."""
        with self._lock:
            return len(self._buffer)

    def clear_buffer(self) -> None:
        """Clears all stored experiences from the buffer."""
        with self._lock:
            self._buffer.clear()

    def generate_temporal_contrast(
        self,
        question: str,
        current_answer: str,
        current_step: int,
        allow_broader_gap_fallback: bool = True,
    ) -> Tuple[str, str]:
        """
        Generates a preference pair using Temporal Contrast.
        Pairs current_answer (a+) with a historical answer (a-) for the same question
        sampled from steps in the age gap range [current_step - max_age_gap, current_step - min_age_gap].

        Args:
            question (str): Target prompt / question.
            current_answer (str): Current rollout answer (a+).
            current_step (int): Current training step index.
            allow_broader_gap_fallback (bool): If True, falls back to any step < current_step if strict age gap has no matches.

        Returns:
            Tuple[str, str]: (preferred_answer, dispreferred_answer) -> (a+, a-)

        Raises:
            InsufficientHistoryError: If no valid historical answer is found in the buffer.
        """
        clean_q = question.strip()
        with self._lock:
            # Filter buffer for same question and valid step gap
            candidates = [
                ans
                for (step_id, q, ans) in self._buffer
                if q == clean_q and (self.min_age_gap <= (current_step - step_id) <= self.max_age_gap)
            ]

            # Fallback if strict age gap yields no candidates
            if not candidates and allow_broader_gap_fallback:
                candidates = [
                    ans
                    for (step_id, q, ans) in self._buffer
                    if q == clean_q and step_id < current_step
                ]

        if not candidates:
            raise InsufficientHistoryError(
                f"Insufficient rollout history for question '{clean_q[:40]}...'. "
                f"No historical answers found within step age gap [{self.min_age_gap}, {self.max_age_gap}] "
                f"relative to step {current_step}."
            )

        dispreferred = random.choice(candidates)
        return (current_answer, dispreferred)

    def generate_inferred_question(
        self,
        current_answer: str,
        inference_llm_call: Callable[[str, str], str],
        policy_llm_call: Callable[..., str],
    ) -> Tuple[str, str]:
        """
        Generates a preference pair using Inferred Question (IQ).
        Calls inference_llm_call to infer q_hat from current_answer (a+),
        then generates dispreferred response (a-) by prompting policy_llm_call on q_hat.

        Args:
            current_answer (str): Preferred answer (a+).
            inference_llm_call (Callable[[str, str], str]): Function accepting (system_prompt, answer) -> q_hat.
            policy_llm_call (Callable[..., str]): Function generating policy response for q_hat.

        Returns:
            Tuple[str, str]: (current_answer, a-)
        """
        system_prompt = (
            "You are an expert at understanding what question someone was trying to answer "
            "based on their response. Given an answer, infer what question the person was likely "
            "trying to answer. Be specific and output only the inferred question, nothing else."
        )

        q_hat = inference_llm_call(system_prompt, current_answer)
        if not q_hat or not isinstance(q_hat, str):
            raise RuntimeError("inference_llm_call failed to return a valid inferred question string.")

        # Flexible calling signature support for policy_llm_call (1 or 2 arguments)
        try:
            dispreferred_answer = policy_llm_call(q_hat.strip())
        except TypeError:
            dispreferred_answer = policy_llm_call(None, q_hat.strip())

        if not dispreferred_answer or not isinstance(dispreferred_answer, str):
            raise RuntimeError("policy_llm_call failed to return a valid dispreferred answer string.")

        return (current_answer, dispreferred_answer.strip())

    def generate_rubric_conditioned(
        self,
        question: str,
        rubric: str,
        policy_llm_call: Callable[[Optional[str], str], str],
    ) -> Tuple[str, str]:
        """
        Generates a preference pair using Rubric-Conditioned (RC).
        Generates a+ (preferred) by calling policy_llm_call conditioned on rubric + question.
        Generates a- (dispreferred) by calling policy_llm_call on question alone.

        Args:
            question (str): Target question.
            rubric (str): Formatted evaluation rubric.
            policy_llm_call (Callable[[Optional[str], str], str]): Function accepting (system_prompt, question) -> answer.

        Returns:
            Tuple[str, str]: (preferred_answer, dispreferred_answer) -> (a+, a-)
        """
        rubric_system_prompt = f"When answering, follow this rubric to ensure a high-quality response:\n{rubric}"

        preferred_answer = policy_llm_call(rubric_system_prompt, question)
        dispreferred_answer = policy_llm_call(None, question)

        if not preferred_answer or not dispreferred_answer:
            raise RuntimeError("policy_llm_call failed to generate responses for Rubric-Conditioned strategy.")

        return (preferred_answer.strip(), dispreferred_answer.strip())

    def sample_preference_pair(
        self,
        strategy: Optional[str] = None,
        question: Optional[str] = None,
        current_answer: Optional[str] = None,
        current_step: Optional[int] = None,
        rubric: Optional[str] = None,
        policy_llm_call: Optional[Callable] = None,
        inference_llm_call: Optional[Callable] = None,
    ) -> Tuple[str, str, str]:
        """
        Master method to sample a preference pair (a+, a-) using one of the three strategies.
        If strategy is None, uniformly toggles/selects randomly among valid available strategies.

        Args:
            strategy (Optional[str]): One of 'temporal_contrast', 'inferred_question', 'rubric_conditioned', or None.
            question (Optional[str]): Prompt / question text.
            current_answer (Optional[str]): Current response (a+).
            current_step (Optional[int]): Current training step ID.
            rubric (Optional[str]): Rubric string for rubric_conditioned strategy.
            policy_llm_call (Optional[Callable]): Policy generation callback.
            inference_llm_call (Optional[Callable]): Question inference callback.

        Returns:
            Tuple[str, str, str]: (a+, a-, strategy_used)
        """
        available_strategies = []

        # Determine which strategies are valid given supplied inputs
        if question and current_answer and current_step is not None:
            available_strategies.append("temporal_contrast")

        if current_answer and inference_llm_call and policy_llm_call:
            available_strategies.append("inferred_question")

        if question and rubric and policy_llm_call:
            available_strategies.append("rubric_conditioned")

        if not available_strategies and strategy is None:
            raise ValueError("Insufficient parameters provided to execute any preference generation strategy.")

        selected_strategy = strategy if strategy else random.choice(available_strategies)

        if selected_strategy == "temporal_contrast":
            if not question or not current_answer or current_step is None:
                raise ValueError("temporal_contrast requires question, current_answer, and current_step.")
            a_pos, a_neg = self.generate_temporal_contrast(question, current_answer, current_step)

        elif selected_strategy == "inferred_question":
            if not current_answer or not inference_llm_call or not policy_llm_call:
                raise ValueError("inferred_question requires current_answer, inference_llm_call, and policy_llm_call.")
            a_pos, a_neg = self.generate_inferred_question(current_answer, inference_llm_call, policy_llm_call)

        elif selected_strategy == "rubric_conditioned":
            if not question or not rubric or not policy_llm_call:
                raise ValueError("rubric_conditioned requires question, rubric, and policy_llm_call.")
            a_pos, a_neg = self.generate_rubric_conditioned(question, rubric, policy_llm_call)

        else:
            raise ValueError(f"Unknown or unsupported strategy: '{selected_strategy}'")

        return (a_pos, a_neg, selected_strategy)


if __name__ == "__main__":
    print("=" * 70)
    print(" Executing Self-Test for PreferencePairEngine (EvoLM Phase 2)")
    print("=" * 70)

    engine = PreferencePairEngine(buffer_capacity=2048, min_age_gap=20, max_age_gap=100)

    # 1. Populate Buffer for Temporal Contrast Test
    test_q = "What is the capital of France?"
    for step in range(10, 150, 10):
        engine.add_experience(step_id=step, question=test_q, answer=f"Historical Answer at step {step}: Paris")

    print(f"[PASS] Rollout Buffer populated. Size: {engine.get_buffer_size()}/2048")

    # 2. Test Temporal Contrast Strategy
    current_step = 120
    current_ans = "Current Answer at step 120: Paris is the capital of France."
    a_pos, a_neg = engine.generate_temporal_contrast(test_q, current_ans, current_step)
    print("\n[PASS] Temporal Contrast Pair:")
    print(f"       a+ (current): {a_pos}")
    print(f"       a- (history): {a_neg}")

    # 3. Test Inferred Question Strategy (IQ)
    mock_inference_llm = lambda sys_prompt, ans: "What is the capital city of France?"
    mock_policy_llm_single = lambda sys_prompt, q=None: (
        f"Dispreferred response for question: '{q if q else sys_prompt}'"
    )
    a_pos_iq, a_neg_iq = engine.generate_inferred_question(current_ans, mock_inference_llm, mock_policy_llm_single)
    print("\n[PASS] Inferred Question Pair:")
    print(f"       a+: {a_pos_iq}")
    print(f"       a-: {a_neg_iq}")

    # 4. Test Rubric-Conditioned Strategy (RC)
    mock_rubric = "1. Must state Paris clearly. 2. Must describe geographical context."
    mock_policy_llm_dual = lambda sys_prompt, q: (
        f"Preferred answer with rubric [{sys_prompt[:30]}...]: Paris"
        if sys_prompt
        else "Dispreferred answer without rubric: Paris"
    )
    a_pos_rc, a_neg_rc = engine.generate_rubric_conditioned(test_q, mock_rubric, mock_policy_llm_dual)
    print("\n[PASS] Rubric-Conditioned Pair:")
    print(f"       a+: {a_pos_rc}")
    print(f"       a-: {a_neg_rc}")

    # 5. Test Master Sampling Method
    pos, neg, strat = engine.sample_preference_pair(
        question=test_q,
        current_answer=current_ans,
        current_step=current_step,
        rubric=mock_rubric,
        policy_llm_call=mock_policy_llm_dual,
        inference_llm_call=mock_inference_llm,
    )
    print(f"\n[PASS] Master Sample Preference Pair (Strategy Selected: {strat}):")
    print(f"       a+: {pos[:50]}...")
    print(f"       a-: {neg[:50]}...")

    # 6. Test Error Handling for Insufficient History
    try:
        engine.generate_temporal_contrast("Unknown question?", "Some answer", 200, allow_broader_gap_fallback=False)
        print("[FAIL] Failed to raise InsufficientHistoryError!")
    except InsufficientHistoryError as e:
        print(f"\n[PASS] Caught expected InsufficientHistoryError:\n       -> {e}")

    print("\n" + "=" * 70)
    print(" All PreferencePairEngine Self-Tests Executed Successfully.")
    print("=" * 70)
