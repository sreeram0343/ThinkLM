import inspect
import asyncio
from typing import Callable, Dict, List, Optional, Tuple, Union


def compute_fleiss_kappa(votes: List[List[int]]) -> float:
    """
    Computes standard Fleiss's Kappa statistical agreement score over binary judge votes
    and clips the result to [0.0, 1.0] (Equation 9).

    Args:
        votes (List[List[int]]): Matrix of shape (N_items, J_judges) where entry is 1 if judge preferred a+ over a-, else 0.

    Returns:
        float: Clipped Fleiss's Kappa in [0.0, 1.0].
    """
    if not votes or not isinstance(votes, list):
        return 0.0

    N = len(votes)
    if N == 0:
        return 0.0

    J = len(votes[0])
    if J <= 1:
        return 1.0

    for row in votes:
        if len(row) != J:
            raise ValueError("All rows in votes matrix must have the same number of judges.")

    n_i0 = [row.count(0) for row in votes]
    n_i1 = [row.count(1) for row in votes]

    P_i_list = []
    for i in range(N):
        sum_sq = (n_i0[i] ** 2) + (n_i1[i] ** 2)
        P_i = (sum_sq - J) / (J * (J - 1))
        P_i_list.append(P_i)

    P_bar = sum(P_i_list) / N
    total_ratings = N * J
    p0 = sum(n_i0) / total_ratings
    p1 = sum(n_i1) / total_ratings
    P_bar_e = (p0 ** 2) + (p1 ** 2)

    if abs(1.0 - P_bar_e) < 1e-9:
        kappa = 1.0
    else:
        kappa = (P_bar - P_bar_e) / (1.0 - P_bar_e)

    return float(max(0.0, min(1.0, kappa)))


def compute_maf_reward(
    scores_pref: List[float],
    scores_dispref: List[float],
    valid_format: bool,
    kappa_val: float,
    w_m: float = 0.5,
    w_f: float = 0.3,
    w_k: float = 0.2,
) -> float:
    """
    Computes continuous Margin+Agreement+Format (MAF) reward using Equation 9:
    R_MJ = w_m * mean_margin + w_f * 1[valid_format] + w_k * kappa_val
    """
    if len(scores_pref) != len(scores_dispref):
        raise ValueError("scores_pref and scores_dispref must have identical lengths.")
    if not scores_pref:
        return 0.0

    margins = [p - d for p, d in zip(scores_pref, scores_dispref)]
    mean_margin = sum(margins) / len(margins)
    format_indicator = 1.0 if valid_format else 0.0

    return float(w_m * mean_margin + w_f * format_indicator + w_k * kappa_val)


def compute_ba_reward(votes_pref_ratio: float, kappa_val: float) -> float:
    """
    Computes Alternative Binary Agreement (BA) reward using Equation 10:
    R_BA = average_vote - (1.0 - kappa_val)
    """
    return float(votes_pref_ratio - (1.0 - kappa_val))


async def _evaluate_single_judge(
    judge_fn: Callable[[str, str, str, str], Tuple[float, float]],
    question: str,
    rubric: str,
    answer_pref: str,
    answer_dispref: str,
) -> Tuple[float, float]:
    """Helper coroutine executing a single judge function in an async pool."""
    if inspect.iscoroutinefunction(judge_fn):
        return await judge_fn(question, rubric, answer_pref, answer_dispref)
    else:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, judge_fn, question, rubric, answer_pref, answer_dispref
        )


def evaluate_rubric_ensemble(
    judges: List[Callable[[str, str, str, str], Tuple[float, float]]],
    question: str,
    rubric: str,
    answer_pref: str,
    answer_dispref: str,
    valid_format: bool = True,
) -> Dict[str, Union[float, List[float], List[int]]]:
    """
    Orchestrates evaluation across an ensemble of J frozen judges in parallel.
    """
    if not judges:
        raise ValueError("Judges list cannot be empty.")

    async def _run_pool():
        tasks = [
            _evaluate_single_judge(j_fn, question, rubric, answer_pref, answer_dispref)
            for j_fn in judges
        ]
        return await asyncio.gather(*tasks)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        results = asyncio.run_coroutine_threadsafe(_run_pool(), loop).result()
    else:
        results = asyncio.run(_run_pool())

    scores_pref = [r[0] for r in results]
    scores_dispref = [r[1] for r in results]

    votes = [1 if p > d else 0 for p, d in zip(scores_pref, scores_dispref)]
    votes_matrix = [votes]
    kappa_val = compute_fleiss_kappa(votes_matrix)

    votes_pref_ratio = sum(votes) / len(votes)
    maf_reward = compute_maf_reward(scores_pref, scores_dispref, valid_format, kappa_val)
    ba_reward = compute_ba_reward(votes_pref_ratio, kappa_val)

    margins = [p - d for p, d in zip(scores_pref, scores_dispref)]

    return {
        "scores_pref": scores_pref,
        "scores_dispref": scores_dispref,
        "margins": margins,
        "mean_margin": float(sum(margins) / len(margins)),
        "votes": votes,
        "votes_pref_ratio": float(votes_pref_ratio),
        "kappa": kappa_val,
        "maf_reward": maf_reward,
        "ba_reward": ba_reward,
    }


if __name__ == "__main__":
    print("=" * 70)
    print(" Executing Self-Test for Multi-Judge Evaluator (judge.py)")
    print("=" * 70)

    sample_votes = [
        [1, 1, 1, 1],
        [1, 1, 0, 1],
        [0, 0, 0, 0],
        [1, 0, 1, 0],
        [1, 1, 1, 0],
    ]
    kappa = compute_fleiss_kappa(sample_votes)
    print(f"[PASS] Fleiss's Kappa calculation: {kappa:.4f} (clipped in [0.0, 1.0])")

    scores_p = [0.9, 0.85, 0.75, 0.8]
    scores_d = [0.4, 0.50, 0.70, 0.3]
    maf_r = compute_maf_reward(scores_p, scores_d, valid_format=True, kappa_val=0.85)
    ba_r = compute_ba_reward(votes_pref_ratio=1.0, kappa_val=0.85)

    print("\n[PASS] MAF Reward Computation:")
    print(f"       MAF Reward (w_m=0.5, w_f=0.3, w_k=0.2): {maf_r:.4f}")
    print(f"       BA Reward: {ba_r:.4f}")

    def mock_judge_1(q, r, a_pos, a_neg):
        return (0.9, 0.4)

    def mock_judge_2(q, r, a_pos, a_neg):
        return (0.85, 0.5)

    def mock_judge_3(q, r, a_pos, a_neg):
        return (0.7, 0.6)

    judge_ensemble = [mock_judge_1, mock_judge_2, mock_judge_3]

    eval_result = evaluate_rubric_ensemble(
        judges=judge_ensemble,
        question="Explain photosynthesis",
        rubric="1. Mention sunlight. 2. Mention CO2.",
        answer_pref="Plants convert sunlight and CO2 into glucose.",
        answer_dispref="Plants grow in dirt.",
        valid_format=True,
    )

    print("\n[PASS] Multi-Judge Ensemble Evaluation Output:")
    print(f"       Preferred Scores  : {eval_result['scores_pref']}")
    print(f"       Dispreferred Scores: {eval_result['scores_dispref']}")
    print(f"       Mean Margin       : {eval_result['mean_margin']:.4f}")
    print(f"       Votes             : {eval_result['votes']}")
    print(f"       Kappa             : {eval_result['kappa']:.4f}")
    print(f"       MAF Reward        : {eval_result['maf_reward']:.4f}")
    print(f"       BA Reward         : {eval_result['ba_reward']:.4f}")

    print("\n" + "=" * 70)
    print(" All judge.py Self-Tests Executed Successfully.")
    print("=" * 70)
