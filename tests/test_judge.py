import pytest
from thinklm.rubric.judge import (
    compute_fleiss_kappa,
    compute_maf_reward,
    compute_ba_reward,
    evaluate_rubric_ensemble,
)

def test_fleiss_kappa_perfect_agreement():
    votes = [
        [1, 1, 1],
        [0, 0, 0],
    ]
    kappa = compute_fleiss_kappa(votes)
    assert kappa == 1.0

def test_fleiss_kappa_clipping():
    # Split votes leading to low or negative agreement
    votes = [
        [1, 0, 1, 0],
        [0, 1, 0, 1],
    ]
    kappa = compute_fleiss_kappa(votes)
    assert 0.0 <= kappa <= 1.0

def test_maf_reward_equation9():
    scores_pref = [0.8, 0.9, 0.7]
    scores_dispref = [0.2, 0.4, 0.3]
    # Margins: [0.6, 0.5, 0.4] -> Mean Margin: 0.5
    # w_m=0.5 * 0.5 = 0.25
    # w_f=0.3 * 1.0 = 0.30
    # w_k=0.2 * 0.8 = 0.16
    # Total MAF = 0.25 + 0.30 + 0.16 = 0.71
    maf = compute_maf_reward(scores_pref, scores_dispref, valid_format=True, kappa_val=0.8)
    assert abs(maf - 0.71) < 1e-4

def test_ba_reward_equation10():
    # R_BA = v_bar - (1.0 - kappa) = 0.9 - (1.0 - 0.7) = 0.6
    ba = compute_ba_reward(votes_pref_ratio=0.9, kappa_val=0.7)
    assert abs(ba - 0.6) < 1e-4

def test_evaluate_rubric_ensemble_orchestrator():
    def mock_j1(q, r, ap, an):
        return (0.9, 0.3)

    def mock_j2(q, r, ap, an):
        return (0.8, 0.4)

    res = evaluate_rubric_ensemble(
        judges=[mock_j1, mock_j2],
        question="Q",
        rubric="R",
        answer_pref="A+",
        answer_dispref="A-",
        valid_format=True,
    )

    assert len(res["scores_pref"]) == 2
    assert res["votes"] == [1, 1]
    assert res["kappa"] == 1.0
    assert "maf_reward" in res
    assert "ba_reward" in res
