import pytest
from schemas import RubricModel, CriterionModel, parse_and_validate_rubric

def test_valid_rubric_parsing():
    payload = """
    ```json
    {
      "criteria": [
        {
          "criterion": "Direct factual correctness",
          "weight": 0.7,
          "scoring_levels": {"1.0": "Fully correct", "0.5": "Partially correct", "0.0": "Incorrect"}
        },
        {
          "criterion": "Format compliance",
          "weight": 0.3,
          "scoring_levels": {"1.0": "Compliant", "0.0": "Non-compliant"}
        }
      ]
    }
    ```
    """
    model = parse_and_validate_rubric(payload)
    assert len(model.criteria) == 2
    assert model.criteria[0].weight == 0.7
    assert model.criteria[1].weight == 0.3

def test_invalid_criteria_count():
    payload = '{"criteria": [{"criterion": "Single", "weight": 1.0, "scoring_levels": {"1.0": "A", "0.0": "B"}}]}'
    with pytest.raises(ValueError, match="Criteria count must be between 2 and 5"):
        parse_and_validate_rubric(payload)

def test_invalid_weight_budget():
    payload = """
    {
      "criteria": [
        {"criterion": "A", "weight": 0.2, "scoring_levels": {"1.0": "Pass", "0.0": "Fail"}},
        {"criterion": "B", "weight": 0.2, "scoring_levels": {"1.0": "Pass", "0.0": "Fail"}}
      ]
    }
    """
    with pytest.raises(ValueError, match="tolerance budget"):
        parse_and_validate_rubric(payload)

def test_missing_required_scoring_level_keys():
    payload = """
    {
      "criteria": [
        {"criterion": "A", "weight": 0.5, "scoring_levels": {"0.8": "Good", "0.0": "Fail"}},
        {"criterion": "B", "weight": 0.5, "scoring_levels": {"1.0": "Pass", "0.0": "Fail"}}
      ]
    }
    """
    with pytest.raises(ValueError, match="scoring_levels must explicitly contain both '1.0'"):
        parse_and_validate_rubric(payload)
