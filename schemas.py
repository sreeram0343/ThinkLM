"""
schemas.py

Pydantic v2 validation layer for ThinkLM v2 Rubric Engine based on the EvoLM paper (arXiv:2605.03871).
Enforces structural constraints specified in EvoLM (criterion count, weight boundaries, scoring levels, and format checks).
"""

import json
import re
from typing import Dict, List
from pydantic import BaseModel, Field, field_validator, model_validator


class CriterionModel(BaseModel):
    """
    Represents an individual evaluation check within a rubric.
    """
    criterion: str = Field(..., description="Non-empty description of what this criterion measures.")
    weight: float = Field(..., description="Weight of the criterion, strictly in (0.0, 1.0].")
    scoring_levels: Dict[str, str] = Field(
        ...,
        description="Map of float string score levels (e.g. '1.0', '0.5', '0.0') to detailed descriptions."
    )

    @field_validator("criterion")
    @classmethod
    def validate_criterion(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Criterion description must be a non-empty string.")
        return v.strip()

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"Weight must be strictly within (0.0, 1.0]. Got: {v}")
        return v

    @field_validator("scoring_levels")
    @classmethod
    def validate_scoring_levels(cls, v: Dict[str, str]) -> Dict[str, str]:
        if not v:
            raise ValueError("scoring_levels dictionary cannot be empty.")
        
        parsed_keys = set()
        for key, desc in v.items():
            try:
                fk = float(key)
                parsed_keys.add(fk)
            except (ValueError, TypeError):
                raise ValueError(f"scoring_levels key '{key}' must be parseable as a float.")
            
            if not isinstance(desc, str) or not desc.strip():
                raise ValueError(f"Description for scoring level '{key}' must be a non-empty string.")

        if 1.0 not in parsed_keys or 0.0 not in parsed_keys:
            raise ValueError(
                "scoring_levels must explicitly contain both '1.0' (full satisfaction) "
                "and '0.0' (zero satisfaction) keys."
            )
        
        return v


class RubricModel(BaseModel):
    """
    Represents a full set of evaluation criteria for a prompt/question.
    """
    criteria: List[CriterionModel] = Field(..., description="List of 2 to 5 evaluation criteria.")

    @field_validator("criteria")
    @classmethod
    def validate_criteria_count(cls, v: List[CriterionModel]) -> List[CriterionModel]:
        if not (2 <= len(v) <= 5):
            raise ValueError(
                f"Criteria count must be between 2 and 5 (inclusive) to prevent over-segmentation or over-simplification. Got: {len(v)}"
            )
        return v

    @model_validator(mode="after")
    def validate_weight_budget(self) -> "RubricModel":
        total_weight = sum(c.weight for c in self.criteria)
        # R_format tolerance budget of 1.0 ± 0.15 (Appendix A.3)
        min_allowed = 0.85
        max_allowed = 1.15
        if not (min_allowed <= total_weight <= max_allowed):
            raise ValueError(
                f"Sum of criteria weights ({total_weight:.4f}) is outside the R_format tolerance budget "
                f"of 1.0 ± 0.15 [{min_allowed}, {max_allowed}]."
            )
        return self


def parse_and_validate_rubric(json_str: str) -> RubricModel:
    """
    Parses raw JSON output from the rubric generator, strips trailing/leading markdown wrappers,
    and returns a validated RubricModel or raises a detailed ValueError.
    """
    if not json_str or not isinstance(json_str, str):
        raise ValueError("Input json_str must be a non-empty string.")

    cleaned_str = json_str.strip()

    # Strip markdown code fence wrappers if present
    if cleaned_str.startswith("```"):
        cleaned_str = re.sub(r"^```(?:json)?\s*", "", cleaned_str, flags=re.IGNORECASE)
        cleaned_str = re.sub(r"\s*```$", "", cleaned_str)

    cleaned_str = cleaned_str.strip()

    # Extract JSON object substring if surrounded by preambles/postambles
    first_brace = cleaned_str.find("{")
    last_brace = cleaned_str.rfind("}")
    if first_brace != -1 and last_brace != -1 and first_brace < last_brace:
        cleaned_str = cleaned_str[first_brace : last_brace + 1]

    try:
        data = json.loads(cleaned_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse raw JSON content: {e}") from e

    try:
        return RubricModel.model_validate(data)
    except Exception as e:
        raise ValueError(f"Rubric schema validation failed:\n{e}") from e


if __name__ == "__main__":
    print("=" * 70)
    print(" Executing Self-Test for ThinkLM v2 Rubric Engine Validation Layer")
    print("=" * 70)

    # 1. Test Valid Payload
    valid_payload = """
    ```json
    {
      "criteria": [
        {
          "criterion": "Provides the correct final mathematical solution of 144.",
          "weight": 0.8,
          "scoring_levels": {
            "1.0": "The response clearly states 144 as the maximum area.",
            "0.5": "The response arrives at 144 but has minor errors in explanation.",
            "0.0": "The response gives an incorrect maximum area or fails to solve."
          }
        },
        {
          "criterion": "Uses formulas for perimeter and area correctly in the derivation.",
          "weight": 0.2,
          "scoring_levels": {
            "1.0": "Correctly applies 2(L+W)=48 and A=L*W.",
            "0.0": "Formula application contains major algebraic errors."
          }
        }
      ]
    }
    ```
    """

    try:
        validated_rubric = parse_and_validate_rubric(valid_payload)
        print("[PASS] Valid Payload Test:")
        print(f"       Parsed {len(validated_rubric.criteria)} criteria successfully.")
        print(f"       Total weight budget: {sum(c.weight for c in validated_rubric.criteria):.2f}")
    except Exception as e:
        print(f"[FAIL] Valid Payload Test Unexpectedly Failed:\n{e}")

    # 2. Test Malformed Payloads
    malformed_tests = [
        (
            "Invalid Weight Sum (Outside 1.0 ± 0.15)",
            """
            {
              "criteria": [
                {
                  "criterion": "Check answer",
                  "weight": 0.5,
                  "scoring_levels": {"1.0": "Pass", "0.0": "Fail"}
                },
                {
                  "criterion": "Check steps",
                  "weight": 0.2,
                  "scoring_levels": {"1.0": "Pass", "0.0": "Fail"}
                }
              ]
            }
            """
        ),
        (
            "Invalid Criteria Count (Only 1 criterion)",
            """
            {
              "criteria": [
                {
                  "criterion": "Single check",
                  "weight": 1.0,
                  "scoring_levels": {"1.0": "Pass", "0.0": "Fail"}
                }
              ]
            }
            """
        ),
        (
            "Missing '1.0' key in scoring_levels",
            """
            {
              "criteria": [
                {
                  "criterion": "Check logic",
                  "weight": 0.5,
                  "scoring_levels": {"0.8": "Good", "0.0": "Fail"}
                },
                {
                  "criterion": "Check style",
                  "weight": 0.5,
                  "scoring_levels": {"1.0": "Pass", "0.0": "Fail"}
                }
              ]
            }
            """
        ),
        (
            "Weight Out of Bounds (weight = 1.5)",
            """
            {
              "criteria": [
                {
                  "criterion": "Overweighted check",
                  "weight": 1.5,
                  "scoring_levels": {"1.0": "Pass", "0.0": "Fail"}
                },
                {
                  "criterion": "Normal check",
                  "weight": 0.1,
                  "scoring_levels": {"1.0": "Pass", "0.0": "Fail"}
                }
              ]
            }
            """
        )
    ]

    print("\n" + "-" * 70)
    print(" Running Malformed Payload Tests (Expecting Validation Errors)")
    print("-" * 70)

    for title, payload in malformed_tests:
        try:
            parse_and_validate_rubric(payload)
            print(f"[FAIL] {title}: Failed to catch error!")
        except ValueError as e:
            print(f"[PASS] {title}: Caught expected ValueError:")
            first_line = str(e).split('\n')[0]
            print(f"       -> {first_line[:80]}...")

    print("\n" + "=" * 70)
    print(" All Validation Self-Tests Executed.")
    print("=" * 70)
