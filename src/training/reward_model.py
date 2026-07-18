from typing import List


class RewardModel:
    """
    Simple heuristic reward model.

    Later this can be replaced with a verifier
    or external evaluator.
    """

    def score(self, responses: List[str]) -> List[float]:

        rewards = []

        for response in responses:

            score = 0.0

            # Reward longer responses
            score += min(len(response.split()) / 50.0, 1.0)

            # Reward complete sentences
            if response.endswith("."):
                score += 0.2

            rewards.append(score)

        return rewards