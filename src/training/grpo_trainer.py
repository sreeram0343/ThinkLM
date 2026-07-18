import torch
from typing import List


class GRPOTrainer:
    """
    Implements the mathematical core of
    Group Relative Policy Optimization (GRPO).
    """

    def __init__(self, beta: float = 0.05):
        self.beta = beta

    def compute_advantages(self, rewards: List[float]) -> torch.Tensor:
        """
        Compute normalized group-relative advantages.

        A_i = (R_i - mean(R)) / std(R)

        Args:
            rewards: List of reward values

        Returns:
            Tensor of normalized advantages
        """

        rewards = torch.tensor(rewards, dtype=torch.float32)

        group_mean = rewards.mean()
        group_std = rewards.std(unbiased=False)

        if group_std == 0:
            return torch.zeros_like(rewards)

        advantages = (rewards - group_mean) / group_std

        return advantages

    def compute_policy_loss(
        self,
        advantages: torch.Tensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Simplified GRPO loss.

        L = -(A * ratio).mean() + beta * KL

        """

        ratio = torch.exp(log_probs - old_log_probs)

        policy_loss = -(advantages * ratio).mean()

        kl = (old_log_probs - log_probs).mean()

        total_loss = policy_loss + self.beta * kl

        return total_loss