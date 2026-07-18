"""
grpo_trainer.py

Core implementation of Group Relative Policy Optimization (GRPO)
for ThinkLM-Lite.
"""

from typing import List

import torch
from torch.optim import AdamW

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from src.training.reward_model import RewardModel


class GRPOTrainer:
    """
    Implements Group Relative Policy Optimization (GRPO)
    for ThinkLM.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        beta: float = 0.05,
        lr: float = 1e-5,
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.beta = beta
        self.lr = lr

        self.reward_model = RewardModel()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
        )

        # Device where the model is loaded
        self.device = next(self.model.parameters()).device

        print("=" * 60)
        print(" GRPO Trainer Initialized")
        print("=" * 60)
        print(f"Device : {self.device}")
        print(f"KL Beta: {self.beta}")
        print(f"LR     : {self.lr}")
        print("=" * 60)

    def generate_responses(
        self,
        prompt: str,
        num_responses: int = 8,
        max_new_tokens: int = 20,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> List[str]:
        """
        Generate multiple candidate responses for a prompt.
        """

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        input_length = inputs["input_ids"].shape[1]

        responses = []

        print(f"\nGenerating {num_responses} responses...\n")

        for i in range(num_responses):

            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            # Remove the prompt tokens
            generated_tokens = outputs[0][input_length:]

            response = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            ).strip()

            responses.append(response)

            print(f"[{i + 1}] {response}")

        return responses
    def compute_log_probs(
        self,
        prompt: str,
        responses: List[str]
    ) -> torch.Tensor:
        """
        Compute the average log probability of each generated response.

        Returns:
            Tensor of shape (num_responses,)
        """

        log_probs = []

        for response in responses:

            full_text = prompt + "\n" + response

            inputs = self.tokenizer(
                full_text,
                return_tensors="pt"
            ).to(self.device)

            # DO NOT use torch.no_grad() here
            outputs = self.model(**inputs)

            logits = outputs.logits

            shift_logits = logits[:, :-1, :]

            shift_labels = inputs["input_ids"][:, 1:]

            log_softmax = torch.nn.functional.log_softmax(
                shift_logits,
                dim=-1
            )

            token_log_probs = log_softmax.gather(
                2,
                shift_labels.unsqueeze(-1)
            ).squeeze(-1)

            average_log_prob = token_log_probs.mean()

            log_probs.append(average_log_prob)

        return torch.stack(log_probs)
        
    def compute_rewards(
            self,
            responses: List[str]
        ) -> torch.Tensor:
            """
            Compute reward scores for generated responses.

            Returns:
                Tensor of rewards on the model device.
            """

            rewards = self.reward_model.score(responses)

            rewards = torch.tensor(
                rewards,
                dtype=torch.float32,
                device=self.device
            )

            return rewards    
    def compute_advantages(
        self,
        rewards: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute normalized group-relative advantages.

        A_i = (R_i - mean(R)) / std(R)
        """

        group_mean = rewards.mean()

        group_std = rewards.std(unbiased=False)

        if group_std.item() < 1e-8:
            return torch.zeros_like(rewards)

        advantages = (
            rewards - group_mean
        ) / group_std

        return advantages
    def compute_policy_loss(
        self,
        advantages: torch.Tensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the GRPO policy loss.

        L = -(A * ratio).mean() + beta * KL
        """

        # Policy ratio
        ratio = torch.exp(
            log_probs - old_log_probs
        )

        # Policy gradient objective
        policy_loss = -(
            advantages * ratio
        ).mean()

        # KL divergence penalty
        kl = (
            old_log_probs - log_probs
        ).mean()

        # Total GRPO loss
        total_loss = (
            policy_loss +
            self.beta * kl
        )

        return total_loss
    def train_step(
        self,
        prompt: str,
        num_responses: int = 8,
    ) -> dict:
        """
        Perform one complete GRPO training step.
        """

        # Step 1: Generate responses
        responses = self.generate_responses(
            prompt=prompt,
            num_responses=num_responses,
        )

        # Step 2: Compute log probabilities
        log_probs = self.compute_log_probs(
            prompt,
            responses
        )

        # Step 3: Save reference policy
        old_log_probs = log_probs.detach().clone()

        # Step 4: Compute rewards
        rewards = self.compute_rewards(
            responses
        )

        # Step 5: Compute advantages
        advantages = self.compute_advantages(
            rewards
        )

        # Step 6: Compute loss
        loss = self.compute_policy_loss(
            advantages,
            log_probs,
            old_log_probs
        )

        # Step 7: Optimize
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return {
            "loss": loss.item(),
            "responses": responses,
            "rewards": rewards.detach().cpu().tolist(),
            "advantages": advantages.detach().cpu().tolist(),
            "log_probs": log_probs.detach().cpu().tolist(),
        }