import torch

from src.training.grpo_trainer import GRPOTrainer

trainer = GRPOTrainer()

rewards = [
    0.90,
    0.85,
    0.95,
    0.70,
    0.80,
    0.60,
    0.75,
    1.00
]

advantages = trainer.compute_advantages(rewards)

print("Advantages:\n")
print(advantages)

print("\nMean:", advantages.mean().item())
print("Std :", advantages.std(unbiased=False).item())

log_probs = torch.randn(8)
old_log_probs = torch.randn(8)

loss = trainer.compute_policy_loss(
    advantages,
    log_probs,
    old_log_probs
)

print("\nPolicy Loss:", loss.item())