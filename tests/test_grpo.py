import torch
from src.training.grpo_trainer import GRPOTrainer

trainer = GRPOTrainer.__new__(GRPOTrainer)

rewards = [1, 2, 3, 4, 5, 6, 7, 8]

advantages = trainer.compute_advantages(rewards)

print("Advantages:", advantages)
print("Mean:", advantages.mean().item())
print("Std:", advantages.std(unbiased=False).item())