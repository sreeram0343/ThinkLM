import torch

from src.training.model_loader import load_thinklm_model
from src.training.grpo_trainer import GRPOTrainer

model, tokenizer = load_thinklm_model()

trainer = GRPOTrainer(model, tokenizer)

advantages = torch.tensor(
    [1.2, -0.6, -0.6],
    device=trainer.device
)

log_probs = torch.tensor(
    [-1.4, -1.8, -1.6],
    device=trainer.device,
    requires_grad=True
)

old_log_probs = log_probs.detach().clone()

loss = trainer.compute_policy_loss(
    advantages,
    log_probs,
    old_log_probs
)

print("\nLoss:")
print(loss)