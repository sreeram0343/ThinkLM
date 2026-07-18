from src.training.model_loader import load_thinklm_model
from src.training.grpo_trainer import GRPOTrainer

print("Loading model...")

model, tokenizer = load_thinklm_model()

trainer = GRPOTrainer(model, tokenizer)

responses = trainer.generate_responses(
    "Explain Artificial Intelligence.",
    num_responses=3,
    max_new_tokens=20
)

rewards = trainer.compute_rewards(responses)

advantages = trainer.compute_advantages(rewards)

print("\nRewards:")
print(rewards)

print("\nAdvantages:")
print(advantages)

print("\nMean:", advantages.mean().item())

print("Std :", advantages.std(unbiased=False).item())