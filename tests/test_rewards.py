from src.training.model_loader import load_thinklm_model
from src.training.grpo_trainer import GRPOTrainer

print("Loading model...")

model, tokenizer = load_thinklm_model()

trainer = GRPOTrainer(model, tokenizer)

responses = trainer.generate_responses(
    prompt="Explain Artificial Intelligence.",
    num_responses=3,
    max_new_tokens=20
)

rewards = trainer.compute_rewards(responses)

print("\nResponses:\n")

for i, response in enumerate(responses, start=1):
    print(f"{i}. {response}\n")

print("Reward Tensor:")
print(rewards)