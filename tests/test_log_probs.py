from src.training.model_loader import load_thinklm_model
from src.training.grpo_trainer import GRPOTrainer

model, tokenizer = load_thinklm_model()

trainer = GRPOTrainer(model, tokenizer)

prompt = "Explain AI in one sentence."

responses = trainer.generate_responses(
    prompt,
    num_responses=3
)

log_probs = trainer.compute_log_probs(
    prompt,
    responses
)

print("\nResponses:\n")

for r in responses:
    print(r)
    print()

print("Log probabilities:\n")
print(log_probs)