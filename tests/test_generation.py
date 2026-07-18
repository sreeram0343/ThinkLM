from src.training.model_loader import load_thinklm_model
from src.training.grpo_trainer import GRPOTrainer

print("Loading model...")

model, tokenizer = load_thinklm_model(
    "Qwen/Qwen2.5-1.5B-Instruct"
)

trainer = GRPOTrainer(model, tokenizer)

responses = trainer.generate_responses(
    prompt="Explain Artificial Intelligence in one sentence.",
    num_responses=3,
    max_new_tokens=30
)

print("\nReturned Responses:\n")

for i, response in enumerate(responses, start=1):
    print(f"{i}. {response}")