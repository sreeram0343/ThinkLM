from src.training.model_loader import load_thinklm_model
from src.training.grpo_trainer import GRPOTrainer

print("Loading model...")

model, tokenizer = load_thinklm_model(
    "Qwen/Qwen2.5-1.5B-Instruct"
)

trainer = GRPOTrainer(
    model,
    tokenizer
)

print("\nTrainer created successfully.")