from src.training.model_loader import load_thinklm_model
from src.training.grpo_trainer import GRPOTrainer

model, tokenizer = load_thinklm_model(
    "Qwen/Qwen2.5-1.5B-Instruct"
)

trainer = GRPOTrainer(model, tokenizer)

result = trainer.train_step(
    "What is Artificial Intelligence?"
)

print(result["loss"])

print(result["rewards"])

print(result["responses"][0][:200])
