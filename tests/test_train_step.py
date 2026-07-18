from src.training.model_loader import load_thinklm_model
from src.training.grpo_trainer import GRPOTrainer

print("Loading model...")

model, tokenizer = load_thinklm_model()

trainer = GRPOTrainer(model, tokenizer)

result = trainer.train_step(
    prompt="Explain Artificial Intelligence.",
    num_responses=3,
)

print("\nTraining Complete!\n")

print("Loss:")
print(result["loss"])

print("\nRewards:")
print(result["rewards"])

print("\nAdvantages:")
print(result["advantages"])

print("\nLog Probabilities:")
print(result["log_probs"])

print("\nFirst Response:")
print(result["responses"][0])