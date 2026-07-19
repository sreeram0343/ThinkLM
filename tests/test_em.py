from src.training.model_loader import load_thinklm_model
from src.training.stable_gate import STABLEGate

print("Loading model...")

model, tokenizer = load_thinklm_model()

gate = STABLEGate(
    model,
    tokenizer
)

em = gate.evaluate_em(
    "data/squad_anchor.json",
    max_samples=5
)

print("\nFinal EM:", em)