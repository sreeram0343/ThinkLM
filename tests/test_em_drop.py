from src.training.model_loader import load_thinklm_model
from src.training.stable_gate import STABLEGate

print("Loading model...")

model, tokenizer = load_thinklm_model()

gate = STABLEGate(
    model,
    tokenizer
)

# Example values
baseline = 0.92
candidate = 0.81

drop = gate.compute_em_drop(
    baseline,
    candidate
)

print("\nReturned Drop:", drop)