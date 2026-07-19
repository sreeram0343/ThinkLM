from src.training.model_loader import load_thinklm_model
from src.training.stable_gate import STABLEGate

print("Loading model...")

model, tokenizer = load_thinklm_model()

gate = STABLEGate(
    model,
    tokenizer
)

scale = gate.binary_search_clip(
    baseline_em=0.92,
    candidate_em=0.81,
)

print("\nReturned Scale:", scale)