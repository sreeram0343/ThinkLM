from src.training.model_loader import load_thinklm_model
from src.training.stable_gate import STABLEGate

print("Loading model...")

model, tokenizer = load_thinklm_model()

gate = STABLEGate(model, tokenizer)

result = gate.gate_merge_with_clip(
    baseline_em=0.92,
    candidate_em=0.81,
)

print("\nReturned Result")
print("=" * 50)

print(result)