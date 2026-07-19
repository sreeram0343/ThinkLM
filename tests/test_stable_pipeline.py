from src.training.model_loader import load_thinklm_model
from src.training.stable_gate import STABLEGate

print("Loading model...")

model, tokenizer = load_thinklm_model()

gate = STABLEGate(
    model,
    tokenizer
)

anchors = gate.load_anchor_set(
    "data/squad_anchor.json"
)

print(f"\nLoaded {len(anchors)} anchors.")

# Simulated EM scores
baseline_em = 0.92
candidate_em = 0.81

result = gate.gate_merge_with_clip(
    baseline_em,
    candidate_em
)

print("\n" + "=" * 60)
print("FINAL STABLE DECISION")
print("=" * 60)

print("Accepted :", result["accepted"])
print("Scale    :", result["scale"])
print("EM Drop  :", result["em_drop"])

print("=" * 60)