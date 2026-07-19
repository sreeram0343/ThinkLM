from src.training.model_loader import load_thinklm_model
from src.training.stable_gate import STABLEGate

model, tokenizer = load_thinklm_model()

gate = STABLEGate(
    model,
    tokenizer
)

anchors = gate.load_anchor_set(
    "data/squad_anchor.json"
)

print("\nFirst Anchor:\n")
print(anchors[0])