from src.utils.anchor_dataset import AnchorDataset

dataset = AnchorDataset("data/squad_anchor.json")

print("Total samples:", len(dataset))

print("\nFirst sample:")
print(dataset.get(0))

print("\nFirst batch size:")
for batch in dataset.batch(8):
    print(len(batch))
    break