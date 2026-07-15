import torch

from src.training.model_loader import load_thinklm_model


print("=" * 50)
print("Loading ThinkLM Model...")
print("=" * 50)

model, tokenizer = load_thinklm_model()

print("\nModel Loaded Successfully!\n")

print("Tokenizer:", tokenizer.name_or_path)

if torch.cuda.is_available():

    print("\nGPU:", torch.cuda.get_device_name(0))

    print(
        "Allocated VRAM:",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
        "GB",
    )

    print(
        "Reserved VRAM:",
        round(torch.cuda.memory_reserved() / 1024**3, 2),
        "GB",
    )

else:

    print("\nCUDA not available.")
    print("Running on CPU.")