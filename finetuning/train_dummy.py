"""Dummy training script for testing (CPU-only, no real model training)."""

import json
import os


def load_dummy_data(
    data_file: str = "finetuning/data/samples/dummy_instructions.jsonl",
):
    """Load dummy training data."""
    if not os.path.exists(data_file):
        print(f"⚠️  Data file not found: {data_file}")
        return []

    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                samples.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    return samples


def train_dummy(num_epochs: int = 3):
    """Dummy training loop (no actual model updates)."""
    print("=== Dummy Training (CPU-only, no real updates) ===\n")

    data = load_dummy_data()
    if not data:
        print("⚠️  No training data loaded")
        return

    print(f"Loaded {len(data)} training samples")
    print(f"Epochs: {num_epochs}")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        for i, sample in enumerate(data):
            # Dummy loss calculation (no real training)
            dummy_loss = 1.0 / (1 + i + epoch)
            epoch_loss += dummy_loss
            print(f"  Sample {i + 1}/{len(data)}: loss={dummy_loss:.4f}")

        avg_loss = epoch_loss / len(data) if data else 0
        print(f"  Epoch average loss: {avg_loss:.4f}")

    print("\n✅ Dummy training completed successfully")
    print("📁 Model would be saved to: finetuning/models/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dummy training script")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--data",
        type=str,
        default="finetuning/data/samples/dummy_instructions.jsonl",
        help="Data file path",
    )
    args = parser.parse_args()

    train_dummy(num_epochs=args.epochs)
