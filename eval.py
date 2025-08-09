import argparse
import os
import re
import glob

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import FSCDataset
from model import SLUTransformer
from utils import build_label_map, pad_collate


def _extract_epoch_from_filename(filename):
    """Extract leading epoch integer from checkpoint filename like '12_SLU_Transformer.pth'.
    Falls back to -1 if not found so sorting still works.
    """
    base = os.path.basename(filename)
    match = re.match(r"^(\d+)", base)
    if match:
        return int(match.group(1))
    return -1


def _evaluate_checkpoint_accuracy(checkpoint_path, num_classes, test_loader, device):
    model = SLUTransformer(num_classes=num_classes).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            _, preds = torch.max(logits, dim=1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()
    return 100.0 * correct / total if total > 0 else 0.0


def evaluate_checkpoints_dir(checkpoints_dir, dataset_base_dir, output_path, batch_size=64, num_workers=6):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print(f"Created output folder: {output_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_csv = os.path.join(dataset_base_dir, "data", "train_data.csv")
    test_csv = os.path.join(dataset_base_dir, "data", "test_data.csv")

    train_df = pd.read_csv(train_csv)
    label_map = build_label_map(train_df)
    num_classes = len(label_map)

    test_set = FSCDataset(test_csv, label_map, dataset_base_dir)
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=num_workers
    )

    checkpoint_files = glob.glob(os.path.join(checkpoints_dir, "*.pth"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No .pth checkpoints found in {checkpoints_dir}")

    checkpoint_files.sort(key=_extract_epoch_from_filename)

    epochs = []
    accuracies = []

    print(f"Evaluating {len(checkpoint_files)} checkpoints from '{checkpoints_dir}'...")
    for ckpt in checkpoint_files:
        epoch = _extract_epoch_from_filename(ckpt)
        acc = _evaluate_checkpoint_accuracy(ckpt, num_classes, test_loader, device)
        epochs.append(epoch)
        accuracies.append(acc)
        print(f"Epoch {epoch:>4}: {acc:.2f}%  ({os.path.basename(ckpt)})")

    # Plot accuracy vs epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, marker="o", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Checkpoint Accuracy on Test Set")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = os.path.join(output_path, "checkpoint_accuracies.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.show()

    # Save CSV of results
    results_csv = os.path.join(output_path, "checkpoint_accuracies.csv")
    pd.DataFrame({"epoch": epochs, "accuracy": accuracies}).to_csv(results_csv, index=False)

    best_idx = int(np.argmax(accuracies))
    best_epoch = epochs[best_idx]
    best_acc = accuracies[best_idx]
    print(f"\nBest accuracy: {best_acc:.2f}% at epoch {best_epoch}")

    print(f"Saved plot to: {fig_path}")
    print(f"Saved CSV to:  {results_csv}")
    return fig_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints in a directory on the test set and plot accuracy vs epoch.")

    parser.add_argument("--checkpoints_dir", required=True, help="Directory containing .pth checkpoints")
    parser.add_argument("--dataset_base_dir", required=True, help="Base directory of the FSC dataset")
    parser.add_argument("--output_path", required=True, help="Directory to save the accuracy plot and results CSV")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=6)

    args = parser.parse_args()

    evaluate_checkpoints_dir(
        checkpoints_dir=args.checkpoints_dir,
        dataset_base_dir=args.dataset_base_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
