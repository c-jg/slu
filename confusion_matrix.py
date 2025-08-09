import argparse
import os

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import FSCDataset
from model import SLUTransformer
from utils import build_label_map, pad_collate


def generate_confusion_matrix(checkpoint_path, dataset_base_dir,output_path, batch_size=64, num_workers=6, show_image=True):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print(f"Created output folder: {output_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_csv = os.path.join(dataset_base_dir, "data", "train_data.csv")
    test_csv = os.path.join(dataset_base_dir, "data", "test_data.csv")

    train_df = pd.read_csv(train_csv)
    label_map = build_label_map(train_df)

    test_set = FSCDataset(test_csv, label_map, dataset_base_dir)
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=pad_collate,
        num_workers=num_workers
    )

    num_classes = len(label_map)
    model = SLUTransformer(num_classes=num_classes).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    all_predictions = []
    all_true_labels = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            _, preds = torch.max(logits, dim=1)

            all_predictions.extend(preds.detach().cpu().numpy())
            all_true_labels.extend(targets.detach().cpu().numpy())

    id_to_label = {v: k for k, v in label_map.items()}
    class_names = [id_to_label[i] for i in range(len(id_to_label))]

    cm = confusion_matrix(all_true_labels, all_predictions)

    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d", ax=plt.gca(), xticks_rotation=45)
    plt.title("Confusion Matrix - Test Set", fontsize=16, pad=20)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    if show_image:
        plt.show()

    # Compute and print test accuracy
    test_accuracy = (cm.trace() / cm.sum()) * 100
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    plt.savefig(os.path.join(output_path, "test_confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a confusion matrix on the test set from a trained checkpoint.")

    parser.add_argument("--checkpoint", required=True, help="Path to the trained checkpoint (.pth)")
    parser.add_argument("--dataset_base_dir", required=True, help="Base directory of the FSC dataset")
    parser.add_argument("--output_path", required=True, help="Path to folder where confusion matrix image will be saved")
    parser.add_argument("--show_image", default=True, help="Image will not be displayed in GUI if set to False")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=6)

    args = parser.parse_args()

    saved_path = generate_confusion_matrix(
        checkpoint_path=args.checkpoint,
        dataset_base_dir=args.dataset_base_dir,
        output_path=args.output_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        show_image=args.show_image
    )

    print(f"Confusion matrix saved to: {saved_path}")
