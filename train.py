import argparse
import time
import os

import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

from dataset import FSCDataset
from model import SLUTransformer
from utils import build_label_map, pad_collate


def train(output_path, dataset_base_dir, epochs, lr, batch_size, num_workers):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        print(f"Created output folder: {output_path}")

    train_csv = os.path.join(dataset_base_dir, "data", "train_data.csv")
    val_csv = os.path.join(dataset_base_dir, "data", "valid_data.csv")
    test_csv = os.path.join(dataset_base_dir, "data", "test_data.csv")

    train_df = pd.read_csv(train_csv)
    label_map = build_label_map(train_df)

    train_set = FSCDataset(train_csv, label_map, dataset_base_dir)
    val_set = FSCDataset(val_csv, label_map, dataset_base_dir)
    test_set = FSCDataset(test_csv, label_map, dataset_base_dir)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=pad_collate, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=pad_collate, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=pad_collate, num_workers=num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    num_classes = len(label_map)
    print(f"Number of classes: {num_classes}")

    model = SLUTransformer(num_classes=num_classes).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of model parameters: {num_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Lists to store training and validation metrics
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        epoch_start_time = time.time()
        print(f"Epoch {epoch+1}\n-------------------------------")
        size = len(train_loader.dataset)
        model.train()
        for batch, (X, y) in enumerate(train_loader):
            # Move data to GPU
            X = X.to(device)
            y = y.to(device)

            # Compute prediction and loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # Store training loss
        train_losses.append(loss.item())
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val = X_val.to(device)
                y_val = y_val.to(device)
                val_pred = model(X_val)
                val_batch_loss = loss_fn(val_pred, y_val)
                val_loss += val_batch_loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        minutes = int(epoch_duration // 60)
        seconds = int(epoch_duration % 60)
        print(f"Epoch {epoch+1} took {minutes}:{seconds:02d}")

        torch.save(model.state_dict(), os.path.join(output_path, f"{epoch}_SLU_Transformer.pth"))
        print("Saved checkpoint.")

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'training_plots.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Training completed! Plots saved as 'training_plots.png'")
    
    # Test set evaluation
    print("\n" + "="*50)
    print("TEST SET EVALUATION")
    print("="*50)
    
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for X_test, y_test in test_loader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            test_pred = model(X_test)
            _, predicted = torch.max(test_pred.data, 1)
            test_total += y_test.size(0)
            test_correct += (predicted == y_test).sum().item()
            
            # Store predictions and true labels for confusion matrix
            all_predictions.extend(predicted.cpu().numpy())
            all_true_labels.extend(y_test.cpu().numpy())
    
    test_accuracy = 100 * test_correct / test_total
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Get class names
    class_names = list(label_map.keys())
    
    # Create confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
    plt.title('Confusion Matrix - Test Set', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Confusion matrix saved as 'confusion_matrix.png'")
    print("="*50)

    # Print best epochs
    best_val_loss_epoch = int(np.argmin(val_losses)) + 1
    best_val_loss_value = val_losses[best_val_loss_epoch - 1]
    best_train_loss_epoch = int(np.argmin(train_losses)) + 1
    best_train_loss_value = train_losses[best_train_loss_epoch - 1]
    print(f"\nBest Validation Loss: {best_val_loss_value:.4f} at epoch {best_val_loss_epoch}")
    print(f"Lowest Training Loss: {best_train_loss_value:.4f} at epoch {best_train_loss_epoch}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train SLUTransformer on Fluent Speech Commands (FSC) Dataset")

    parser.add_argument("--output_path", type=str, required=True, help="Path to save outputs and checkpoints")
    parser.add_argument("--dataset_base_dir", type=str, required=True, help="Base directory of the FSC dataset")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training and evaluation")
    parser.add_argument("--num_workers", type=int, default=6, help="Number of DataLoader worker processes")

    args = parser.parse_args()

    train(
        output_path=args.output_path, 
        dataset_base_dir=args.dataset_base_dir, 
        epochs=args.epochs, 
        lr=args.lr, 
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
