import argparse
import json

import torch
import pandas as pd

from model import SLUTransformer
from utils import load_fbank, build_label_map


def inference(wav, model, train_csv):
    print("Loading train CSV and label map...")
    train_df = pd.read_csv(train_csv)
    label_map = build_label_map(train_df)
    id2label = {v: k for k, v in label_map.items()}

    # Load model
    print("Loading model...")
    num_classes = len(label_map)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SLUTransformer(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model, map_location=device))
    model.eval()

    # Feature extraction
    print("Extracting features...")
    features = load_fbank(wav)  # (T, 320)
    features = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, 320)

    # Inference
    with torch.no_grad():
        logits = model(features)
        pred_id = torch.argmax(logits, dim=1).item()
        pred_label = id2label[pred_id]
        probabilities = torch.softmax(logits, dim=1)
        confidence_score = probabilities[0, pred_id].item()

    action, object, location = pred_label.split("-")

    response_data = {
        "action": action,
        "object": object,
        "location": location
    }

    print("\n===== INFERENCE RESULT =====")
    print(f"WAV file: {wav}")
    print(f"Prediction:\n{json.dumps(response_data, indent=4)}")
    print(f"Confidence score: {confidence_score:.4f}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLU Transformer Inference")

    parser.add_argument('--wav', type=str, required=True, help='Path to wav file')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to train_data.csv to get labels')

    args = parser.parse_args()

    inference(
        wav=args.wav,
        model=args.model,
        train_csv=args.train_csv
    )
