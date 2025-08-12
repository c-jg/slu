import time

import gradio as gr
import torch
import pandas as pd
import numpy as np

from model import SLUTransformer
from utils import load_fbank, build_label_map


class SLUInference:
    def __init__(self, model_path, train_csv):
        self.model_path = model_path
        self.train_csv = train_csv
        assert torch.cuda.is_available(), "CUDA not available."
        self.device = torch.device('cuda')

        # Load label map
        train_df = pd.read_csv(train_csv)
        label_map = build_label_map(train_df)
        self.id2label = {v: k for k, v in label_map.items()}
        num_classes = len(label_map)

        # Load and compile model
        self.model = SLUTransformer(num_classes=num_classes).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()

        # Warm up
        self.warm_up()

    def warm_up(self):
        dummy_input = torch.randn(1, 192, 320, device=self.device)
        with torch.no_grad():
            self.model(dummy_input)
            torch.cuda.synchronize()

    def predict(self, audio):
        sample_rate, wav = audio

        wav = wav.astype(np.float32) / 32768.0
        features = load_fbank(torch.from_numpy(wav), sample_rate, is_waveform=True)  # (T, 320)
        features = torch.as_tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)  # (1, T, 320)

        # Inference
        with torch.no_grad():
            logits = self.model(features)
            torch.cuda.synchronize()

        pred_id = torch.argmax(logits, dim=1).item()
        pred_label = self.id2label[pred_id]
        confidence = torch.softmax(logits, dim=1)[0, pred_id].item()

        action, obj, loc = pred_label.split("-")

        result = {
            "Action": action,
            "Object": obj,
            "Location": loc,
            "Confidence": f"{confidence:.4f}"
        }

        return result


MODEL_PATH = "EP_81_SLU_Transformer.pth"
TRAIN_CSV = "fluent_speech_commands_dataset/data/train_data.csv"

slu = SLUInference(model_path=MODEL_PATH, train_csv=TRAIN_CSV)

with gr.Blocks(title="🎙️ End-to-End Spoken Language Understanding") as demo:
    gr.Markdown("# 🎙️ End-to-End Spoken Language Understanding")
    gr.Markdown(
        "Upload a `.wav` file or record a voice command (max 5 seconds). "
        "The model maps speech directly to intent: **action-object-location**."
    )

    audio_input = gr.Audio(
        label="🎤 Record or Upload Audio",
        type="numpy",
        sources=["upload", "microphone"],
        max_length=5,
        waveform_options=gr.WaveformOptions(
            show_recording_waveform=True
        ),
        interactive=True
    )

    with gr.Row():
        btn = gr.Button("🔍 Classify Intent", variant="primary")

    with gr.Row():
        output_json = gr.JSON(label="Predicted Intent")

    # Run inference
    btn.click(fn=slu.predict, inputs=audio_input, outputs=[output_json])

    # Examples
    gr.Examples(
        examples=[
            ["examples/i-cant-hear-that_increase-volume-none.wav"],
            ["examples/turn-the-lights-on-in-the-kitchen_activate-lights-kitchen.wav"],
            ["examples/turn-the-lights-on_activate-lights-none.wav"]
        ],
        inputs=audio_input,
        outputs=[output_json],
        fn=slu.predict,
        cache_examples=False
    )


if __name__ == "__main__":
    demo.launch(share=False)  # Set share=True for public link
