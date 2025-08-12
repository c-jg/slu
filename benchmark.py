import argparse

import torch
from torch.utils.benchmark import Timer

from model import SLUTransformer


def run_inference(model, features):
    with torch.no_grad():
        _ = model(features)
        torch.cuda.synchronize()


def benchmark(num_classes, model_path, num_runs):
    assert torch.cuda.is_available(), "CUDA not available."
    device = torch.device('cuda')
    model = SLUTransformer(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    torch.set_float32_matmul_precision('high')
    model.compile()

    features = torch.randn(1, 192, 320, device=device)  # 2-second signal

    timer = Timer(
        stmt='run_inference(model, features)',
        setup='from __main__ import run_inference',
        globals={
            'model': model,
            'features': features}
    )

    print("\n===== Benchmarking Model Inference =====")
    result = timer.timeit(num_runs)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLU Transformer Inference")

    parser.add_argument('--num_classes', type=int, default=31, help='Number of classes the model was trained on.')
    parser.add_argument('--model', type=str, default="EP_81_SLU_Transformer.pth", help='Path to model checkpoint (.pth)')
    parser.add_argument('--num_runs', type=int, default=1000, help='Number of times to run inference for timing measurement.')

    args = parser.parse_args()

    benchmark(
        num_classes=args.num_classes,
        model_path=args.model,
        num_runs=args.num_runs
    )
