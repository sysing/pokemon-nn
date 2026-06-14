"""
Predict Pokemon type from a card image.
Auto-detects model input channels (3 for v0.1/v0.2, 4 for v0.3/v0.4).

Usage:
    python src/predict.py path/to/card.jpg
    python src/predict.py path/to/card.jpg --model runs/v0.4/model.pt
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from dataset import IDX_TO_TYPE, NUM_CLASSES
from model import PokemonTypeCNN


def _stack_gray(tensor):
    gray = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
    return torch.cat([tensor, gray.unsqueeze(0)], dim=0)


def detect_input_channels(ckpt_path: str) -> int:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    w = ckpt["features.0.weight"]
    return w.shape[1]  # 3 or 4


def get_transform(channels: int):
    base = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ]
    if channels == 4:
        base.append(transforms.Lambda(_stack_gray))
        base.append(transforms.Normalize(mean=[0, 0, 0, 0.5],
                                         std=[1, 1, 1, 0.225]))
    return transforms.Compose(base)


def predict(image_path: str, model_path: str, top_k: int = 3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    channels = detect_input_channels(model_path)
    print(f"Model input: {channels} channels")

    model = PokemonTypeCNN(num_classes=NUM_CLASSES, in_channels=channels)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    transform = get_transform(channels)
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).squeeze()

    top_probs, top_indices = probs.topk(top_k)
    print(f"\nImage: {image_path}")
    for i in range(len(top_indices)):
        idx = top_indices[i].item()
        prob = top_probs[i].item()
        print(f"  {IDX_TO_TYPE[idx]:12s}: {prob:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Predict Pokemon type")
    parser.add_argument("image", help="Path to card image")
    parser.add_argument("--model", default="runs/v0.4/model.pt", help="Path to model checkpoint")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / args.model
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Download from: https://github.com/sysing/pokemon-nn/releases")
        return

    predict(args.image, str(model_path), args.top_k)


if __name__ == "__main__":
    main()
