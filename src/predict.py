"""
Predict Pokemon type from a card image.

Usage:
    python src/predict.py path/to/card.jpg
    python src/predict.py path/to/card.jpg --model my_model.pt
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

from dataset import IDX_TO_TYPE, NUM_CLASSES, get_val_transform
from model import PokemonTypeCNN


def predict(image_path: str, model_path: str, top_k: int = 3):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = PokemonTypeCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert("RGB")
    transform = get_val_transform()
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
    parser.add_argument("--model", default="best_model.pt", help="Path to model checkpoint")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / args.model
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Train first: python src/train.py")
        return

    predict(args.image, str(model_path), args.top_k)


if __name__ == "__main__":
    main()
