"""
PyTorch Dataset for Pokemon card type classification.
v0.3: 4-channel input [R, G, B, Gray] for stacked color + grayscale.
"""

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

TYPE_TO_IDX = {
    "Colorless": 0, "Darkness": 1, "Dragon": 2,
    "Fighting": 3, "Fire": 4, "Grass": 5, "Lightning": 6,
    "Metal": 7, "Psychic": 8, "Water": 9,
}
IDX_TO_TYPE = {v: k for k, v in TYPE_TO_IDX.items()}
NUM_CLASSES = 10


def _stack_gray(tensor):
    """Stack grayscale as 4th channel: [3, H, W] -> [4, H, W]."""
    gray = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
    return torch.cat([tensor, gray.unsqueeze(0)], dim=0)


def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Lambda(_stack_gray),
        transforms.Normalize(mean=[0, 0, 0, 0.5],
                             std=[1, 1, 1, 0.225]),
    ])


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.Lambda(_stack_gray),
        transforms.Normalize(mean=[0, 0, 0, 0.5],
                             std=[1, 1, 1, 0.225]),
    ])


class PokemonDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.samples = []

        labels_path = self.data_dir / "labels.csv"
        with open(labels_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_path = self.data_dir / "images" / row["image_path"]
                if img_path.exists():
                    self.samples.append((str(img_path), TYPE_TO_IDX[row["type"]]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataloaders(data_dir: str, batch_size: int = 32, val_split: float = 0.2):
    full_dataset = PokemonDataset(data_dir, transform=get_train_transform())

    label_counts = [0] * NUM_CLASSES
    for _, label in full_dataset.samples:
        label_counts[label] += 1

    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    val_ds.transform = get_val_transform()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    return train_loader, val_loader, label_counts
