"""
Collect Pokemon card images and labels from pokemontcg.io API.

Usage:
    python src/collect.py [--data-dir data]

Outputs:
    data/images/        — downloaded card images (named by card ID)
    data/labels.csv     — columns: id,name,type,image_path
"""

import argparse
import csv
import time
from pathlib import Path

import requests
from tqdm import tqdm

API_BASE = "https://api.pokemontcg.io/v2/cards"
PAGE_SIZE = 250
TYPES = [
    "Colorless", "Darkness", "Dragon",
    "Fighting", "Fire", "Grass", "Lightning",
    "Metal", "Psychic", "Water",
]
MAX_PER_TYPE = 100
REQUEST_TIMEOUT = 60


def fetch_with_retry(url, params=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            tqdm.write(f"  Retry {attempt+1}/{max_retries}: {e}")
            time.sleep(2 ** attempt)


def collect(data_dir: Path):
    image_dir = data_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "labels.csv"

    per_type = {t: 0 for t in TYPES}
    all_full = all(per_type[t] >= MAX_PER_TYPE for t in TYPES)
    page = 1
    records = []

    pbar = tqdm(desc="Collecting cards", unit="pages")
    while not all_full:
        resp = fetch_with_retry(API_BASE, params={
            "q": "supertype:pokemon",
            "pageSize": PAGE_SIZE,
            "page": page,
            "select": "id,name,types,images",
        })
        data = resp.json()
        batch = data.get("data", [])
        if not batch:
            break

        for card in batch:
            types = card.get("types", [])
            if not types or types[0] not in TYPES:
                continue
            ptype = types[0]
            if per_type[ptype] >= MAX_PER_TYPE:
                continue

            card_id = card["id"]
            image_url = card.get("images", {}).get("large") or card.get("images", {}).get("small")
            if not image_url:
                continue

            ext = ".png" if ".png" in image_url else ".jpg"
            filename = f"{card_id}{ext}"
            filepath = image_dir / filename

            if not filepath.exists():
                try:
                    img_resp = fetch_with_retry(image_url)
                    filepath.write_bytes(img_resp.content)
                except Exception as e:
                    tqdm.write(f"  Image failed {card_id}: {e}")
                    continue

            records.append((card_id, card["name"], ptype, filename))
            per_type[ptype] += 1

        all_full = all(per_type[t] >= MAX_PER_TYPE for t in TYPES)
        pbar.set_postfix(per_type)
        pbar.update(1)
        page += 1
        time.sleep(0.15)

    pbar.close()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "name", "type", "image_path"])
        writer.writerows(records)

    print("\nType distribution:")
    for t in TYPES:
        print(f"  {t:12s}: {per_type[t]}")
    print(f"  Total: {sum(per_type.values())}")


def main():
    parser = argparse.ArgumentParser(description="Collect Pokemon card data")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / args.data_dir

    print(f"Target: {MAX_PER_TYPE} per type, data dir: {data_dir}")
    collect(data_dir)
    print(f"\nSaved to {data_dir / 'labels.csv'}")


if __name__ == "__main__":
    main()
