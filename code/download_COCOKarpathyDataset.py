from datasets import load_dataset
from pathlib import Path
import requests
from PIL import Image
from io import BytesIO
import argparse
from paths import DATASET_PATH


def download_split(split):
    print(f"Downloading images from the COCO-Karpathy {split} split...")
    dataset = load_dataset("yerevann/coco-karpathy")[split]
    image_dir = Path(DATASET_PATH) / "images" / split
    image_dir.mkdir(parents=True, exist_ok=True)

    total = len(dataset)
    for i, sample in enumerate(dataset):
        image_id = sample["cocoid"]
        url = sample["url"]
        image_path = image_dir / f"{image_id:012d}.jpg"

        if i % 100 == 0:
            print(f"Processing {i}/{total} images...")

        if not image_path.exists():
            try:
                response = requests.get(url)
                image = Image.open(BytesIO(response.content)).convert("RGB")
                image.save(image_path)
                print(f"Downloaded: {image_path.name}")
            except Exception as e:
                print(f"Error downloading {url}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Download COCO-Karpathy dataset")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test", "restval", "all"],
        help="Dataset split to download (train, validation, test, or all)",
    )
    args = parser.parse_args()

    if args.split == "all":
        for split in ["train", "validation", "test", "restval"]:
            download_split(split)
    else:
        download_split(args.split)


if __name__ == "__main__":
    main()
