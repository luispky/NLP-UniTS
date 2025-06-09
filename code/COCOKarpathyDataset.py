from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Dict, Any, Optional


class COCOKarpathyDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "test",
        start_idx: int = 0,
        max_samples: Optional[int] = None,
        return_images: bool = True,
    ):
        """
        Initialize the COCO Karpathy dataset with pre-downloaded images.

        Args:
            split: 'train', 'val', or 'test'
            start_idx: Starting index for sample selection (optional)
            max_samples: Limit the number of samples (optional)
            data_dir: Base directory with images and annotations
            return_images: Whether to return images (False for captions/IDs only)
        """
        dataset = load_dataset("yerevann/coco-karpathy")[split]
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "images" / split
        self.return_images = return_images

        if max_samples and start_idx + max_samples <= len(dataset):
            self.dataset = dataset.select(range(start_idx, start_idx + max_samples))  # [start_idx, start_idx + max_samples)
        elif start_idx < len(dataset):
            self.dataset = dataset.select(range(start_idx, len(dataset)))
        else:
            raise ValueError(
                f"start_idx {start_idx} is out of range for dataset of length {len(dataset)}"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.dataset[idx]
        caption = max(sample["sentences"], key=len)  # Longest caption
        image_id = sample["cocoid"]

        result = {"caption": caption, "image_id": image_id}

        if self.return_images:
            image_path = self.image_dir / f"{image_id:012d}.jpg"
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Image {image_path} not found. Ensure dataset is pre-downloaded."
                )
            image = Image.open(image_path).convert("RGB")
            result["image"] = image

        return result


def get_coco_karpathy_dataloader(
    data_dir: str,
    split: str = "test",
    start_idx: int = 0,
    max_samples: Optional[int] = None,
    batch_size: int = 2,
    num_workers: int = 4,
    return_images: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for the COCO Karpathy dataset.

    Args:
        split: Dataset split
        start_idx: Starting index for sample selection
        max_samples: Max samples to use
        batch_size: Number of samples per batch
        num_workers: Number of worker processes for loading
        data_dir: Directory with pre-downloaded data
        return_images: Whether to return images in the dataset
    """
    dataset = COCOKarpathyDataset(
        split=split,
        start_idx=start_idx,
        max_samples=max_samples,
        data_dir=data_dir,
        return_images=return_images,
    )

    def coco_collate_fn(batch):
        captions = [item["caption"] for item in batch]
        image_ids = [item["image_id"] for item in batch]
        images = [item["image"] for item in batch] if "image" in batch[0] else None
        result = {
            "caption": captions,
            "image_id": image_ids,
        }
        if images is not None:
            result["image"] = images
        return result

    # Single dataloader configuration
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=coco_collate_fn,
    )
