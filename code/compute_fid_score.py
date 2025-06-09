"""
Fréchet Inception Distance (FID) Calculator

This script calculates the FID score between real images and generated images.
FID is a metric that measures the similarity between two datasets of images.
Lower FID values indicate better quality of generated images (more similar to real images).

This implementation uses torchmetrics.image.fid.FrechetInceptionDistance for calculation.

Usage:
    - To compute FID for a specific number of images:
      python compute_fid_score.py fid --num_images 1000 [--ft_model] [--size 512] [--guidance_scale 3.0]

    - To compute FID across different dataset sizes:
      python compute_fid_score.py curve [--ft_model] [--size 512] [--min_images 500] [--max_images 5000] [--step 500] [--guidance_scale 3.0]
"""

import argparse
import os
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from rich import print
from paths import DATASET_PATH
import csv


# Define a custom dataset class to load images
class ImageDataset(Dataset):
    def __init__(
        self, image_paths=None, image_dir=None, image_names=None, transform=None
    ):
        if image_paths is not None:
            # Use provided image paths
            self.image_paths = image_paths
            if image_names is not None:
                # Filter by image names if provided
                self.image_paths = [
                    path
                    for path in self.image_paths
                    if os.path.splitext(os.path.basename(path))[0] in image_names
                ]
        elif image_dir is not None:
            # Use all images in a directory
            self.image_paths = [
                os.path.join(image_dir, img)
                for img in os.listdir(image_dir)
                if img.endswith((".png", ".jpg"))
                and (image_names is None or os.path.splitext(img)[0] in image_names)
            ]  # os.path.splitext(img)[0] could be simplified to img
        else:
            raise ValueError("Either image_paths or image_dir must be provided")

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


def collect_real_images(base_path):
    """
    Collects image paths from test, validation, and train directories if they exist.

    Args:
        base_path: Base directory containing the split directories

    Returns:
        List of image paths
    """
    image_paths = []
    splits = ["test", "validation", "train", "restval"]

    for split in splits:
        split_path = os.path.join(base_path, split)
        if os.path.exists(split_path) and os.path.isdir(split_path):
            split_images = [
                os.path.join(split_path, img)
                for img in os.listdir(split_path)
                if img.endswith((".png", ".jpg"))
            ]
            image_paths.extend(split_images)
            print(f"Found {len(split_images)} images in {split} directory")

    return image_paths


def compute_fid(
    batch_size=256,
    ft_model=False,
    size=512,
    feature_dim=2048,
    num_images=None,
    random_seed=42,
    guidance_scale=None,
):
    """
    Computes the FID score between real and generated images using torchmetrics.

    Args:
        batch_size: Batch size for processing images
        ft_model: Whether to use the fine-tuned model
        size: Image size
        feature_dim: Feature dimension for FID calculation
        num_images: Number of random images to use (if None, use all)
        random_seed: Random seed for reproducible image selection
        guidance_scale: Guidance scale used for image generation

    Returns:
        Tuple of (fid_score, actual_num_images_used) or (None, None) if computation fails
    """
    # Set random seed for reproducibility
    if num_images is not None:
        random.seed(random_seed)
        print(f"Using random seed: {random_seed}")

    # Determine the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Base path for real images
    real_base_path = os.path.join(script_dir, DATASET_PATH, "images")

    # Collect real image paths from all available splits
    real_image_paths = collect_real_images(real_base_path)

    if not real_image_paths:
        print(f"No real images found in any split directory under: {real_base_path}")
        return None, None

    # Path to generated images
    path_generated = os.path.join(
        script_dir,
        f"../generated_images_{'ft' if ft_model else 'hf'}_{size}"
        + (f"_cfg{guidance_scale:.1f}" if guidance_scale else ""),
    )

    # Check if directory exists and contains images
    if not os.path.exists(path_generated):
        print(f"Directory for generated images does not exist: {path_generated}")
        return None, None
    if not os.listdir(path_generated):
        print(f"No images found in the generated images directory: {path_generated}")
        return None, None

    # Get the set of image names (without extensions) from the generated images directory, e.g. 000000000001.png -> 000000000001
    generated_image_names = {
        os.path.splitext(img)[0]
        for img in os.listdir(
            path_generated
        )  # List only the filenames in the directory
        if img.endswith((".png", ".jpg"))
    }

    # Track actual number of images used
    actual_num_images = len(generated_image_names)

    # If num_images is specified, randomly select that many image names
    if num_images is not None:
        # Adjust num_images if it exceeds available images
        if num_images > len(generated_image_names):
            print(
                f"WARNING: Requested {num_images} images but only {len(generated_image_names)} are available. Using all available images."
            )
            num_images = len(generated_image_names)

        # Only sample if we need fewer than all images
        if num_images < len(generated_image_names):
            selected_image_names = set(
                random.sample(
                    list(generated_image_names),
                    num_images,
                )
            )
            print(
                f"Randomly selected {len(selected_image_names)} images out of {len(generated_image_names)} available"
            )
            generated_image_names = selected_image_names

        actual_num_images = num_images

    # Define transforms - FID expects 3x299x299 uint8 images
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).type(torch.uint8)),
        ]
    )

    # Load datasets using the same image names for both real and generated datasets
    real_dataset = ImageDataset(
        image_paths=real_image_paths,
        image_names=generated_image_names,
        transform=transform,
    )
    generated_dataset = ImageDataset(
        image_dir=path_generated, image_names=generated_image_names, transform=transform
    )

    print(
        f"Using {len(real_dataset)} real images and {len(generated_dataset)} generated images for FID calculation"
    )

    # Verify that we have matching image counts, which should be the case if the same names were used
    if len(real_dataset) != len(generated_dataset):
        print(
            f"WARNING: Number of real images ({len(real_dataset)}) doesn't match number of generated images ({len(generated_dataset)})"
        )
        print("Some image names might not be found in both sets")
        # Update actual_num_images to the smaller of the two sets
        actual_num_images = min(len(real_dataset), len(generated_dataset))

    # Create dataloaders
    real_dataloader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    generated_dataloader = DataLoader(
        generated_dataset, batch_size=batch_size, shuffle=False
    )

    # Automatically detect the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize FID metric
    metric = FrechetInceptionDistance(feature=feature_dim).to(device)

    # Process real images
    print("Processing real images...")
    for batch in tqdm(real_dataloader):
        batch = batch.to(device)
        metric.update(batch, real=True)

    # Process generated images
    print("Processing generated images...")
    for batch in tqdm(generated_dataloader):
        batch = batch.to(device)
        metric.update(batch, real=False)

    # Compute FID
    fid_value = metric.compute()
    print(f"FID Score: {fid_value.item()}")

    return fid_value.item(), actual_num_images


def compute_fid_curve(
    ft_model=False,
    size=512,
    feature_dim=2048,
    min_images=500,
    max_images=5000,
    step=500,
    random_seed=42,
    guidance_scale=None,
):
    """
    Computes FID scores for different dataset sizes and plots a curve.

    Args:
        ft_model: Whether to use the fine-tuned model
        size: Image size
        feature_dim: Feature dimension for FID calculation
        min_images: Minimum number of images to test
        max_images: Maximum number of images to test
        step: Step size for the number of images
        random_seed: Random seed for reproducible image selection
        guidance_scale: Guidance scale used for image generation

    Returns:
        Tuple of (image_counts, fid_scores)
    """
    # Define the dataset sizes to test
    requested_counts = list(range(min_images, max_images + 1, step))
    actual_counts = []
    fid_scores = []

    print(f"Computing FID scores for {len(requested_counts)} different dataset sizes:")
    print(f"Requested sizes: {requested_counts}")

    # Compute FID scores for each dataset size
    for count in requested_counts:
        print(f"\n=== Computing FID for {count} images ===")
        score, actual_count = compute_fid(
            batch_size=256,
            ft_model=ft_model,
            size=size,
            feature_dim=feature_dim,
            num_images=count,
            random_seed=random_seed,
            guidance_scale=guidance_scale,
        )
        if score is not None and actual_count is not None:
            fid_scores.append(score)
            actual_counts.append(actual_count)
        else:
            print(f"Failed to compute FID for {count} images. Skipping this point.")

    # Create the plot with scientific aesthetics
    plt.figure(figsize=(10, 6))

    # Plot with markers and line
    plt.plot(
        actual_counts,  # Use the actual counts instead of the requested counts
        fid_scores,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=8,
        color="#1f77b4",
    )

    # Add grid for better readability
    plt.grid(True, linestyle="--", alpha=0.7)

    # Add labels and title
    plt.xlabel("Number of Images", fontsize=12)
    plt.ylabel("FID Score", fontsize=12)
    plt.title(
        f"FID Score vs Number of Images (Model: {'Fine-tuned' if ft_model else 'HuggingFace'})",
        fontsize=14,
    )

    # Format axes
    plt.tight_layout()
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Add text with details
    plt.figtext(
        0.02,
        0.02,
        f"Image Size: {size}px"
        + (f", CFG: {guidance_scale:.1f}" if guidance_scale else ""),
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.7, "pad": 5},
    )

    # Save the figure
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(script_dir, "../plots")
    os.makedirs(plots_dir, exist_ok=True)
    output_filename = os.path.join(
        plots_dir,
        f"fid_scores_curve_{'ft' if ft_model else 'hf'}_{size}"
        + (f"_cfg{guidance_scale:.1f}" if guidance_scale else "")
        + ".png",
    )
    plt.savefig(
        output_filename,
        bbox_inches="tight",
        dpi=300,
    )
    print(f"Saved plot to {output_filename}")

    # Also save the data as CSV for later use
    results_dir = os.path.join(script_dir, "../results")
    os.makedirs(results_dir, exist_ok=True)
    csv_filename = os.path.join(
        results_dir,
        f"fid_scores_data_{'ft' if ft_model else 'hf'}_{size}"
        + (f"_cfg{guidance_scale:.1f}" if guidance_scale else "")
        + ".csv",
    )
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Number of Images", "FID Score"])
        for act, score in zip(actual_counts, fid_scores):
            writer.writerow([act, score])
    print(f"Saved data to {csv_filename}")

    return actual_counts, fid_scores


def setup_parser():
    """
    Sets up the command-line argument parser.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Calculate FID score between real and generated images"
    )

    # Create subparsers for the two modes
    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")

    # Parser for single FID calculation
    fid_parser = subparsers.add_parser(
        "fid", help="Compute FID score for a specific number of images"
    )
    fid_parser.add_argument("--batch_size", type=int, default=256)
    fid_parser.add_argument("--ft_model", action="store_true")
    fid_parser.add_argument("--size", type=int, default=512)
    fid_parser.add_argument("--guidance_scale", type=float, default=None)
    fid_parser.add_argument("--feature_dim", type=int, default=2048)
    fid_parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="Number of random images to use for FID calculation. If not specified, all images are used.",
    )
    fid_parser.add_argument("--seed", type=int, default=42)

    # Parser for FID curve calculation
    curve_parser = subparsers.add_parser(
        "curve", help="Compute FID scores for different dataset sizes"
    )
    curve_parser.add_argument("--ft_model", action="store_true")
    curve_parser.add_argument("--size", type=int, default=512)
    curve_parser.add_argument("--guidance_scale", type=float, default=None)
    curve_parser.add_argument("--feature_dim", type=int, default=2048)
    curve_parser.add_argument("--min_images", type=int, default=500)
    curve_parser.add_argument("--max_images", type=int, default=5000)
    curve_parser.add_argument("--step", type=int, default=500)
    curve_parser.add_argument("--seed", type=int, default=42)

    return parser


def main():
    """
    Main entry point for the script.
    Parses command-line arguments and calls the appropriate function.
    """
    parser = setup_parser()
    args = parser.parse_args()

    # Default to 'fid' mode if no mode is specified
    if args.mode is None:
        print("No mode specified, defaulting to 'fid' mode.")
        args.mode = "fid"

    if args.mode == "fid":
        compute_fid(
            batch_size=args.batch_size if hasattr(args, "batch_size") else 256,
            ft_model=args.ft_model,
            size=args.size,
            guidance_scale=args.guidance_scale,
            feature_dim=args.feature_dim if hasattr(args, "feature_dim") else 2048,
            num_images=args.num_images if hasattr(args, "num_images") else None,
            random_seed=args.seed,
        )
    elif args.mode == "curve":
        compute_fid_curve(
            ft_model=args.ft_model,
            size=args.size,
            guidance_scale=args.guidance_scale,
            feature_dim=args.feature_dim if hasattr(args, "feature_dim") else 2048,
            min_images=args.min_images,
            max_images=args.max_images,
            step=args.step,
            random_seed=args.seed,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
