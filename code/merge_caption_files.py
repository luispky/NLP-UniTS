import os
import json
import glob
import argparse


def merge_caption_files(ft_model=True, size=512, guidance_scale=None):
    """
    Merge multiple JSON caption files into a single file with unique image ID and caption pairs.

    Args:
        ft_model (bool): If True, use fine-tuned model directory. If False, use HuggingFace model directory.
        size (int): Image resolution (e.g., 512)
    """
    # Determine the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct path to the directory containing caption files
    model_type = "ft" if ft_model else "hf"
    directory_path = os.path.join(
        script_dir,
        f"../generated_images_{model_type}_{size}"
        + (f"_cfg{guidance_scale:.1f}" if guidance_scale else ""),
    )

    # Set output file path
    output_file = os.path.join(directory_path, "caption_mapping_merged.json")

    # Find all caption mapping JSON files in the directory
    json_pattern = os.path.join(directory_path, "caption_mapping_*.json")
    json_files = glob.glob(json_pattern)

    if not json_files:
        print(f"No caption mapping JSON files found in {directory_path}")
        return

    print(f"Found {len(json_files)} caption mapping files to merge:")
    for file in json_files:
        print(f"  - {os.path.basename(file)}")

    # Initialize merged dictionary
    merged_data = {}
    total_entries = 0

    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                file_entries = len(data)
                total_entries += file_entries
                print(
                    f"  Processing {os.path.basename(json_file)}: {file_entries} entries"
                )

                # Update merged dictionary with current file's data
                merged_data.update(data)

        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")

    # Save merged data to output file
    try:
        with open(output_file, "w") as f:
            json.dump(merged_data, f, indent=2)

        unique_entries = len(merged_data)
        print(f"\nSuccessfully merged {len(json_files)} files")
        print(f"Total entries processed: {total_entries}")
        print(f"Unique entries in merged file: {unique_entries}")
        print(f"Merged file saved to: {output_file}")

    except Exception as e:
        print(f"Error saving merged file: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge multiple caption JSON files into one file with unique entries"
    )
    parser.add_argument("--ft_model", action="store_true")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=None)

    args = parser.parse_args()
    merge_caption_files(args.ft_model, args.size, args.guidance_scale)
