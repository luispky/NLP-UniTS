import os
import time
import argparse
import numpy as np
from rich import print
import torch
from COCOKarpathyDataset import get_coco_karpathy_dataloader
from utils import (
    seed_everything,
    check_existing_image,
    setup_model_and_processor,
    prefix_allowed_tokens_fn,
    add_positive_prompt_suffix,
    setup_cfg_logits_processor,
)
from paths import (
    MODEL_NAME_OR_PATH,
    PEFT_PATH,
    DATASET_PATH,
)
import json
from functools import partial


def main():
    parser = argparse.ArgumentParser(description="EMU-3 Image Generation on Single GPU")
    parser.add_argument(
        "--start_idx", type=int, default=0, help="Starting index for dataset samples"
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--ft_model", action="store_true")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--guidance_scale", type=float, default=3)
    args = parser.parse_args()

    # Sanity check
    if args.guidance_scale < 1:
        raise ValueError("Guidance scale must be >= 1.0")

    # Seed for reproducibility
    seed_everything(args.seed)

    # Set up model and processor with Accelerator
    model, processor = setup_model_and_processor(
        model_name_or_path=MODEL_NAME_OR_PATH,
        peft_path=PEFT_PATH,
        ft_model=args.ft_model,
        dtype=torch.float16,
    )

    # Get the downsample ratio of the processor and compute the max tokens per image
    spatial_factor = processor.image_processor.spatial_factor
    max_tokens_per_image = (args.size // spatial_factor) * (
        args.size // spatial_factor + 1
    ) + 3

    # Load dataset (no images needed for generation)
    dataloader = get_coco_karpathy_dataloader(
        data_dir=DATASET_PATH,
        split=args.split,
        start_idx=args.start_idx,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        return_images=False,
    )

    # Output directory - include CFG scale in the directory name if CFG is enabled
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(
        script_dir,
        f"../generated_images_{'ft' if args.ft_model else 'hf'}_{args.size}"
        + (f"_cfg{args.guidance_scale:.1f}" if args.guidance_scale > 1 else ""),
    )
    os.makedirs(output_dir, exist_ok=True)

    # Caption mapping
    caption_mapping_path = os.path.join(
        output_dir, f"caption_mapping_{args.split}_{args.start_idx}.json"
    )
    if os.path.exists(caption_mapping_path):
        with open(caption_mapping_path, "r") as f:
            caption_mapping = json.load(f)
    else:
        caption_mapping = {}

    print(
        f"Generating images from index {args.start_idx} to {args.start_idx + args.max_samples if args.max_samples else 'end'}..."
    )
    if args.guidance_scale > 1:
        print(f"Using classifier-free guidance with scale {args.guidance_scale}")

    times = []
    num_images_generated = 0
    total_batches = len(dataloader)

    for idx, batch in enumerate(dataloader):
        print(f"\nProcessing batch {idx + 1}/{total_batches}")
        captions = batch["caption"]
        imgids = batch["image_id"]

        # Check if image has already been generated and skip if so
        skip_mask = torch.tensor(
            [check_existing_image(imgid, output_dir) for imgid in imgids],
            dtype=torch.bool,
        )
        if skip_mask.all():
            print(
                f"All images in batch {idx + 1} have already been generated, skipping"
            )
            continue

        # Filter captions and IDs for non-existing images only
        captions_to_process = [c for c, skip in zip(captions, skip_mask) if not skip]
        imgids_to_process = [iid for iid, skip in zip(imgids, skip_mask) if not skip]

        if not captions_to_process:  # Nothing to process in this batch
            print(f"All images in batch {idx + 1} have already been generated")
            continue

        # Print progress information
        print(f"Generating {len(captions_to_process)} images in batch {idx + 1}")

        # Add positive prompt suffix to each caption when using CFG
        if args.guidance_scale > 1:
            captions_to_process = add_positive_prompt_suffix(captions_to_process)
            print("Added positive prompt suffix to captions")

        # Tokenize the captions for new images with positive prompts
        tokenized_inputs = processor(
            text=captions_to_process,
            padding=True,
            return_tensors="pt",
            return_for_image_generation=True,
            image_area=args.size * args.size,  # Fixes the image size
            padding_side="left",
        ).to(model.device)

        # Get the height and width of one image
        height, width = tokenized_inputs["image_sizes"][0]  # [batch_size, 2]

        # Clean up memory
        del captions, imgids, skip_mask
        torch.cuda.empty_cache()

        # Compute the max tokens for the batch
        max_tokens = max_tokens_per_image * len(captions_to_process)

        start_time = time.time()
        try:
            # Generate images with or without CFG
            # CFG formula: scores = uncond_scores + guidance_scale * (cond_scores - uncond_scores)
            # When guidance_scale=1, this simplifies to just cond_scores (no guidance)
            if args.guidance_scale == 1:  # No CFG: just use the conditional scores
                with torch.no_grad():
                    model_outputs = model.generate(
                        **tokenized_inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        prefix_allowed_tokens_fn=partial(
                            prefix_allowed_tokens_fn,
                            model=model,
                            processor=processor,
                            height=height,
                            width=width,
                        ),
                    )
            else:  # CFG: use both conditional and unconditional scores
                # Set up CFG logits processor with appropriate guidance scale
                try:
                    logits_processor = setup_cfg_logits_processor(
                        model=model,
                        processor=processor,
                        num_captions=len(captions_to_process),
                        image_area=args.size * args.size,
                        constrained_fn=partial(
                            prefix_allowed_tokens_fn,
                            model=model,
                            processor=processor,
                            height=height,
                            width=width,
                        ),
                        guidance_scale=args.guidance_scale,
                    )

                    with torch.no_grad():
                        model_outputs = model.generate(
                            tokenized_inputs.input_ids,
                            max_new_tokens=max_tokens,
                            do_sample=True,
                            logits_processor=logits_processor,
                            attention_mask=tokenized_inputs.attention_mask,
                        )

                    # Clean up memory
                    del logits_processor
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error setting up CFG: {e}")
                    raise

            # Record timing stats
            times.append(time.time() - start_time)
            print(
                f"Batch {idx + 1}: Generated {len(captions_to_process)} images in {times[-1] / 60:.2f} minutes"
            )
            num_images_generated += len(captions_to_process)

            # Remove text tokens
            model_outputs = model_outputs[:, tokenized_inputs.input_ids.shape[1] :]

            # Clean up memory
            del tokenized_inputs
            torch.cuda.empty_cache()

            # Decode images one at a time because of memory issues
            for img_id, model_output, caption in zip(
                imgids_to_process, model_outputs, captions_to_process
            ):
                try:
                    decoded_image_tensor = (
                        model.decode_image_tokens(
                            model_output.unsqueeze(0),  # add batch dimension
                            height=height,
                            width=width,
                        )
                        .squeeze(0)  # remove batch dimension
                        .cpu()
                    )
                    pil_image = processor.postprocess(
                        [decoded_image_tensor.float()], return_tensors="PIL.Image.Image"
                    )["pixel_values"][0]
                    pil_image.save(os.path.join(output_dir, f"{img_id:012d}.png"))
                    caption_mapping[str(img_id)] = caption
                except Exception as e:
                    print(f"Error processing image {img_id}: {e}")

                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error in batch {idx + 1}: {e}")

        # Clean up memory
        del model_outputs, imgids_to_process, captions_to_process
        torch.cuda.empty_cache()

        # Save updated caption mapping after each batch
        with open(caption_mapping_path, "w") as f:
            json.dump(caption_mapping, f, indent=2)

    # Filter out zero times (from skipped or failed generations)
    times = np.array(times)
    valid_times = times[times > 0]

    if len(valid_times) > 0:
        # Print timing statistics
        total_time_minutes = np.sum(valid_times) / 60
        print(f"\nTotal time taken: {total_time_minutes:.2f} minutes")
        print(f"Total images generated: {num_images_generated}")
        print(
            f"Average time taken per image: {total_time_minutes / num_images_generated:.2f} minutes"
        )
    else:
        print("No images were generated.")


if __name__ == "__main__":
    main()
