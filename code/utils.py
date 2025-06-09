import os
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoProcessor, Emu3Config, Emu3ForConditionalGeneration
from peft import PeftModel
from typing import Tuple, List
from transformers.generation import (
    LogitsProcessorList,
    PrefixConstrainedLogitsProcessor,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)


def seed_everything(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Seed value for random, numpy, torch and cuda:0
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_existing_image(
    imgid: int,
    output_dir: str,
) -> bool:
    """
    Check if an image has already been generated for the given parameters.

    Args:
        imgid: Image ID
        output_dir: Directory to check for existing outputs

    Returns:
        True if the output file exists, False otherwise
    """
    filepath = os.path.join(output_dir, f"{imgid:012d}.png")
    if os.path.exists(filepath):
        return True
    return False


def prefix_allowed_tokens_fn(batch_id, input_ids, model, processor, height, width):
    """
    Controls token generation for structured image output with Emu3.

    This function enforces the correct sequence of tokens during image generation:
    - Visual tokens for pixel data within the image grid
    - End-of-line tokens after each row
    - End-of-frame, end-of-image, and end-of-sequence tokens at appropriate positions
    - Padding tokens after the complete sequence

    Args:
        batch_id: Batch identifier for the current generation
        input_ids: Current sequence of token IDs
        model: The Emu3 model being used for generation
        processor: The processor for the model
        height: Height of the image in tokens
        width: Width of the image in tokens

    Returns:
        Tuple of allowed token IDs for the next position
    """
    visual_tokens = model.vocabulary_mapping.image_tokens
    # Special tokens for structuring the output
    image_wrapper_token_id = torch.tensor(
        [processor.tokenizer.image_wrapper_token_id], device=model.device
    )  # Start of image token
    eoi_token_id = torch.tensor(
        [processor.tokenizer.eoi_token_id], device=model.device
    )  # End of image token
    eos_token_id = torch.tensor(
        [processor.tokenizer.eos_token_id], device=model.device
    )  # End of sequence token
    pad_token_id = torch.tensor(
        [processor.tokenizer.pad_token_id], device=model.device
    )  # Padding token
    eof_token_id = torch.tensor(
        [processor.tokenizer.eof_token_id], device=model.device
    )  # End of frame token
    eol_token_id = processor.tokenizer.encode("<|extra_200|>", return_tensors="pt")[
        0
    ]  # End of line token

    # Find where the image starts in the sequence
    position = torch.nonzero(input_ids == image_wrapper_token_id, as_tuple=True)[0][0]
    offset = input_ids.shape[0] - position  # How many tokens since image start
    # Control token generation based on position:
    if offset % (width + 1) == 0:  # At the end of each row
        return (eol_token_id,)  # Insert end-of-line token
    elif offset == (width + 1) * height + 1:  # After all rows
        return (eof_token_id,)  # Insert end-of-frame token
    elif offset == (width + 1) * height + 2:  # After frame
        return (eoi_token_id,)  # Insert end-of-image token
    elif offset == (width + 1) * height + 3:  # After image
        return (eos_token_id,)  # Insert end-of-sequence token
    elif offset > (width + 1) * height + 3:  # Beyond image
        return (pad_token_id,)  # Pad remaining tokens
    else:  # During image generation
        return visual_tokens  # Generate pixel tokens


def setup_model_and_processor(
    model_name_or_path: str,
    peft_path: str,
    ft_model: bool = False,
    dtype: torch.dtype = torch.float16,
) -> Tuple[nn.Module, AutoProcessor]:
    """
    Set up the Emu3 model and processor for single-GPU inference.

    Args:
        model_name_or_path: Path to base model
        peft_path: Path to PEFT weights
        ft_model: Whether to use fine-tuned model
        dtype: Data type (float16 or bfloat16)
    """
    processor = AutoProcessor.from_pretrained(
        model_name_or_path,
    )

    # Load model
    if ft_model:
        config = Emu3Config.from_pretrained(model_name_or_path)
        model = Emu3ForConditionalGeneration.from_pretrained(
            model_name_or_path,
            from_tf=bool(".ckpt" in model_name_or_path),
            config=config,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            device_map="cuda:0",
        )
        model = PeftModel.from_pretrained(model, peft_path)
        model = model.merge_and_unload()

    else:
        model = Emu3ForConditionalGeneration.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
            device_map="cuda:0",
        )

    return model.eval(), processor


# Prompt constants for classifier-free guidance
POSITIVE_PROMPT = " masterpiece, film grained, best quality."
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, \
    missing fingers, extra digit, fewer digits, cropped, worst quality, \
    low quality, normal quality, jpeg artifacts, signature, watermark, username, \
    blurry."


def add_positive_prompt_suffix(prompts: List[str]) -> List[str]:
    """
    Add a positive prompt suffix to enhance the quality of generated images.
    
    Args:
        prompts: List of original prompt strings
        
    Returns:
        List of prompts with positive suffixes added
    """
    return [prompt + POSITIVE_PROMPT for prompt in prompts]


def setup_cfg_logits_processor(
    model: nn.Module,
    processor: AutoProcessor,
    num_captions: int,
    image_area: int,
    constrained_fn: callable,
    guidance_scale: float,
) -> LogitsProcessorList:
    """
    Set up the classifier-free guidance logits processor for image generation.
    
    This combines the UnbatchedClassifierFreeGuidanceLogitsProcessor with a
    PrefixConstrainedLogitsProcessor to ensure both the proper guidance and
    image structure constraints are maintained.
    
    Args:
        model: The Emu3 model
        processor: The Emu3 processor
        num_captions: Number of captions being processed
        image_area: Area of the target image in pixels
        constrained_fn: Function to enforce image token structure
        guidance_scale: Scale factor for classifier-free guidance
        
    Returns:
        LogitsProcessorList configured for classifier-free guidance
    """
    try:
        # Tokenize negative prompts
        neg_tokenized_inputs = processor(
            text=[NEGATIVE_PROMPT] * num_captions,
            padding=True,
            return_tensors="pt",
            return_for_image_generation=True,
            image_area=image_area,
            padding_side="left",
        ).to(model.device)

        # Create combined logits processor
        logits_processor = LogitsProcessorList(
            [
                UnbatchedClassifierFreeGuidanceLogitsProcessor(
                    guidance_scale,
                    model,
                    unconditional_ids=neg_tokenized_inputs.input_ids,
                ),
                PrefixConstrainedLogitsProcessor(
                    constrained_fn,
                    num_beams=1,
                ),
            ]
        )
        return logits_processor
    except Exception as e:
        raise RuntimeError(f"Failed to set up CFG logits processor: {e}")
