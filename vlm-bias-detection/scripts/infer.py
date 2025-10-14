#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference script for the fine-tuned LlamaV-o1 model for gender bias detection.
This script processes a JSONL file containing image-caption pairs and outputs
the model's bias analysis for each pair.
"""

import os
import argparse
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from tqdm import tqdm

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, CLIPProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with the fine-tuned model")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input JSONL file with image-caption pairs")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save output JSONL file with predictions")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing the images")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length for generation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (cuda or cpu)")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 precision for inference")
    return parser.parse_args()

def load_input_data(input_file: str) -> List[Dict[str, str]]:
    """Load input data from a JSONL file."""
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse line: {line.strip()}")
    
    logger.info(f"Loaded {len(data)} samples from {input_file}")
    return data

def load_model(
    model_path: str, 
    device: str, 
    fp16: bool
) -> tuple:
    """Load the fine-tuned model, tokenizer, and image processor."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device if device == "auto" else None,
        torch_dtype=torch.float16 if fp16 else None,
    )
    
    # Move model to device if not using "auto" device mapping
    if device != "auto":
        model = model.to(device)
    
    # Load image processor
    # For LlamaV-o1, we use CLIP processor for image preprocessing
    image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    logger.info(f"Model loaded from {model_path}")
    return model, tokenizer, image_processor

def process_batch(
    batch: List[Dict[str, str]],
    model: torch.nn.Module,
    tokenizer,
    image_processor,
    image_dir: str,
    device: str,
    max_length: int
) -> List[Dict[str, Any]]:
    """Process a batch of image-caption pairs."""
    results = []
    
    for item in batch:
        image_id = item["image_id"]
        caption = item["caption"]
        
        # Load image
        image_path = os.path.join(image_dir, image_id)
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue
        
        try:
            # Process image
            image = Image.open(image_path).convert("RGB")
            pixel_values = image_processor(image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            
            # Prepare prompt
            prompt = f"Analyze this image and caption for gender bias: \"{caption}\""
            
            # Tokenize prompt
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate prediction
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=False,
                    num_beams=1,
                )
            
            # Decode output
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract reasoning and bias label from generated text
            result_text = generated_text[len(prompt):].strip()
            
            # Parse to extract bias label
            if "Biased" in result_text.split("\n")[-1]:
                bias_label = "Biased"
            elif "Not Biased" in result_text.split("\n")[-1]:
                bias_label = "Not Biased"
            else:
                bias_label = "Unclear"
            
            # Create result
            result = {
                "image_id": image_id,
                "caption": caption,
                "reasoning": result_text,
                "bias_label": bias_label
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing {image_id}: {e}")
    
    return results

def main():
    """Main function to run inference."""
    args = setup_args()
    
    # Load model, tokenizer, and image processor
    model, tokenizer, image_processor = load_model(args.model_path, args.device, args.fp16)
    
    # Load input data
    input_data = load_input_data(args.input)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # Process data in batches
    all_results = []
    
    for i in tqdm(range(0, len(input_data), args.batch_size)):
        batch = input_data[i:i+args.batch_size]
        results = process_batch(
            batch, model, tokenizer, image_processor, 
            args.image_dir, args.device, args.max_length
        )
        all_results.extend(results)
        
        # Write results incrementally to avoid losing progress
        with open(args.output, 'a') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
    
    logger.info(f"Processed {len(all_results)} samples")
    
    # Count bias labels
    bias_counts = {"Biased": 0, "Not Biased": 0, "Unclear": 0}
    for result in all_results:
        bias_counts[result["bias_label"]] += 1
    
    logger.info(f"Bias distribution: {bias_counts}")

if __name__ == "__main__":
    main()
