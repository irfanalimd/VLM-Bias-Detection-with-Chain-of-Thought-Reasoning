#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLaVA-CoT baseline for gender bias detection in image-caption pairs.
This script uses the LLaVA-CoT model (Large Language and Vision Assistant with Chain-of-Thought)
to analyze image-caption pairs for gender bias.
"""

import os
import argparse
import json
import logging
import re
from typing import Dict, List, Any, Optional
from pathlib import Path
from tqdm import tqdm

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llava_cot_baseline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="LLaVA-CoT baseline for gender bias detection")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input JSONL file with image-caption pairs")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save output JSONL file with predictions")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing the images")
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-1.5-13b-hf",
                        help="Hugging Face model name for LLaVA")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run inference on (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing (usually 1 for large models)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length for generation")
    return parser.parse_args()

def load_input_data(input_file: str) -> List[Dict[str, Any]]:
    """Load input data from a JSONL file."""
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line in {input_file}: {e}")
    
    logger.info(f"Loaded {len(data)} samples from {input_file}")
    return data

def load_llava_model(model_name: str, device: str) -> tuple:
    """Load the LLaVA model and processor."""
    # Load the processor
    processor = AutoProcessor.from_pretrained(model_name)
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    logger.info(f"Loaded LLaVA model: {model_name}")
    return model, processor

def construct_prompt(caption: str) -> str:
    """Construct a prompt for analyzing gender bias."""
    return (
        f"Analyze the following image and caption for gender bias. "
        f"Caption: \"{caption}\"\n\n"
        f"Provide your analysis as a chain-of-thought reasoning and conclude "
        f"with a binary label: 'Biased' or 'Not Biased'."
    )

def analyze_image_caption_pair(
    image_path: str,
    caption: str,
    model,
    processor,
    device: str,
    max_length: int
) -> Dict[str, Any]:
    """Analyze an image-caption pair for gender bias using LLaVA-CoT."""
    try:
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Construct prompt
        prompt = construct_prompt(caption)
        
        # Process inputs
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=False,
                num_beams=1
            )
        
        # Decode response
        response = processor.decode(output[0], skip_special_tokens=True)
        
        # Extract the assistant's response (remove the prompt)
        assistant_response = response[len(prompt):].strip()
        
        # Extract bias label
        if "Biased" in assistant_response.split("\n")[-1]:
            bias_label = "Biased"
        elif "Not Biased" in assistant_response.split("\n")[-1]:
            bias_label = "Not Biased"
        else:
            # Search for "Biased" or "Not Biased" in the full response
            if re.search(r'\bBiased\b', assistant_response) and not re.search(r'\bNot Biased\b', assistant_response):
                bias_label = "Biased"
            elif re.search(r'\bNot Biased\b', assistant_response):
                bias_label = "Not Biased"
            else:
                bias_label = "Unclear"
                logger.warning(f"Could not determine bias label from response for caption: {caption}")
        
        return {
            "reasoning": assistant_response,
            "bias_label": bias_label
        }
    
    except Exception as e:
        logger.error(f"Error analyzing image {image_path}: {e}")
        return {
            "reasoning": f"Error: {str(e)}",
            "bias_label": "Error"
        }

def process_data(
    data: List[Dict[str, Any]],
    model,
    processor,
    image_dir: str,
    device: str,
    max_length: int,
    batch_size: int
) -> List[Dict[str, Any]]:
    """Process all image-caption pairs in the dataset."""
    results = []
    
    # Process in batches (though batch_size=1 is typical for large models like LLaVA)
    for i in tqdm(range(0, len(data), batch_size), desc="Processing"):
        batch = data[i:i+batch_size]
        
        for item in batch:
            image_id = item["image_id"]
            caption = item["caption"]
            image_path = os.path.join(image_dir, image_id)
            
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
            
            # Analyze the image-caption pair
            analysis = analyze_image_caption_pair(
                image_path, caption, model, processor, device, max_length
            )
            
            # Create result
            result = {
                "image_id": image_id,
                "caption": caption,
                "reasoning": analysis["reasoning"],
                "bias_label": analysis["bias_label"]
            }
            
            results.append(result)
    
    return results

def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save results to a JSONL file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    logger.info(f"Saved {len(results)} results to {output_file}")

def main():
    """Main function to run the LLaVA-CoT baseline."""
    args = setup_args()
    
    # Load input data
    data = load_input_data(args.input)
    
    # Load LLaVA model
    model, processor = load_llava_model(args.model_name, args.device)
    
    # Process data
    results = process_data(
        data, model, processor, args.
        image_dir,
        args.device, args.max_length, args.batch_size
    )
    # Save results
save_results(results, args.output)

# Print summary statistics
bias_counts = {"Biased": 0, "Not Biased": 0, "Unclear": 0, "Error": 0}
for result in results:
    bias_label = result["bias_label"]
    bias_counts[bias_label] = bias_counts.get(bias_label, 0) + 1

logger.info(f"Bias distribution: {bias_counts}")
