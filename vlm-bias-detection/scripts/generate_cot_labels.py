#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate chain-of-thought (CoT) reasoning and bias labels for image-caption pairs using GPT-4.
This script reads image-caption pairs (e.g., from Flickr30k) and queries GPT-4 to determine
if the caption exhibits gender bias, saving the results in JSONL format.
"""

import os
import json
import time
import argparse
import logging
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

import openai
from PIL import Image
import base64
from io import BytesIO
import concurrent.futures
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("generate_cot_labels.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate CoT reasoning and bias labels using GPT-4")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to the input file with image-caption pairs (JSONL)")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save the output JSONL file with reasoning and labels")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing the images")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key (if not provided, will use OPENAI_API_KEY env var)")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Number of image-caption pairs to process in parallel")
    parser.add_argument("--max_retries", type=int, default=3,
                        help="Maximum number of retries for API calls")
    parser.add_argument("--continue_from_existing", action="store_true",
                        help="Continue from an existing output file")
    return parser.parse_args()

def encode_image(image_path: str) -> str:
    """Encode an image as a base64 string for GPT-4 Vision API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def create_prompt_for_gpt4(image_base64: str, caption: str) -> List[Dict[str, Any]]:
    """Create a prompt for GPT-4 to analyze gender bias in the given image-caption pair."""
    return [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that analyzes image-caption pairs for gender bias. "
                "Your task is to determine if a caption exhibits gender bias when paired with its image. "
                "Gender bias can include stereotypical assumptions, unequal representation, "
                "or using gendered language unnecessarily. "
                "Provide a step-by-step chain-of-thought reasoning and conclude with a binary label: "
                "'Biased' or 'Not Biased'."
            )
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": f"Analyze this image and its caption for gender bias. Caption: \"{caption}\"\n\nProvide your chain-of-thought reasoning and conclude with a binary label: 'Biased' or 'Not Biased'."
                }
            ]
        }
    ]

def query_gpt4(
    messages: List[Dict[str, Any]], 
    api_key: str, 
    max_retries: int = 3
) -> Optional[Dict[str, Any]]:
    """Query GPT-4 with retry logic for transient errors."""
    client = openai.OpenAI(api_key=api_key)
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=messages,
                max_tokens=1000
            )
            return response
        except openai.APIError as e:
            wait_time = 2 ** attempt  # Exponential backoff
            logger.warning(f"API error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
        except openai.RateLimitError:
            wait_time = 15 + 5 * attempt  # Rate limit backoff
            logger.warning(f"Rate limited (attempt {attempt+1}/{max_retries}). Retrying in {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None
    
    logger.error(f"Failed after {max_retries} attempts")
    return None

def parse_gpt4_response(response: Dict[str, Any]) -> Tuple[str, str]:
    """Parse GPT-4's response to extract reasoning and bias label."""
    content = response.choices[0].message.content
    
    # Look for the bias label at the end
    if "Biased" in content.split("\n")[-1]:
        bias_label = "Biased"
    elif "Not Biased" in content.split("\n")[-1]:
        bias_label = "Not Biased"
    else:
        # If not found in the last line, search in the full content
        if "Biased" in content and "Not Biased" not in content:
            bias_label = "Biased"
        elif "Not Biased" in content:
            bias_label = "Not Biased"
        else:
            # Default if we can't clearly determine
            bias_label = "Unclear"
            logger.warning(f"Could not clearly determine bias label from response: {content}")
    
    # Return the full reasoning and the extracted label
    return content, bias_label

def process_image_caption_pair(
    item: Dict[str, str], 
    image_dir: str, 
    api_key: str, 
    max_retries: int
) -> Optional[Dict[str, Any]]:
    """Process a single image-caption pair."""
    image_id = item["image_id"]
    caption = item["caption"]
    image_path = os.path.join(image_dir, image_id)
    
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return None
    
    try:
        image_base64 = encode_image(image_path)
        messages = create_prompt_for_gpt4(image_base64, caption)
        response = query_gpt4(messages, api_key, max_retries)
        
        if response is None:
            return None
        
        reasoning, bias_label = parse_gpt4_response(response)
        
        result = {
            "image_id": image_id,
            "caption": caption,
            "reasoning": reasoning,
            "bias_label": bias_label
        }
        
        return result
    except Exception as e:
        logger.error(f"Error processing {image_id}: {e}")
        return None

def load_existing_results(output_file: str) -> Tuple[List[Dict[str, Any]], set]:
    """Load existing results from an output file and return processed image IDs."""
    existing_results = []
    processed_ids = set()
    
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    existing_results.append(item)
                    processed_ids.add(item["image_id"])
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse line: {line.strip()}")
    
    return existing_results, processed_ids

def main():
    """Main function to process image-caption pairs and generate labels."""
    args = setup_args()
    
    # Set API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No OpenAI API key provided. Set OPENAI_API_KEY env var or use --api_key")
    
    # Load input data
    input_data = []
    with open(args.input_file, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                input_data.append(item)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse input line: {line.strip()}")
    
    logger.info(f"Loaded {len(input_data)} image-caption pairs from {args.input_file}")
    
    # Handle continuation from existing file
    if args.continue_from_existing and os.path.exists(args.output_file):
        existing_results, processed_ids = load_existing_results(args.output_file)
        logger.info(f"Continuing from existing file with {len(existing_results)} results")
        # Filter input data to only process new items
        input_data = [item for item in input_data if item["image_id"] not in processed_ids]
        logger.info(f"Remaining items to process: {len(input_data)}")
    else:
        existing_results = []
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    
    # Process image-caption pairs in parallel
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.batch_size) as executor:
        future_to_item = {
            executor.submit(
                process_image_caption_pair, 
                item, 
                args.image_dir, 
                api_key, 
                args.max_retries
            ): item for item in input_data
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_item), total=len(input_data)):
            result = future.result()
            if result:
                results.append(result)
                # Write incrementally to avoid losing progress
                with open(args.output_file, 'a') as f:
                    f.write(json.dumps(result) + '\n')
    
    # If continuing, combine with existing results (though we've already written incrementally)
    all_results = existing_results + results
    
    logger.info(f"Processed {len(results)} new items")
    logger.info(f"Total results: {len(all_results)}")
    
    # Final stats on bias distribution
    bias_counts = {"Biased": 0, "Not Biased": 0, "Unclear": 0}
    for result in all_results:
        bias_counts[result["bias_label"]] += 1
    
    logger.info(f"Bias distribution: {bias_counts}")

if __name__ == "__main__":
    main()
