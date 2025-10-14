#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare and filter Flickr30K dataset for bias detection task.
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Keywords for filtering social context
PEOPLE_KEYWORDS = [
    "man", "woman", "men", "women", "person", "people", "boy", "girl",
    "male", "female", "guy", "lady", "gentleman", "child", "children",
    "human", "individual", "adult", "kid", "teen", "teenager", "toddler",
    "baby", "infant", "elderly", "senior", "youth", "family", "couple"
]

SOCIAL_ROLE_KEYWORDS = [
    "doctor", "nurse", "teacher", "student", "worker", "engineer",
    "scientist", "artist", "musician", "athlete", "chef", "cook",
    "waiter", "waitress", "manager", "CEO", "president", "leader",
    "assistant", "secretary", "receptionist", "cleaner", "driver",
    "pilot", "surgeon", "dentist", "lawyer", "judge", "police",
    "firefighter", "soldier", "farmer", "builder", "carpenter",
    "plumber", "electrician", "mechanic", "programmer", "designer",
    "writer", "journalist", "photographer", "actor", "actress",
    "model", "dancer", "coach", "counselor", "therapist", "professor",
    "researcher", "analyst", "consultant", "accountant", "banker"
]

ACTIVITY_KEYWORDS = [
    "cooking", "cleaning", "working", "teaching", "learning", "playing",
    "running", "walking", "sitting", "standing", "talking", "eating",
    "reading", "writing", "studying", "exercising", "shopping", "driving"
]

def setup_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare Flickr30K dataset for bias detection"
    )
    parser.add_argument(
        "--flickr_dir",
        type=str,
        required=True,
        help="Path to Flickr30K dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.818,
        help="Ratio of training data (default: 0.818 = 81.8%%)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.091,
        help="Ratio of validation data (default: 0.091 = 9.1%%)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--min_caption_length",
        type=int,
        default=10,
        help="Minimum caption length in characters"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    return parser.parse_args()


def load_flickr_captions(flickr_dir: str) -> List[Dict[str, str]]:
    """Load captions from Flickr30K dataset."""
    captions_file = os.path.join(flickr_dir, "results_20130124.token")
    
    if not os.path.exists(captions_file):
        # Try alternative filename
        captions_file = os.path.join(flickr_dir, "results.token")
    
    if not os.path.exists(captions_file):
        raise FileNotFoundError(
            f"Captions file not found in {flickr_dir}\n"
            f"Expected: results_20130124.token or results.token\n"
            f"Please ensure Flickr30K dataset is properly downloaded."
        )
    
    data = []
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                image_caption_id, caption = parts
                # Extract image ID (format: 12345.jpg#0)
                image_id = image_caption_id.split('#')[0]
                data.append({
                    "image_id": image_id,
                    "caption": caption.strip()
                })
    
    logger.info(f"Loaded {len(data)} captions from Flickr30K")
    return data


def filter_by_length(data: List[Dict[str, str]], min_length: int) -> List[Dict[str, str]]:
    """Filter captions by minimum length."""
    filtered = [item for item in data if len(item["caption"]) >= min_length]
    logger.info(f"After length filter: {len(filtered)} captions (min_length={min_length})")
    return filtered


def filter_social_context(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Filter captions containing people and social contexts."""
    filtered = []
    
    for item in data:
        caption_lower = item["caption"].lower()
        
        # Check for people keywords
        has_people = any(keyword in caption_lower for keyword in PEOPLE_KEYWORDS)
        
        # Check for social role keywords
        has_social_role = any(
            keyword in caption_lower for keyword in SOCIAL_ROLE_KEYWORDS
        )
        
        # Check for activity keywords
        has_activity = any(
            keyword in caption_lower for keyword in ACTIVITY_KEYWORDS
        )
        
        # Include if has people AND (social role OR activity)
        if has_people and (has_social_role or has_activity):
            filtered.append(item)
    
    logger.info(
        f"Filtered to {len(filtered)} captions with social context "
        f"({len(filtered)/len(data)*100:.1f}% of input)"
    )
    return filtered


def check_image_exists(image_id: str, image_dir: str) -> bool:
    """Check if image file exists."""
    image_path = os.path.join(image_dir, image_id)
    return os.path.exists(image_path)


def filter_by_image_existence(
    data: List[Dict[str, str]], 
    flickr_dir: str
) -> List[Dict[str, str]]:
    """Filter out captions where image doesn't exist."""
    image_dir = os.path.join(flickr_dir, "flickr30k-images")
    if not os.path.exists(image_dir):
        image_dir = os.path.join(flickr_dir, "images")
    
    if not os.path.exists(image_dir):
        logger.warning(f"Image directory not found in {flickr_dir}")
        logger.warning("Skipping image existence check")
        return data
    
    filtered = []
    missing_count = 0
    
    for item in data:
        if check_image_exists(item["image_id"], image_dir):
            filtered.append(item)
        else:
            missing_count += 1
    
    logger.info(f"Found {len(filtered)} captions with existing images")
    if missing_count > 0:
        logger.warning(f"Skipped {missing_count} captions with missing images")
    
    return filtered


def deduplicate_images(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Keep only one caption per image (the first one)."""
    seen_images = set()
    deduped = []
    
    for item in data:
        if item["image_id"] not in seen_images:
            deduped.append(item)
            seen_images.add(item["image_id"])
    
    logger.info(
        f"Deduplicated to {len(deduped)} unique images "
        f"(removed {len(data) - len(deduped)} duplicate image IDs)"
    )
    return deduped


def split_data(
    data: List[Dict[str, str]],
    train_ratio: float,
    val_ratio: float,
    seed: int
) -> Dict[str, List[Dict[str, str]]]:
    """Split data into train/val/test sets."""
    random.seed(seed)
    random.shuffle(data)
    
    n_total = len(data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    splits = {
        "train": data[:n_train],
        "val": data[n_train:n_train+n_val],
        "test": data[n_train+n_val:]
    }
    
    logger.info(
        f"Split data: train={len(splits['train'])} ({len(splits['train'])/n_total*100:.1f}%), "
        f"val={len(splits['val'])} ({len(splits['val'])/n_total*100:.1f}%), "
        f"test={len(splits['test'])} ({len(splits['test'])/n_total*100:.1f}%)"
    )
    
    return splits


def save_splits(
    splits: Dict[str, List[Dict[str, str]]],
    output_dir: str
):
    """Save data splits to JSONL files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in splits.items():
        output_file = os.path.join(
            output_dir, 
            f"{split_name}_captions.jsonl"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in split_data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Saved {len(split_data)} items to {output_file}")


def create_readme(output_dir: str, stats: Dict[str, int]):
    """Create README for processed data."""
    readme_path = os.path.join(output_dir, "DATA_INFO.md")
    
    content = f"""# Processed Flickr30K Data for Bias Detection

## Overview

This directory contains processed subsets of the Flickr30K dataset filtered for
social context and bias detection tasks.

## Statistics

- **Original captions**: {stats['original']}
- **After filtering**: {stats['filtered']}
- **Unique images**: {stats['unique']}
- **Training set**: {stats['train']}
- **Validation set**: {stats['val']}
- **Test set**: {stats['test']}

## Files

- `train_captions.jsonl`: Training set image-caption pairs
- `val_captions.jsonl`: Validation set image-caption pairs
- `test_captions.jsonl`: Test set image-caption pairs

## Format

Each line in the JSONL files contains:

```json
{{
    "image_id": "12345.jpg",
    "caption": "A person walking in the park."
}}
```

## Next Steps

1. **Generate CoT reasoning labels using GPT-4**:
   ```bash
   export OPENAI_API_KEY="your-key"
   
   python scripts/generate_cot_labels.py \\
     --input_file data/processed/train_captions.jsonl \\
     --output_file data/processed/train_annotations.jsonl \\
     --image_dir /path/to/flickr30k/images
   ```

2. **Train the model**:
   ```bash
   python scripts/train.py --config config/train_config.yaml
   ```

## Filtering Criteria

Captions were filtered to include:
- References to people (person, man, woman, child, etc.)
- Social roles and occupations
- Activities involving human interaction

This filtering resulted in approximately {stats['filtered']} caption-image pairs focused on
social contexts where bias is most relevant.

## Data Sources

- **Original Dataset**: Flickr30K (Young et al., 2014)
- **Processing Date**: {stats['date']}
- **Random Seed**: {stats['seed']}
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"Created README at {readme_path}")


def print_statistics(data: List[Dict[str, str]]):
    """Print dataset statistics."""
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    # Caption lengths
    lengths = [len(item['caption']) for item in data]
    print(f"\nCaption Lengths:")
    print(f"  Mean: {sum(lengths)/len(lengths):.1f} characters")
    print(f"  Min: {min(lengths)} characters")
    print(f"  Max: {max(lengths)} characters")
    
    # Word counts
    word_counts = [len(item['caption'].split()) for item in data]
    print(f"\nWord Counts:")
    print(f"  Mean: {sum(word_counts)/len(word_counts):.1f} words")
    print(f"  Min: {min(word_counts)} words")
    print(f"  Max: {max(word_counts)} words")
    
    print("="*70 + "\n")


def main():
    """Main function."""
    args = setup_args()
    
    logger.info("="*70)
    logger.info("Flickr30K Data Preparation for Bias Detection")
    logger.info("="*70)
    
    # Load Flickr30K captions
    logger.info("\n[1/7] Loading Flickr30K captions...")
    data = load_flickr_captions(args.flickr_dir)
    original_count = len(data)
    
    # Apply max_samples limit if specified
    if args.max_samples:
        data = data[:args.max_samples]
        logger.info(f"Limited to {args.max_samples} samples for testing")
    
    # Filter by length
    logger.info("\n[2/7] Filtering by caption length...")
    data = filter_by_length(data, args.min_caption_length)
    
    # Filter for social context
    logger.info("\n[3/7] Filtering for social context...")
    filtered_data = filter_social_context(data)
    
    # Check image existence
    logger.info("\n[4/7] Checking image existence...")
    filtered_data = filter_by_image_existence(filtered_data, args.flickr_dir)
    
    # Deduplicate images (keep one caption per image)
    logger.info("\n[5/7] Deduplicating images...")
    deduped_data = deduplicate_images(filtered_data)
    
    # Print statistics
    print_statistics(deduped_data)
    
    # Split data
    logger.info("\n[6/7] Splitting data...")
    splits = split_data(
        deduped_data,
        args.train_ratio,
        args.val_ratio,
        args.seed
    )
    
    # Save splits
    logger.info("\n[7/7] Saving splits...")
    save_splits(splits, args.output_dir)
    
    # Create README
    from datetime import datetime
    stats = {
        'original': original_count,
        'filtered': len(filtered_data),
        'unique': len(deduped_data),
        'train': len(splits['train']),
        'val': len(splits['val']),
        'test': len(splits['test']),
        'date': datetime.now().strftime("%Y-%m-%d"),
        'seed': args.seed
    }
    create_readme(args.output_dir, stats)
    
    logger.info("\n" + "="*70)
    logger.info("Data preparation complete!")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
