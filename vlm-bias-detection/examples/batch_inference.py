#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Batch inference example for processing multiple images.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import BiasDetector
from PIL import Image


def load_data(input_file: str) -> List[Dict[str, str]]:
    """Load image-caption pairs from JSONL file."""
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError:
                continue
    return data


def process_batch(
    detector: BiasDetector,
    data: List[Dict[str, str]],
    image_dir: str,
    batch_size: int = 8,
    verbose: bool = True
) -> List[Dict[str, str]]:
    """Process data in batches."""
    results = []
    total_time = 0
    successful = 0
    failed = 0
    
    print(f"\nProcessing {len(data)} samples in batches of {batch_size}...")
    
    for i in tqdm(range(0, len(data), batch_size), desc="Processing batches", disable=not verbose):
        batch = data[i:i+batch_size]
        
        for item in batch:
            image_id = item["image_id"]
            caption = item["caption"]
            image_path = os.path.join(image_dir, image_id)
            
            if not os.path.exists(image_path):
                if verbose:
                    tqdm.write(f"⚠ Warning: Image not found: {image_path}")
                failed += 1
                continue
            
            try:
                # Load image
                image = Image.open(image_path).convert("RGB")
                
                # Run inference with timing
                start_time = time.time()
                result = detector.analyze(image, caption)
                inference_time = time.time() - start_time
                
                total_time += inference_time
                successful += 1
                
                # Store result
                results.append({
                    "image_id": image_id,
                    "caption": caption,
                    "bias_label": result["bias_label"],
                    "reasoning": result["reasoning"],
                    "inference_time_ms": inference_time * 1000
                })
                
            except Exception as e:
                if verbose:
                    tqdm.write(f"✗ Error processing {image_id}: {e}")
                failed += 1
                continue
    
    # Print statistics
    if verbose and successful > 0:
        avg_time = (total_time / successful) * 1000  # Convert to ms
        print(f"\n✓ Successfully processed: {successful}")
        print(f"✗ Failed: {failed}")
        print(f"⏱  Average inference time: {avg_time:.2f}ms")
        print(f"⏱  Total time: {total_time:.2f}s")
    
    return results


def save_results(results: List[Dict[str, str]], output_file: str):
    """Save results to JSONL file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"\n✓ Saved {len(results)} results to {output_file}")


def print_summary(results: List[Dict[str, str]]):
    """Print summary statistics."""
    total = len(results)
    if total == 0:
        print("\n⚠ No results to summarize")
        return
    
    biased_count = sum(1 for r in results if r["bias_label"] == "Biased")
    not_biased_count = sum(1 for r in results if r["bias_label"] == "Not Biased")
    unclear_count = sum(1 for r in results if r["bias_label"] == "Unclear")
    
    # Calculate inference time statistics
    times = [r.get("inference_time_ms", 0) for r in results]
    avg_time = sum(times) / len(times) if times else 0
    min_time = min(times) if times else 0
    max_time = max(times) if times else 0
    
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    print(f"Total processed: {total}")
    print(f"\nBias Labels:")
    print(f"  Biased:     {biased_count:4d} ({biased_count/total*100:5.1f}%)")
    print(f"  Not Biased: {not_biased_count:4d} ({not_biased_count/total*100:5.1f}%)")
    print(f"  Unclear:    {unclear_count:4d} ({unclear_count/total*100:5.1f}%)")
    
    if avg_time > 0:
        print(f"\nInference Time:")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Min:     {min_time:.2f}ms")
        print(f"  Max:     {max_time:.2f}ms")
    
    print("=" * 70)


def show_sample_results(results: List[Dict[str, str]], n_samples: int = 3):
    """Show sample results."""
    print("\n" + "=" * 70)
    print("SAMPLE RESULTS")
    print("=" * 70)
    
    for i, result in enumerate(results[:n_samples], 1):
        print(f"\n[Sample {i}]")
        print(f"Image:   {result['image_id']}")
        print(f"Caption: {result['caption']}")
        print(f"Label:   {result['bias_label']}")
        
        # Show first 200 characters of reasoning
        reasoning = result['reasoning']
        if len(reasoning) > 200:
            reasoning = reasoning[:200] + "..."
        print(f"Reasoning: {reasoning}")
        
        if 'inference_time_ms' in result:
            print(f"Time:    {result['inference_time_ms']:.2f}ms")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Batch inference for bias detection"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing images"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/llamav_o1_bias_detector/final",
        help="Path to model"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu)"
    )
    parser.add_argument(
        "--show_samples",
        type=int,
        default=3,
        help="Number of sample results to display"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    if verbose:
        print("=" * 70)
        print("VLM Bias Detection - Batch Inference")
        print("=" * 70)
    
    # Load data
    if verbose:
        print(f"\n[1/3] Loading data from {args.input}...")
    data = load_data(args.input)
    
    if args.max_samples:
        data = data[:args.max_samples]
    
    if verbose:
        print(f"✓ Loaded {len(data)} samples")
    
    # Load model
    if verbose:
        print(f"\n[2/3] Loading model from {args.model_path}...")
    try:
        detector = BiasDetector.from_pretrained(
            args.model_path,
            device=args.device
        )
        if verbose:
            print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nPlease ensure:")
        print("1. Model path is correct")
        print("2. Model files exist")
        print("3. You have sufficient GPU/CPU memory")
        return
    
    # Process data
    if verbose:
        print(f"\n[3/3] Processing {len(data)} samples...")
    results = process_batch(
        detector,
        data,
        args.image_dir,
        args.batch_size,
        verbose=verbose
    )
    
    if len(results) == 0:
        print("\n✗ No results generated. Check errors above.")
        return
    
    # Save results
    save_results(results, args.output)
    
    # Print summary
    if verbose:
        print_summary(results)
        
        # Show sample results
        if args.show_samples > 0:
            show_sample_results(results, args.show_samples)
        
        print("\n" + "=" * 70)
        print("Batch inference complete!")
        print("\nNext steps:")
        print(f"1. Review results: cat {args.output}")
        print(f"2. Analyze results: python evaluation/analyze_results.py --results {args.output}")
        print(f"3. Compare with baseline: python scripts/evaluate_bias.py")
        print("=" * 70)


if __name__ == "__main__":
    main()
