#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quick start example for bias detection.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import BiasDetector
from PIL import Image
import numpy as np
import requests
from io import BytesIO


def download_image(url: str) -> Image.Image:
    """Download an image from URL."""
    response = requests.get(url, timeout=10)
    return Image.open(BytesIO(response.content)).convert("RGB")


def create_sample_image() -> Image.Image:
    """Create a sample image for demonstration."""
    # Create a simple gradient image
    return Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )


def main():
    """Run quick start examples."""
    print("=" * 70)
    print("VLM Bias Detection - Quick Start")
    print("=" * 70)
    
    # Load model
    print("\n[1/4] Loading model...")
    print("Note: For first run, you need a fine-tuned model or use base LLaMA-V-O1")
    
    # Use base model for demo (replace with your fine-tuned model path)
    model_path = "models/llamav_o1_bias_detector/final"
    
    # For testing without a trained model, you can use the base model
    # model_path = "mbzuai-oryx/LlamaV-o1-7B"
    
    try:
        detector = BiasDetector.from_pretrained(model_path)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nTo use this example:")
        print("1. Fine-tune a model using: python scripts/train.py")
        print("2. Or specify path to a pretrained model")
        print("3. Or download our pretrained model from HuggingFace")
        return
    
    # Example 1: Potentially biased caption
    print("\n[2/4] Example 1: Analyzing potentially biased caption...")
    print("-" * 70)
    
    caption1 = "A woman in the kitchen preparing dinner for her family."
    print(f"Caption: {caption1}")
    
    # Create a sample image for demonstration
    print("\nNote: Using sample image for demonstration")
    sample_image1 = create_sample_image()
    
    try:
        result1 = detector.analyze(sample_image1, caption1)
        print(f"\n✓ Analysis complete!")
        print(f"\nBias Label: {result1['bias_label']}")
        print(f"\nReasoning:")
        # Print first 500 characters of reasoning
        reasoning_preview = result1['reasoning'][:500]
        if len(result1['reasoning']) > 500:
            reasoning_preview += "..."
        print(reasoning_preview)
    except Exception as e:
        print(f"✗ Error during analysis: {e}")
    
    # Example 2: Non-biased caption
    print("\n" + "=" * 70)
    print("[3/4] Example 2: Analyzing non-biased caption...")
    print("-" * 70)
    
    caption2 = "Two healthcare workers collaborating on patient care in a hospital."
    print(f"Caption: {caption2}")
    
    sample_image2 = create_sample_image()
    
    try:
        result2 = detector.analyze(sample_image2, caption2)
        print(f"\n✓ Analysis complete!")
        print(f"\nBias Label: {result2['bias_label']}")
        print(f"\nReasoning:")
        reasoning_preview = result2['reasoning'][:500]
        if len(result2['reasoning']) > 500:
            reasoning_preview += "..."
        print(reasoning_preview)
    except Exception as e:
        print(f"✗ Error during analysis: {e}")
    
    # Example 3: Batch analysis
    print("\n" + "=" * 70)
    print("[4/4] Example 3: Batch analysis...")
    print("-" * 70)
    
    captions = [
        "A male surgeon performing surgery.",
        "A person working at a desk.",
        "A female nurse helping patients."
    ]
    
    print(f"Analyzing {len(captions)} captions...")
    
    images = [create_sample_image() for _ in range(len(captions))]
    
    try:
        results = detector.batch_analyze(images, captions, batch_size=2)
        
        print(f"\n✓ Batch analysis complete!")
        print(f"\nResults:")
        for i, (caption, result) in enumerate(zip(captions, results), 1):
            print(f"\n{i}. Caption: {caption}")
            print(f"   Label: {result['bias_label']}")
    except Exception as e:
        print(f"✗ Error during batch analysis: {e}")
    
    print("\n" + "=" * 70)
    print("Quick start complete!")
    print("\nNext steps:")
    print("1. Try with your own images: detector.analyze(Image.open('path.jpg'), 'caption')")
    print("2. Process multiple images: detector.batch_analyze(images, captions)")
    print("3. Train on custom data: python scripts/train.py --config config/train_config.yaml")
    print("4. Run evaluation: python scripts/evaluate_bias.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
