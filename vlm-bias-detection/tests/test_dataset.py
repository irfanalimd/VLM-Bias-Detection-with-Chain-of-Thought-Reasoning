#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for dataset module.
"""

import os
import json
import tempfile
import pytest
import torch
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, CLIPProcessor

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import BiasDetectionDataset, InferenceDataset


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    return [
        {
            "image_id": "test_001.jpg",
            "caption": "A man working at a desk.",
            "reasoning": "The caption is descriptive without bias.",
            "bias_label": "Not Biased"
        },
        {
            "image_id": "test_002.jpg",
            "caption": "A woman cooking in the kitchen.",
            "reasoning": "Reinforces gender stereotypes.",
            "bias_label": "Biased"
        }
    ]


@pytest.fixture
def temp_data_dir(sample_data):
    """Create temporary directory with test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create JSONL file
        data_file = os.path.join(tmpdir, "test_data.jsonl")
        with open(data_file, 'w') as f:
            for item in sample_data:
                f.write(json.dumps(item) + '\n')
        
        # Create dummy images
        img_dir = os.path.join(tmpdir, "images")
        os.makedirs(img_dir)
        
        for item in sample_data:
            img = Image.fromarray(
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            )
            img.save(os.path.join(img_dir, item["image_id"]))
        
        yield tmpdir, data_file, img_dir


def test_bias_detection_dataset_init(temp_data_dir):
    """Test dataset initialization."""
    tmpdir, data_file, img_dir = temp_data_dir
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    dataset = BiasDetectionDataset(
        data_path=data_file,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_dir=img_dir,
        max_length=512
    )
    
    assert len(dataset) == 2
    assert dataset.max_length == 512


def test_bias_detection_dataset_getitem(temp_data_dir):
    """Test getting items from dataset."""
    tmpdir, data_file, img_dir = temp_data_dir
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    dataset = BiasDetectionDataset(
        data_path=data_file,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_dir=img_dir,
        max_length=512
    )
    
    item = dataset[0]
    
    # Check keys
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "labels" in item
    assert "pixel_values" in item
    
    # Check types
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["attention_mask"], torch.Tensor)
    assert isinstance(item["labels"], torch.Tensor)
    assert isinstance(item["pixel_values"], torch.Tensor)
    
    # Check shapes
    assert item["input_ids"].shape[0] == 512
    assert item["attention_mask"].shape[0] == 512
    assert item["labels"].shape[0] == 512


def test_inference_dataset(temp_data_dir):
    """Test inference dataset."""
    tmpdir, data_file, img_dir = temp_data_dir
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    dataset = InferenceDataset(
        data_path=data_file,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_dir=img_dir,
        max_length=512
    )
    
    assert len(dataset) == 2
    
    item = dataset[0]
    
    # Check keys (no labels for inference)
    assert "input_ids" in item
    assert "attention_mask" in item
    assert "pixel_values" in item
    assert "image_id" in item
    assert "caption" in item
    assert "labels" not in item


def test_dataset_with_missing_image(temp_data_dir):
    """Test dataset handles missing images gracefully."""
    tmpdir, data_file, img_dir = temp_data_dir
    
    # Remove one image
    os.remove(os.path.join(img_dir, "test_001.jpg"))
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    dataset = BiasDetectionDataset(
        data_path=data_file,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_dir=img_dir,
        max_length=512
    )
    
    # Should return fallback (zeros) for missing image
    item = dataset[0]
    assert item["pixel_values"].shape == torch.Size([3, 224, 224])


def test_dataset_empty_file():
    """Test dataset with empty file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = os.path.join(tmpdir, "empty.jsonl")
        with open(data_file, 'w') as f:
            pass  # Empty file
        
        img_dir = os.path.join(tmpdir, "images")
        os.makedirs(img_dir)
        
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        dataset = BiasDetectionDataset(
            data_path=data_file,
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_dir=img_dir,
            max_length=512
        )
        
        assert len(dataset) == 0


def test_dataset_prompt_format(temp_data_dir):
    """Test prompt formatting."""
    tmpdir, data_file, img_dir = temp_data_dir
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    dataset = BiasDetectionDataset(
        data_path=data_file,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_dir=img_dir,
        max_length=512
    )
    
    # Check that prompt contains expected keywords
    prompt = dataset._format_prompt("Test caption")
    assert "Analyze" in prompt
    assert "gender bias" in prompt
    assert "Test caption" in prompt
    assert "Implication" in prompt
    assert "Justification" in prompt


def test_dataset_target_format(temp_data_dir):
    """Test target formatting."""
    tmpdir, data_file, img_dir = temp_data_dir
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    dataset = BiasDetectionDataset(
        data_path=data_file,
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_dir=img_dir,
        max_length=512
    )
    
    # Check that target contains expected format
    target = dataset._format_target("Test reasoning", "Biased")
    assert "Test reasoning" in target
    assert "Final Answer: Biased" in target


def test_collate_fn():
    """Test collate function."""
    from src.dataset import collate_fn
    
    batch = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([1, 2, 3]),
            "pixel_values": torch.randn(3, 224, 224)
        },
        {
            "input_ids": torch.tensor([4, 5, 6]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([4, 5, 6]),
            "pixel_values": torch.randn(3, 224, 224)
        }
    ]
    
    result = collate_fn(batch)
    
    assert result["input_ids"].shape[0] == 2
    assert result["attention_mask"].shape[0] == 2
    assert result["labels"].shape[0] == 2
    assert result["pixel_values"].shape[0] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
