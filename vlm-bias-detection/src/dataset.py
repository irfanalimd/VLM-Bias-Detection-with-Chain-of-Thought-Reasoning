#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Dataset classes for bias detection in vision-language models.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import transformers

logger = logging.getLogger(__name__)


class BiasDetectionDataset(Dataset):
    """Dataset for gender bias detection in image-caption pairs."""
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer: transformers.PreTrainedTokenizer,
        image_processor: Any,
        image_dir: str,
        max_length: int = 512,
        split: str = "train"
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to JSONL file with annotations
            tokenizer: Hugging Face tokenizer
            image_processor: Image preprocessing function/model
            image_dir: Directory containing images
            max_length: Maximum sequence length
            split: Dataset split (train/val/test)
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_dir = image_dir
        self.max_length = max_length
        self.split = split
        
        # Load the data
        self.data = self._load_data()
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Could not parse JSON - {e}")
        
        logger.info(f"Loaded {len(data)} samples from {self.data_path}")
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        item = self.data[idx]
        
        # Load and process the image
        image_path = os.path.join(self.image_dir, item["image_id"])
        try:
            image = Image.open(image_path).convert("RGB")
            if hasattr(self.image_processor, 'preprocess'):
                pixel_values = self.image_processor.preprocess(
                    image, return_tensors="pt"
                )["pixel_values"][0]
            else:
                pixel_values = self.image_processor(
                    image, return_tensors="pt"
                ).pixel_values[0]
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            # Use blank image as fallback
            pixel_values = torch.zeros((3, 224, 224))
        
        # Prepare text
        caption = item["caption"]
        reasoning = item.get("reasoning", "")
        bias_label = item.get("bias_label", "Not Biased")
        
        # Format prompt and target
        prompt = self._format_prompt(caption)
        target = self._format_target(reasoning, bias_label)
        
        # Tokenize
        prompt_ids = self.tokenizer(
            prompt, 
            truncation=True, 
            max_length=self.max_length // 2,
            add_special_tokens=True
        ).input_ids
        
        target_ids = self.tokenizer(
            target,
            truncation=True,
            max_length=self.max_length // 2,
            add_special_tokens=False
        ).input_ids
        
        # Combine for causal LM training
        input_ids = prompt_ids + target_ids
        
        # Create labels (mask prompt tokens)
        labels = [-100] * len(prompt_ids) + target_ids
        
        # Pad to max_length
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            labels = labels + [-100] * padding_length
        else:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
        
        # Create attention mask
        attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 
                         for token_id in input_ids]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixel_values,
        }
    
    def _format_prompt(self, caption: str) -> str:
        """Format the input prompt."""
        return (
            f"Analyze this image and caption for gender bias.\n"
            f"Caption: \"{caption}\"\n\n"
            f"Provide your analysis following these steps:\n"
            f"1. Implication: What does the caption imply?\n"
            f"2. Bias Analysis: Does it contain gender-based assumptions?\n"
            f"3. Justification: Explain your reasoning.\n"
            f"4. Final Answer: Biased or Not Biased\n\n"
            f"Analysis:\n"
        )
    
    def _format_target(self, reasoning: str, bias_label: str) -> str:
        """Format the target output."""
        return f"{reasoning}\n\nFinal Answer: {bias_label}"


class InferenceDataset(Dataset):
    """Simplified dataset for inference without labels."""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        image_processor: Any,
        image_dir: str,
        max_length: int = 512
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_dir = image_dir
        self.max_length = max_length
        
        self.data = self._load_data()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError:
                    continue
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample for inference."""
        item = self.data[idx]
        
        # Load and process image
        image_path = os.path.join(self.image_dir, item["image_id"])
        try:
            image = Image.open(image_path).convert("RGB")
            if hasattr(self.image_processor, 'preprocess'):
                pixel_values = self.image_processor.preprocess(
                    image, return_tensors="pt"
                )["pixel_values"][0]
            else:
                pixel_values = self.image_processor(
                    image, return_tensors="pt"
                ).pixel_values[0]
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            pixel_values = torch.zeros((3, 224, 224))
        
        # Prepare prompt
        caption = item["caption"]
        prompt = (
            f"Analyze this image and caption for gender bias.\n"
            f"Caption: \"{caption}\"\n\n"
            f"Provide your analysis following these steps:\n"
            f"1. Implication: What does the caption imply?\n"
            f"2. Bias Analysis: Does it contain gender-based assumptions?\n"
            f"3. Justification: Explain your reasoning.\n"
            f"4. Final Answer: Biased or Not Biased\n\n"
            f"Analysis:\n"
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "pixel_values": pixel_values,
            "image_id": item["image_id"],
            "caption": caption
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]) if "labels" in batch[0] else None,
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
    }
