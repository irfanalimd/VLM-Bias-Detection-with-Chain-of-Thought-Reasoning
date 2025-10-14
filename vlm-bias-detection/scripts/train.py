#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tune LlamaV-o1 model for gender bias detection in image-caption pairs.
This script loads a pretrained LlamaV-o1 model and fine-tunes it on a dataset of 
image-caption pairs with bias labels and chain-of-thought reasoning.
"""

import os
import yaml
import logging
import argparse
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from transformers import DefaultDataCollator, get_linear_schedule_with_warmup
from PIL import Image
from transformers.trainer_utils import get_last_checkpoint
import numpy as np
import evaluate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune LlamaV-o1 for bias detection")
    parser.add_argument("--config", type=str, default="train_config.yaml",
                        help="Path to the training configuration file")
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load the training configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

class GenderBiasDataset(Dataset):
    """Dataset for gender bias detection in image-caption pairs."""
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer: transformers.PreTrainedTokenizer,
        image_processor,
        image_dir: str,
        max_length: int = 512,
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.image_dir = image_dir
        self.max_length = max_length
        
        # Load the data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load the data from a JSONL file."""
        data = []
        with open(self.data_path, 'r') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    data.append(item)
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse line: {line.strip()}")
        
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
            pixel_values = self.image_processor(image, return_tensors="pt").pixel_values[0]
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            # Use a blank image as fallback
            pixel_values = torch.zeros((3, 224, 224))
        
        # Prepare the text input and output
        caption = item["caption"]
        reasoning = item["reasoning"]
        bias_label = item["bias_label"]
        
        # Format the prompt and target - adjust as needed for the specific LlamaV-o1 format
        prompt = f"Analyze this image and caption for gender bias: \"{caption}\""
        target = f"{reasoning}\nFinal judgment: {bias_label}"
        
        # Tokenize the text
        prompt_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True, 
                                   max_length=self.max_length).input_ids[0]
        target_ids = self.tokenizer(target, return_tensors="pt", truncation=True,
                                   max_length=self.max_length).input_ids[0]
        
        # Combine prompt and target for causal LM training
        input_ids = torch.cat([prompt_ids, target_ids])
        
        # Create attention mask and labels
        attention_mask = torch.ones_like(input_ids)
        
        # For causal LM, we use the whole sequence as input_ids
        # but only compute loss on the target part, setting prompt tokens to -100
        labels = torch.cat([torch.full_like(prompt_ids, -100), target_ids])
        
        # Ensure all tensors have the same length by padding or truncation
        max_len = self.max_length
        if len(input_ids) > max_len:
            input_ids = input_ids[:max_len]
            attention_mask = attention_mask[:max_len]
            labels = labels[:max_len]
        else:
            padding_length = max_len - len(input_ids)
            input_ids = torch.cat([input_ids, torch.zeros(padding_length, dtype=torch.long)])
            attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=torch.long)])
            labels = torch.cat([labels, torch.full((padding_length,), -100, dtype=torch.long)])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }

def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    
    # For sequence classification, we're looking at the final token's prediction
    predictions = predictions.argmax(axis=-1)
    
    # Create a mask for the labels we care about (-100 is padding/ignored)
    mask = labels != -100
    
    # Filter out the padding tokens
    true_labels = [label[mask[i]] for i, label in enumerate(labels)]
    pred_labels = [pred[mask[i]] for i, pred in enumerate(predictions)]
    
    # Process to extract just the bias label
    
    # Compute metrics (example: accuracy)
    accuracy = evaluate.load("accuracy")
    result = accuracy.compute(predictions=predictions, references=labels, 
                              ignore_value=-100)
    
    # Add any other metrics you want to track
    return result

def main():
    """Main function to train the model."""
    args = setup_args()
    config = load_config(args.config)
    
    # Set seed for reproducibility
    transformers.set_seed(config.get("seed", 42))
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    
    # Ensure the tokenizer has a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # For LlamaV-o1, we need the image processor too
    # This is a simplified example; adjust for the actual LlamaV-o1 implementation
    from transformers import CLIPProcessor
    image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # Create the datasets
    train_dataset = GenderBiasDataset(
        data_path=config["train_data"],
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_dir=config["image_dir"],
        max_length=config.get("max_length", 512),
    )
    
    if "validation_data" in config:
        eval_dataset = GenderBiasDataset(
            data_path=config["validation_data"],
            tokenizer=tokenizer,
            image_processor=image_processor,
            image_dir=config["image_dir"],
            max_length=config.get("max_length", 512),
        )
    else:
        eval_dataset = None
    
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config.get("num_epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 4),
        per_device_eval_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        learning_rate=config.get("learning_rate", 5e-5),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_steps=config.get("warmup_steps", 100),
        logging_dir=os.path.join(config["output_dir"], "logs"),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 100),
        save_total_limit=config.get("save_total_limit", 2),
        evaluation_strategy="steps" if eval_dataset else "no",
        eval_steps=config.get("eval_steps", 100) if eval_dataset else None,
        load_best_model_at_end=True if eval_dataset else False,
        report_to=config.get("report_to", "tensorboard"),
        fp16=torch.cuda.is_available(),
        remove_unused_columns=False,  # Important for custom datasets
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics if eval_dataset else None,
        data_collator=DefaultDataCollator(),
    )
    
    # Check for existing checkpoint
    last_checkpoint = get_last_checkpoint(config["output_dir"])
    if last_checkpoint:
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")
        resume_from_checkpoint = last_checkpoint
    else:
        resume_from_checkpoint = None
    
    # Train the model
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save the final model
    trainer.save_model(os.path.join(config["output_dir"], "final"))
    
    # Save the tokenizer alongside the model
    tokenizer.save_pretrained(os.path.join(config["output_dir"], "final"))
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
