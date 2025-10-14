#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training utilities for bias detection model.
"""

import os
import logging
from typing import Dict, Optional, Any
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    CLIPProcessor,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

from src.dataset import BiasDetectionDataset

logger = logging.getLogger(__name__)


class BiasDetectionTrainer:
    """Trainer class for bias detection model."""
    
    def __init__(
        self,
        model_name: str = "mbzuai-oryx/LlamaV-o1-7B",
        train_data: str = "data/processed/train_annotations.jsonl",
        val_data: Optional[str] = "data/processed/val_annotations.jsonl",
        image_dir: str = "data/images",
        output_dir: str = "models/llamav_o1_bias_detector",
        learning_rate: float = 2e-5,
        num_epochs: int = 5,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_length: int = 512,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        fp16: bool = True,
        use_lora: bool = True,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        seed: int = 42,
        **kwargs
    ):
        """
        Initialize the trainer.
        
        Args:
            model_name: HuggingFace model name or path
            train_data: Path to training data
            val_data: Path to validation data
            image_dir: Directory containing images
            output_dir: Directory to save model
            learning_rate: Learning rate
            num_epochs: Number of training epochs
            batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            max_length: Maximum sequence length
            warmup_steps: Warmup steps for learning rate
            weight_decay: Weight decay
            fp16: Use mixed precision training
            use_lora: Use LoRA for efficient fine-tuning
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            seed: Random seed
            **kwargs: Additional arguments
        """
        self.model_name = model_name
        self.train_data = train_data
        self.val_data = val_data
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_length = max_length
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.fp16 = fp16 and torch.cuda.is_available()
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.seed = seed
        
        # Set random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Initialize components
        self.tokenizer = None
        self.image_processor = None
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        self.trainer = None
    
    def setup(self):
        """Set up tokenizer, model, and datasets."""
        logger.info("Setting up trainer...")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load image processor
        logger.info("Loading image processor")
        self.image_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        
        # Load model
        logger.info(f"Loading model from {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.fp16 else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Apply LoRA if requested
        if self.use_lora:
            logger.info("Applying LoRA")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                lora_dropout=self.lora_dropout,
                target_modules=["q_proj", "v_proj"],
                bias="none"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        
        # Create datasets
        logger.info("Loading training dataset")
        self.train_dataset = BiasDetectionDataset(
            data_path=self.train_data,
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            image_dir=self.image_dir,
            max_length=self.max_length,
            split="train"
        )
        
        if self.val_data:
            logger.info("Loading validation dataset")
            self.val_dataset = BiasDetectionDataset(
                data_path=self.val_data,
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                image_dir=self.image_dir,
                max_length=self.max_length,
                split="val"
            )
        
        logger.info("Setup complete")
    
    def train(self):
        """Train the model."""
        if self.model is None:
            self.setup()
        
        logger.info("Starting training...")
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Define training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            fp16=self.fp16,
            logging_dir=os.path.join(self.output_dir, "logs"),
            logging_steps=10,
            save_steps=100,
            save_total_limit=3,
            evaluation_strategy="steps" if self.val_dataset else "no",
            eval_steps=100 if self.val_dataset else None,
            load_best_model_at_end=True if self.val_dataset else False,
            metric_for_best_model="eval_loss" if self.val_dataset else None,
            report_to=["tensorboard"],
            remove_unused_columns=False,
            dataloader_pin_memory=True,
            dataloader_num_workers=4,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics if self.val_dataset else None,
        )
        
        # Train
        logger.info("Training started")
        train_result = self.trainer.train()
        
        # Save model
        logger.info("Saving model")
        self.save_model()
        
        # Log metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        logger.info("Training complete")
        return train_result
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        
        # Simple accuracy computation
        # For generation tasks, this is a simplified version
        accuracy_metric = evaluate.load("accuracy")
        
        # Mask padding tokens
        mask = labels != -100
        predictions_flat = predictions[mask]
        labels_flat = labels[mask]
        
        accuracy = accuracy_metric.compute(
            predictions=predictions_flat,
            references=labels_flat
        )
        
        return accuracy
    
    def save_model(self, save_path: Optional[str] = None):
        """Save the model and tokenizer."""
        if save_path is None:
            save_path = os.path.join(self.output_dir, "final")
        
        os.makedirs(save_path, exist_ok=True)
        
        logger.info(f"Saving model to {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info("Model saved successfully")
    
    def evaluate(self):
        """Evaluate the model."""
        if self.trainer is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.val_dataset is None:
            raise ValueError("No validation dataset provided")
        
        logger.info("Evaluating model...")
        metrics = self.trainer.evaluate()
        
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)
        
        return metrics
