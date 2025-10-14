#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model wrapper for bias detection.
"""

import os
import logging
from typing import Dict, List, Optional, Union
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    CLIPProcessor
)

logger = logging.getLogger(__name__)


class BiasDetector:
    """Wrapper class for bias detection model."""
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        image_processor: CLIPProcessor,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_new_tokens: int = 512
    ):
        """
        Initialize the bias detector.
        
        Args:
            model: The language model
            tokenizer: The tokenizer
            image_processor: The image processor
            device: Device to run on
            max_new_tokens: Maximum tokens to generate
        """
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device
        self.max_new_tokens = max_new_tokens
        
        self.model.to(device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        **kwargs
    ) -> "BiasDetector":
        """
        Load a pretrained bias detection model.
        
        Args:
            model_path: Path to model directory or HF model ID
            device: Device to load on
            torch_dtype: Data type for model weights
            **kwargs: Additional arguments
            
        Returns:
            BiasDetector instance
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if torch_dtype is None:
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
        
        logger.info(f"Loading model from {model_path}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
            **kwargs
        )
        
        # Load image processor
        image_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        
        logger.info("Model loaded successfully")
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            device=device,
            max_new_tokens=kwargs.get("max_new_tokens", 512)
        )
    
    def analyze(
        self,
        image: Union[str, Image.Image],
        caption: str,
        return_full_output: bool = False
    ) -> Dict[str, str]:
        """
        Analyze an image-caption pair for bias.
        
        Args:
            image: PIL Image or path to image
            caption: The caption text
            return_full_output: Whether to return full generation
            
        Returns:
            Dictionary with reasoning and bias_label
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Process image
        pixel_values = self.image_processor(
            image, return_tensors="pt"
        ).pixel_values.to(self.device)
        
        # Format prompt
        prompt = self._format_prompt(caption)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        
        # Extract reasoning and label
        result = self._parse_output(generated_text, prompt)
        
        if return_full_output:
            result["full_output"] = generated_text
        
        return result
    
    def batch_analyze(
        self,
        images: List[Union[str, Image.Image]],
        captions: List[str],
        batch_size: int = 4
    ) -> List[Dict[str, str]]:
        """
        Analyze multiple image-caption pairs.
        
        Args:
            images: List of images
            captions: List of captions
            batch_size: Batch size for processing
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_captions = captions[i:i+batch_size]
            
            for image, caption in zip(batch_images, batch_captions):
                result = self.analyze(image, caption)
                results.append(result)
        
        return results
    
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
    
    def _parse_output(
        self, 
        generated_text: str, 
        prompt: str
    ) -> Dict[str, str]:
        """Parse the model output to extract reasoning and label."""
        # Remove prompt from output
        if generated_text.startswith(prompt):
            reasoning = generated_text[len(prompt):].strip()
        else:
            reasoning = generated_text.strip()
        
        # Extract bias label
        bias_label = "Unclear"
        
        # Check last line first
        last_line = reasoning.split("\n")[-1].lower()
        if "biased" in last_line and "not biased" not in last_line:
            bias_label = "Biased"
        elif "not biased" in last_line:
            bias_label = "Not Biased"
        else:
            # Search in full text
            reasoning_lower = reasoning.lower()
            if "final answer: biased" in reasoning_lower:
                bias_label = "Biased"
            elif "final answer: not biased" in reasoning_lower:
                bias_label = "Not Biased"
            elif "biased" in reasoning_lower and "not biased" not in reasoning_lower:
                bias_label = "Biased"
            elif "not biased" in reasoning_lower:
                bias_label = "Not Biased"
