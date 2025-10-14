#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for bias detection.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml

import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        output_path: Path to save config
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from JSONL file.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Could not parse JSON - {e}")
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """
    Save data to JSONL file.
    
    Args:
        data: List of dictionaries
        file_path: Path to save file
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def setup_logging(
    log_file: Optional[str] = None,
    log_level: str = "INFO"
):
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        log_level: Logging level
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params
    }


def get_device() -> str:
    """
    Get the best available device.
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed
    """
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_time(seconds: float) -> str:
    """
    Format seconds to human-readable time.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def ensure_dir(directory: str):
    """
    Ensure directory exists, create if not.
    
    Args:
        directory: Directory path
    """
    os.makedirs(directory, exist_ok=True)


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from path.
    
    Args:
        image_path: Path to image
        
    Returns:
        PIL Image
    """
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        raise


def validate_image(image: Image.Image, max_size: tuple = (4096, 4096)) -> bool:
    """
    Validate image dimensions.
    
    Args:
        image: PIL Image
        max_size: Maximum allowed size (width, height)
        
    Returns:
        True if valid, False otherwise
    """
    width, height = image.size
    max_width, max_height = max_size
    
    if width > max_width or height > max_height:
        logger.warning(
            f"Image too large: {width}x{height}. "
            f"Maximum: {max_width}x{max_height}"
        )
        return False
    
    return True


def resize_image(
    image: Image.Image,
    max_size: int = 1024
) -> Image.Image:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: PIL Image
        max_size: Maximum dimension
        
    Returns:
        Resized PIL Image
    """
    width, height = image.size
    
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return image


def calculate_metrics(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        predictions: Predicted labels
        references: Ground truth labels
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        confusion_matrix
    )
    
    accuracy = accuracy_score(references, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        references,
        predictions,
        average='binary',
        pos_label='Biased',
        zero_division=0
    )
    
    cm = confusion_matrix(
        references,
        predictions,
        labels=['Biased', 'Not Biased']
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }


def print_metrics(metrics: Dict[str, Any], prefix: str = ""):
    """
    Print metrics in a formatted way.
    
    Args:
        metrics: Metrics dictionary
        prefix: Prefix for metric names
    """
    print(f"\n{'='*50}")
    if prefix:
        print(f"{prefix} Metrics")
        print(f"{'='*50}")
    
    for key, value in metrics.items():
        if key == 'confusion_matrix':
            print(f"\nConfusion Matrix:")
            print(f"  {value}")
        elif isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print(f"{'='*50}\n")


def log_inference_metrics(
    result: Dict[str, Any],
    latency_ms: float,
    log_file: str = "logs/inference_metrics.jsonl"
):
    """
    Log inference metrics to file.
    
    Args:
        result: Inference result
        latency_ms: Inference latency in milliseconds
        log_file: Path to log file
    """
    import time
    
    log_entry = {
        "timestamp": time.time(),
        "bias_label": result.get("bias_label"),
        "latency_ms": latency_ms,
        "reasoning_length": len(result.get("reasoning", "")),
    }
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


def get_gpu_memory_usage() -> Dict[str, float]:
    """
    Get GPU memory usage.
    
    Returns:
        Dictionary with memory usage in GB
    """
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "free": 0}
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    free = total_memory - allocated
    
    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "free_gb": free,
        "total_gb": total_memory
    }


def parse_bias_label(text: str) -> str:
    """
    Parse bias label from generated text.
    
    Args:
        text: Generated text
        
    Returns:
        Bias label ('Biased', 'Not Biased', or 'Unclear')
    """
    text_lower = text.lower()
    
    # Check for explicit final answer
    if "final answer: biased" in text_lower:
        return "Biased"
    elif "final answer: not biased" in text_lower:
        return "Not Biased"
    
    # Check last line
    lines = text.split('\n')
    if lines:
        last_line = lines[-1].lower()
        if "biased" in last_line and "not biased" not in last_line:
            return "Biased"
        elif "not biased" in last_line:
            return "Not Biased"
    
    # Search full text
    if "biased" in text_lower and "not biased" not in text_lower:
        return "Biased"
    elif "not biased" in text_lower:
        return "Not Biased"
    
    return "Unclear"


def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries.
    
    Args:
        *dicts: Variable number of dictionaries
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        result.update(d)
    return result


def filter_dict(d: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """
    Filter dictionary to only include specified keys.
    
    Args:
        d: Input dictionary
        keys: Keys to keep
        
    Returns:
        Filtered dictionary
    """
    return {k: v for k, v in d.items() if k in keys}


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary.
    
    Args:
        d: Input dictionary
        parent_key: Parent key for recursion
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks.
    
    Args:
        lst: Input list
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def batch_generator(data: List[Any], batch_size: int):
    """
    Generate batches from data.
    
    Args:
        data: Input data
        batch_size: Size of each batch
        
    Yields:
        Batches of data
    """
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def get_file_size(file_path: str) -> str:
    """
    Get human-readable file size.
    
    Args:
        file_path: Path to file
        
    Returns:
        Formatted file size string
    """
    size = os.path.getsize(file_path)
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    
    return f"{size:.2f} PB"


def create_directory_structure(base_dir: str):
    """
    Create standard project directory structure.
    
    Args:
        base_dir: Base directory path
    """
    dirs = [
        "data/images",
        "data/processed",
        "models",
        "results",
        "logs",
        "evaluation/results",
        "checkpoints"
    ]
    
    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)
    
    logger.info(f"Created directory structure in {base_dir}")


def check_file_exists(file_path: str, raise_error: bool = False) -> bool:
    """
    Check if file exists.
    
    Args:
        file_path: Path to file
        raise_error: Whether to raise error if file doesn't exist
        
    Returns:
        True if file exists, False otherwise
        
    Raises:
        FileNotFoundError: If file doesn't exist and raise_error is True
    """
    exists = os.path.exists(file_path)
    
    if not exists and raise_error:
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return exists


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value to return if denominator is zero
        
    Returns:
        Result of division or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def percentage(part: float, whole: float, decimals: int = 2) -> float:
    """
    Calculate percentage.
    
    Args:
        part: Part value
        whole: Whole value
        decimals: Number of decimal places
        
    Returns:
        Percentage value
    """
    if whole == 0:
        return 0.0
    return round((part / whole) * 100, decimals)


def remove_duplicates(lst: List[Any], key=None) -> List[Any]:
    """
    Remove duplicates from list while preserving order.
    
    Args:
        lst: Input list
        key: Optional key function for comparison
        
    Returns:
        List with duplicates removed
    """
    if key is None:
        seen = set()
        result = []
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    else:
        seen = set()
        result = []
        for item in lst:
            k = key(item)
            if k not in seen:
                seen.add(k)
                result.append(item)
        return result


def dict_to_list(d: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert dictionary to list of key-value dictionaries.
    
    Args:
        d: Input dictionary
        
    Returns:
        List of dictionaries with 'key' and 'value' fields
    """
    return [{"key": k, "value": v} for k, v in d.items()]


def is_valid_json(text: str) -> bool:
    """
    Check if string is valid JSON.
    
    Args:
        text: Input string
        
    Returns:
        True if valid JSON, False otherwise
    """
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def pretty_print_json(data: Any, indent: int = 2):
    """
    Pretty print JSON data.
    
    Args:
        data: Data to print
        indent: Indentation level
    """
    print(json.dumps(data, indent=indent, ensure_ascii=False))


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_path: str
):
    """
    Save training checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        checkpoint_path: Path to save checkpoint
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        model: PyTorch model
        optimizer: Optional optimizer
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    return checkpoint


def format_number(num: float, decimals: int = 2) -> str:
    """
    Format number with thousands separator.
    
    Args:
        num: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted number string
    """
    return f"{num:,.{decimals}f}"


def get_timestamp() -> str:
    """
    Get current timestamp as string.
    
    Returns:
        Timestamp string in format YYYY-MM-DD_HH-MM-SS
    """
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def create_experiment_dir(base_dir: str = "experiments") -> str:
    """
    Create experiment directory with timestamp.
    
    Args:
        base_dir: Base directory for experiments
        
    Returns:
        Path to created experiment directory
    """
    timestamp = get_timestamp()
    exp_dir = os.path.join(base_dir, timestamp)
    os.makedirs(exp_dir, exist_ok=True)
    logger.info(f"Created experiment directory: {exp_dir}")
    return exp_dir
