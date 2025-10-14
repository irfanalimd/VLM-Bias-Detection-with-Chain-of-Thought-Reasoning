#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VLM Bias Detection Package

A toolkit for detecting and explaining social biases in vision-language models
using Chain-of-Thought reasoning.
"""

from src.model import BiasDetector
from src.dataset import BiasDetectionDataset, InferenceDataset
from src.trainer import BiasDetectionTrainer

__all__ = [
    "BiasDetector",
    "BiasDetectionDataset", 
    "InferenceDataset",
    "BiasDetectionTrainer",
]
