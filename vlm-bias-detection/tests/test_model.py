#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for model module.
"""

import pytest
import torch
from PIL import Image
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import BiasDetector


@pytest.fixture
def sample_image():
    """Create a sample image."""
    return Image.fromarray(
        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    )


@pytest.fixture
def sample_caption():
    """Create a sample caption."""
    return "A person working at a desk."


def test_bias_detector_parse_output():
    """Test output parsing."""
    detector = BiasDetector.__new__(BiasDetector)
    
    # Test biased output
    text1 = "This is biased.\nFinal Answer: Biased"
    result1 = detector._parse_output(text1, "")
    assert result1["bias_label"] == "Biased"
    
    # Test not biased output
    text2 = "This is not biased.\nFinal Answer: Not Biased"
    result2 = detector._parse_output(text2, "")
    assert result2["bias_label"] == "Not Biased"
    
    # Test unclear output
    text3 = "This is unclear."
    result3 = detector._parse_output(text3, "")
    assert result3["bias_label"] == "Unclear"


def test_format_prompt():
    """Test prompt formatting."""
    detector = BiasDetector.__new__(BiasDetector)
    
    caption = "Test caption"
    prompt = detector._format_prompt(caption)
    
    assert "Test caption" in prompt
    assert "Analyze" in prompt
    assert "gender bias" in prompt
    assert "Implication" in prompt
    assert "Bias Analysis" in prompt
    assert "Justification" in prompt
    assert "Final Answer" in prompt


def test_parse_output_variations():
    """Test various output formats."""
    detector = BiasDetector.__new__(BiasDetector)
    
    # Test different formats
    test_cases = [
        ("Final answer: biased", "Biased"),
        ("final answer: not biased", "Not Biased"),
        ("The caption is biased", "Biased"),
        ("This is not biased at all", "Not Biased"),
        ("Random text", "Unclear"),
        ("Final Answer: Biased", "Biased"),
        ("Final Answer: Not Biased", "Not Biased"),
    ]
    
    for text, expected in test_cases:
        result = detector._parse_output(text, "")
        assert result["bias_label"] == expected, f"Failed for: {text}"


def test_parse_output_with_reasoning():
    """Test parsing output with full reasoning."""
    detector = BiasDetector.__new__(BiasDetector)
    
    text = """Implication: The caption describes a professional setting.
Bias Analysis: No gender assumptions present.
Justification: Uses neutral language.
Final Answer: Not Biased"""
    
    result = detector._parse_output(text, "")
    assert result["bias_label"] == "Not Biased"
    assert "Implication" in result["reasoning"]
    assert "Justification" in result["reasoning"]


def test_parse_output_prompt_removal():
    """Test that prompt is removed from output."""
    detector = BiasDetector.__new__(BiasDetector)
    
    prompt = "Analyze this image and caption for gender bias."
    full_text = prompt + " This is the response. Final Answer: Biased"
    
    result = detector._parse_output(full_text, prompt)
    assert prompt not in result["reasoning"]
    assert "This is the response" in result["reasoning"]


def test_parse_output_edge_cases():
    """Test edge cases in output parsing."""
    detector = BiasDetector.__new__(BiasDetector)
    
    # Empty string
    result1 = detector._parse_output("", "")
    assert result1["bias_label"] == "Unclear"
    
    # Only whitespace
    result2 = detector._parse_output("   \n   ", "")
    assert result2["bias_label"] == "Unclear"
    
    # Multiple "Biased" mentions
    result3 = detector._parse_output("This could be biased or not biased. Final Answer: Biased", "")
    assert result3["bias_label"] == "Biased"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
