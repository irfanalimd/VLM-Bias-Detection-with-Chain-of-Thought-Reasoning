#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for metrics module.
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import (
    count_gendered_terms,
    compute_positive_trait_attribution,
    compute_bias_disparity,
    compute_gender_bias_score,
    analyze_reasoning_content,
    analyze_model_response_consistency
)


def test_count_gendered_terms():
    """Test counting gendered terms."""
    # Test male terms
    text1 = "He is a man who works hard."
    counts1 = count_gendered_terms(text1)
    assert counts1["male"] > 0
    assert counts1["female"] == 0
    
    # Test female terms
    text2 = "She is a woman who leads her team."
    counts2 = count_gendered_terms(text2)
    assert counts2["female"] > 0
    assert counts2["male"] == 0
    
    # Test neutral text
    text3 = "The person works at a desk."
    counts3 = count_gendered_terms(text3)
    assert counts3["male"] == 0
    assert counts3["female"] == 0
    
    # Test mixed terms
    text4 = "He and she work together."
    counts4 = count_gendered_terms(text4)
    assert counts4["male"] > 0
    assert counts4["female"] > 0


def test_compute_positive_trait_attribution():
    """Test positive trait attribution."""
    # Test with male terms and positive traits
    text1 = "He is intelligent and capable."
    count1 = compute_positive_trait_attribution(text1, "male")
    assert count1 > 0
    
    # Test with female terms and positive traits
    text2 = "She is a brilliant leader."
    count2 = compute_positive_trait_attribution(text2, "female")
    assert count2 > 0
    
    # Test without traits
    text3 = "He works at the office."
    count3 = compute_positive_trait_attribution(text3, "male")
    assert count3 == 0
    
    # Test with distant trait (should not match)
    text4 = "He went to the store and later someone said he was intelligent"
    count4 = compute_positive_trait_attribution(text4, "male")
    # This may or may not match depending on word distance


def test_compute_bias_disparity():
    """Test bias disparity computation."""
    # Test with equal bias
    male_responses = ["Biased", "Biased", "Not Biased"]
    female_responses = ["Biased", "Biased", "Not Biased"]
    
    result = compute_bias_disparity(male_responses, female_responses)
    
    assert "disparity" in result
    assert "male_bias_rate" in result
    assert "female_bias_rate" in result
    assert result["disparity"] == 0.0
    
    # Test with different bias rates
    male_responses2 = ["Biased", "Biased", "Biased"]
    female_responses2 = ["Not Biased", "Not Biased", "Not Biased"]
    
    result2 = compute_bias_disparity(male_responses2, female_responses2)
    assert result2["disparity"] > 0
    assert result2["male_bias_rate"] == 1.0
    assert result2["female_bias_rate"] == 0.0


def test_compute_bias_disparity_empty():
    """Test bias disparity with empty inputs."""
    result = compute_bias_disparity([], [])
    assert result["disparity"] == 0.0
    assert result["significance"] is None


def test_analyze_reasoning_content():
    """Test reasoning content analysis."""
    male_reasoning = [
        "He is a competent professional.",
        "The man shows leadership."
    ]
    female_reasoning = [
        "She is a capable worker.",
        "The woman demonstrates skill."
    ]
    
    result = analyze_reasoning_content(male_reasoning, female_reasoning)
    
    assert "avg_male_terms" in result
    assert "avg_female_terms" in result
    assert "term_ratio" in result
    assert "avg_male_positive_traits" in result
    assert "avg_female_positive_traits" in result
    assert "trait_ratio" in result


def test_compute_gender_bias_score():
    """Test gender bias score computation."""
    results = [
        {
            "male_bias_label": "Biased",
            "female_bias_label": "Not Biased",
            "male_reasoning": "He is a doctor.",
            "female_reasoning": "She is a nurse."
        },
        {
            "male_bias_label": "Not Biased",
            "female_bias_label": "Not Biased",
            "male_reasoning": "He works hard.",
            "female_reasoning": "She works hard."
        }
    ]
    
    score = compute_gender_bias_score(results)
    
    assert isinstance(score, float)
    assert score >= 0


def test_compute_gender_bias_score_empty():
    """Test bias score with empty input."""
    score = compute_gender_bias_score([])
    assert score == 0.0


def test_analyze_model_response_consistency():
    """Test response consistency analysis."""
    response_pairs = [
        ("Biased", "Biased"),
        ("Not Biased", "Not Biased"),
        ("Biased", "Not Biased"),
    ]
    
    result = analyze_model_response_consistency(response_pairs)
    
    assert "consistency_rate" in result
    assert 0 <= result["consistency_rate"] <= 1
    assert result["consistency_rate"] == 2/3  # 2 out of 3 match


def test_empty_inputs():
    """Test with empty inputs."""
    # Empty text
    assert count_gendered_terms("") == {"male": 0, "female": 0}
    
    # Empty lists
    result = compute_bias_disparity([], [])
    assert result["disparity"] == 0.0
    
    # Empty reasoning
    result2 = analyze_reasoning_content([], [])
    assert result2["avg_male_terms"] == 0


def test_case_insensitivity():
    """Test that term counting is case-insensitive."""
    text1 = "HE is a MAN"
    counts1 = count_gendered_terms(text1)
    assert counts1["male"] > 0
    
    text2 = "SHE is a WOMAN"
    counts2 = count_gendered_terms(text2)
    assert counts2["female"] > 0


def test_multiple_occurrences():
    """Test counting multiple occurrences."""
    text = "He said he would help him with his work."
    counts = count_gendered_terms(text)
    assert counts["male"] >= 4  # he, he, him, his


def test_trait_attribution_with_multiple_traits():
    """Test trait attribution with multiple traits."""
    text = "He is intelligent, capable, and a strong leader."
    count = compute_positive_trait_attribution(text, "male")
    assert count >= 3  # intelligent, capable, strong, leader


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
