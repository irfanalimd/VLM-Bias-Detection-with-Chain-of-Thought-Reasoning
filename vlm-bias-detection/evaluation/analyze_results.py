#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze and visualize evaluation results.
"""

import os
import json
import argparse
import logging
from typing import Dict, List, Any
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_results(file_path: str) -> List[Dict[str, Any]]:
    """Load results from JSONL file."""
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue
    return results


def compute_confusion_matrix(
    predictions: List[str],
    ground_truth: List[str]
) -> np.ndarray:
    """Compute confusion matrix."""
    labels = ['Biased', 'Not Biased']
    return confusion_matrix(ground_truth, predictions, labels=labels)


def plot_confusion_matrix(
    cm: np.ndarray,
    output_path: str,
    title: str = "Confusion Matrix"
):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Biased', 'Not Biased'],
        yticklabels=['Biased', 'Not Biased'],
        cbar_kw={'label': 'Count'}
    )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to {output_path}")


def analyze_bias_distribution(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze distribution of bias labels."""
    distribution = {'Biased': 0, 'Not Biased': 0, 'Unclear': 0}
    
    for result in results:
        label = result.get('bias_label', 'Unclear')
        distribution[label] = distribution.get(label, 0) + 1
    
    return distribution


def plot_bias_distribution(
    distribution: Dict[str, int],
    output_path: str
):
    """Plot bias label distribution."""
    plt.figure(figsize=(10, 6))
    labels = list(distribution.keys())
    values = list(distribution.values())
    colors = ['#ff6b6b', '#4ecdc4', '#95a5a6']
    
    plt.bar(labels, values, color=colors, edgecolor='black', linewidth=1.5)
    plt.title('Distribution of Bias Labels', fontsize=16, fontweight='bold')
    plt.xlabel('Bias Label', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved distribution plot to {output_path}")


def analyze_reasoning_length(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze reasoning text lengths."""
    lengths = [len(r.get('reasoning', '')) for r in results]
    
    return {
        'mean': np.mean(lengths),
        'median': np.median(lengths),
        'std': np.std(lengths),
        'min': np.min(lengths),
        'max': np.max(lengths)
    }


def plot_reasoning_length_distribution(
    results: List[Dict[str, Any]],
    output_path: str
):
    """Plot distribution of reasoning lengths."""
    lengths = [len(r.get('reasoning', '')) for r in results]
    
    plt.figure(figsize=(12, 6))
    plt.hist(lengths, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(lengths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(lengths):.0f}')
    plt.axvline(np.median(lengths), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(lengths):.0f}')
    
    plt.title('Distribution of Reasoning Text Lengths', fontsize=16, fontweight='bold')
    plt.xlabel('Length (characters)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved reasoning length distribution to {output_path}")


def compare_models(
    results_dict: Dict[str, List[Dict[str, Any]]],
    output_path: str
):
    """Compare performance across multiple models."""
    model_names = list(results_dict.keys())
    accuracies = []
    
    for model_name, results in results_dict.items():
        predictions = [r['bias_label'] for r in results]
        ground_truth = [r.get('true_label', r.get('bias_label')) for r in results]
        accuracy = sum(p == g for p, g in zip(predictions, ground_truth)) / len(predictions)
        accuracies.append(accuracy * 100)
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(model_names)))
    bars = plt.bar(model_names, accuracies, color=colors, edgecolor='black', linewidth=1.5)
    
    plt.title('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved model comparison to {output_path}")


def generate_report(
    results: List[Dict[str, Any]],
    output_dir: str
):
    """Generate comprehensive analysis report."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract predictions and ground truth
    predictions = [r['bias_label'] for r in results]
    ground_truth = [r.get('true_label', r.get('bias_label')) for r in results]
    
    # Compute metrics
    cm = compute_confusion_matrix(predictions, ground_truth)
    distribution = analyze_bias_distribution(results)
    reasoning_stats = analyze_reasoning_length(results)
    
    # Generate plots
    plot_confusion_matrix(cm, os.path.join(output_dir, 'confusion_matrix.png'))
    plot_bias_distribution(distribution, os.path.join(output_dir, 'bias_distribution.png'))
    plot_reasoning_length_distribution(results, os.path.join(output_dir, 'reasoning_lengths.png'))
    
    # Create summary report
    report = {
        'total_samples': len(results),
        'bias_distribution': distribution,
        'reasoning_statistics': reasoning_stats,
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(ground_truth, predictions, output_dict=True)
    }
    
    # Save report
    report_path = os.path.join(output_dir, 'analysis_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Generated analysis report in {output_dir}")
    
    return report


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze evaluation results")
    parser.add_argument('--results', type=str, required=True, help='Path to results JSONL')
    parser.add_argument('--output_dir', type=str, default='evaluation/analysis', help='Output directory')
    parser.add_argument('--compare', type=str, nargs='+', help='Additional result files to compare')
    
    args = parser.parse_args()
    
    # Load results
    logger.info(f"Loading results from {args.results}")
    results = load_results(args.results)
    
    # Generate report
    report = generate_report(results, args.output_dir)
    
    # Print summary
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    print(f"Total Samples: {report['total_samples']}")
    print(f"\nBias Distribution:")
    for label, count in report['bias_distribution'].items():
        percentage = (count / report['total_samples']) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")
    
    print(f"\nReasoning Statistics:")
    for stat, value in report['reasoning_statistics'].items():
        print(f"  {stat}: {value:.1f}")
    
    print("\n" + "="*70)
    
    # Compare models if additional files provided
    if args.compare:
        logger.info("Comparing multiple models")
        results_dict = {'Main Model': results}
        for i, compare_file in enumerate(args.compare):
            model_name = f"Model {i+1}"
            results_dict[model_name] = load_results(compare_file)
        
        compare_models(results_dict, os.path.join(args.output_dir, 'model_comparison.png'))


if __name__ == "__main__":
    main()
