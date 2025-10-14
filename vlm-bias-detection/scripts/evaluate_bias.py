#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Evaluate the fine-tuned model's performance in detecting gender bias in image-caption pairs.
This script compares the model's predictions against ground truth labels and baseline models.
"""

import os
import argparse
import json
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.metrics import compute_gender_bias_score, count_gendered_terms, compute_bias_disparity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate model performance on bias detection")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to test data with ground truth labels")
    parser.add_argument("--model_predictions", type=str, required=True,
                        help="Path to model predictions (from infer.py)")
    parser.add_argument("--clip_baseline", type=str, default=None,
                        help="Path to CLIP baseline predictions")
    parser.add_argument("--llava_baseline", type=str, default=None,
                        help="Path to LLaVA baseline predictions")
    parser.add_argument("--output_dir", type=str, default="evaluation/results",
                        help="Directory to save evaluation results")
    parser.add_argument("--bias_evaluation_dataset", type=str, default=None,
                        help="Path to the special bias evaluation dataset (male/female pairs)")
    return parser.parse_args()

def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line in {file_path}: {e}")
    return data

def evaluate_classification_performance(
    ground_truth: List[str],
    predictions: List[str]
) -> Dict[str, float]:
    """Evaluate classification performance metrics."""
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average='binary', pos_label='Biased'
    )
    
    # Calculate confusion matrix
    cm = confusion_matrix(ground_truth, predictions, labels=['Biased', 'Not Biased'])
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

def plot_confusion_matrix(cm: np.ndarray, output_path: str):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Biased', 'Not Biased'],
                yticklabels=['Biased', 'Not Biased'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def compare_models(
    test_data: List[Dict[str, Any]],
    model_predictions: List[Dict[str, Any]],
    clip_predictions: List[Dict[str, Any]] = None,
    llava_predictions: List[Dict[str, Any]] = None
) -> Dict[str, Dict[str, float]]:
    """Compare the performance of different models."""
    # Create lookup dictionaries for predictions
    model_pred_dict = {item['image_id']: item for item in model_predictions}
    
    clip_pred_dict = {}
    if clip_predictions:
        clip_pred_dict = {item['image_id']: item for item in clip_predictions}
    
    llava_pred_dict = {}
    if llava_predictions:
        llava_pred_dict = {item['image_id']: item for item in llava_predictions}
    
    # Collect ground truth and predictions for shared images
    gt_labels = []
    model_preds = []
    clip_preds = []
    llava_preds = []
    
    for item in test_data:
        image_id = item['image_id']
        
        if image_id in model_pred_dict:
            gt_labels.append(item['bias_label'])
            model_preds.append(model_pred_dict[image_id]['bias_label'])
            
            if image_id in clip_pred_dict:
                clip_preds.append(clip_pred_dict[image_id]['bias_label'])
            
            if image_id in llava_pred_dict:
                llava_preds.append(llava_pred_dict[image_id]['bias_label'])
    
    # Evaluate our model
    model_results = evaluate_classification_performance(gt_labels, model_preds)
    
    # Evaluate CLIP baseline if available
    clip_results = None
    if clip_predictions and len(clip_preds) == len(gt_labels):
        clip_results = evaluate_classification_performance(gt_labels, clip_preds)
    
    # Evaluate LLaVA baseline if available
    llava_results = None
    if llava_predictions and len(llava_preds) == len(gt_labels):
        llava_results = evaluate_classification_performance(gt_labels, llava_preds)
    
    # Return results
    results = {
        'model': model_results,
        'clip_baseline': clip_results,
        'llava_baseline': llava_results
    }
    
    return results

def evaluate_bias_in_model_responses(
    bias_eval_data: List[Dict[str, Any]],
    model_pred_dict: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Evaluate if the model itself shows bias in its responses to male vs female subjects.
    Adapted from the VLA gender bias research methodology.
    """
    # Group the data by image pairs (male/female versions)
    paired_data = {}
    for item in bias_eval_data:
        pair_id = item['pair_id']
        gender = item['gender']
        
        if pair_id not in paired_data:
            paired_data[pair_id] = {}
        
        paired_data[pair_id][gender] = item
    
    # Collect model responses for each pair
    results = []
    for pair_id, pair in paired_data.items():
        if 'male' in pair and 'female' in pair:
            male_item = pair['male']
            female_item = pair['female']
            
            # Get model predictions if available
            male_pred = model_pred_dict.get(male_item['image_id'])
            female_pred = model_pred_dict.get(female_item['image_id'])
            
            if male_pred and female_pred:
                # Add to results
                result = {
                    'pair_id': pair_id,
                    'male_image_id': male_item['image_id'],
                    'female_image_id': female_item['image_id'],
                    'male_caption': male_item['caption'],
                    'female_caption': female_item['caption'],
                    'male_bias_label': male_pred['bias_label'],
                    'female_bias_label': female_pred['bias_label'],
                    'male_reasoning': male_pred['reasoning'],
                    'female_reasoning': female_pred['reasoning'],
                }
                
                # Calculate gender term counts
                male_term_count = count_gendered_terms(male_pred['reasoning'])
                female_term_count = count_gendered_terms(female_pred['reasoning'])
                
                result.update({
                    'male_term_count': male_term_count,
                    'female_term_count': female_term_count,
                })
                
                # Check for bias disparity (measures if one gender is more often flagged as biased)
                if male_pred['bias_label'] != female_pred['bias_label']:
                    result['bias_disparity'] = True
                else:
                    result['bias_disparity'] = False
                
                results.append(result)
    
    # Calculate summary statistics
    total_pairs = len(results)
    bias_disparity_count = sum(1 for r in results if r['bias_disparity'])
    
    # Calculate gender term imbalance
    male_terms_total = sum(r['male_term_count']['male'] for r in results)
    female_terms_total = sum(r['female_term_count']['female'] for r in results)
    
    # Calculate bias scores
    bias_score = compute_gender_bias_score(results)
    
    summary = {
        'total_pairs': total_pairs,
        'bias_disparity_count': bias_disparity_count,
        'bias_disparity_percentage': (bias_disparity_count / total_pairs) * 100 if total_pairs > 0 else 0,
        'male_terms_total': male_terms_total,
        'female_terms_total': female_terms_total,
        'male_female_term_ratio': male_terms_total / female_terms_total if female_terms_total > 0 else float('inf'),
        'bias_score': bias_score,
        'pair_results': results
    }
    
    return summary

def main():
    """Main function to evaluate model performance."""
    args = setup_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    test_data = load_data(args.test_data)
    model_predictions = load_data(args.model_predictions)
    
    clip_predictions = None
    if args.clip_baseline:
        clip_predictions = load_data(args.clip_baseline)
    
    llava_predictions = None
    if args.llava_baseline:
        llava_predictions = load_data(args.llava_baseline)
    
    # Compare model performance
    comparison_results = compare_models(
        test_data, model_predictions, clip_predictions, llava_predictions
    )
    
    # Print and save comparison results
    logger.info("Classification Performance Results:")
    for model_name, results in comparison_results.items():
        if results:
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"Accuracy: {results['accuracy']:.4f}")
            logger.info(f"Precision: {results['precision']:.4f}")
            logger.info(f"Recall: {results['recall']:.4f}")
            logger.info(f"F1 Score: {results['f1']:.4f}")
            
            # Plot confusion matrix
            cm_path = os.path.join(args.output_dir, f"{model_name}_confusion_matrix.png")
            plot_confusion_matrix(results['confusion_matrix'], cm_path)
    
    # Save comparison results to file
    comparison_output = os.path.join(args.output_dir, "classification_results.json")
    with open(comparison_output, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, results in comparison_results.items():
            if results:
                serializable_results[model_name] = {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in results.items()
                }
        
        json.dump(serializable_results, f, indent=2)
    
    # Evaluate bias in model responses if bias evaluation dataset is provided
    if args.bias_evaluation_dataset:
        bias_eval_data = load_data(args.bias_evaluation_dataset)
        model_pred_dict = {item['image_id']: item for item in model_predictions}
        
        bias_summary = evaluate_bias_in_model_responses(bias_eval_data, model_pred_dict)
        
        # Print bias evaluation results
        logger.info("\nBias Evaluation Results:")
        logger.info(f"Total pairs analyzed: {bias_summary['total_pairs']}")
        logger.info(f"Pairs with bias disparity: {bias_summary['bias_disparity_count']} ({bias_summary['bias_disparity_percentage']:.2f}%)")
        logger.info(f"Male terms mentioned: {bias_summary['male_terms_total']}")
        logger.info(f"Female terms mentioned: {bias_summary['female_terms_total']}")
        logger.info(f"Male/Female term ratio: {bias_summary['male_female_term_ratio']:.2f}")
        logger.info(f"Bias score: {bias_summary['bias_score']:.4f}")
        
        # Save bias evaluation results
        bias_output = os.path.join(args.output_dir, "bias_evaluation_results.json")
        with open(bias_output, 'w') as f:
            # Remove the detailed pair results for brevity
            summary_for_output = {k: v for k, v in bias_summary.items() if k != 'pair_results'}
            json.dump(summary_for_output, f, indent=2)
        
        # Save detailed pair results separately
        detailed_output = os.path.join(args.output_dir, "bias_evaluation_details.jsonl")
        with open(detailed_output, 'w') as f:
            for result in bias_summary['pair_results']:
                f.write(json.dumps(result) + '\n')
    
    logger.info(f"Evaluation results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
