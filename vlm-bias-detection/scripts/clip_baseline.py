#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CLIP baseline for gender bias detection in image-caption pairs.
This script uses OpenAI's CLIP model to create embeddings for images and captions,
then trains a simple classifier on top to detect gender bias.
"""

import os
import argparse
import json
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("clip_baseline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CLIP baseline for gender bias detection")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input JSONL file with image-caption pairs and labels")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save output JSONL file with predictions")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing the images")
    parser.add_argument("--train_split", type=float, default=0.8,
                        help="Proportion of data to use for training")
    parser.add_argument("--classifier_output", type=str, default="models/clip_baseline_classifier.pkl",
                        help="Path to save the trained classifier")
    parser.add_argument("--clip_model", type=str, default="ViT-L/14",
                        help="CLIP model variant to use")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()

def load_data(input_file: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError as e:
                logger.warning(f"Error parsing line in {input_file}: {e}")
    
    logger.info(f"Loaded {len(data)} samples from {input_file}")
    return data

def load_clip_model(model_name: str) -> Tuple[Any, Any]:
    """Load the CLIP model and processor."""
    import clip
    
    model, preprocess = clip.load(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loaded CLIP model: {model_name}")
    
    return model, preprocess

def extract_clip_features(
    data: List[Dict[str, Any]],
    model,
    preprocess,
    image_dir: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract CLIP features for images and captions."""
    # Initialize arrays for features and labels
    image_features = []
    text_features = []
    labels = []
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Process each sample
    for item in tqdm(data, desc="Extracting CLIP features"):
        try:
            # Load and process image
            image_path = os.path.join(image_dir, item["image_id"])
            image = Image.open(image_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            
            # Process caption
            caption = item["caption"]
            import clip
            text_input = clip.tokenize([caption]).to(device)
            
            # Extract features
            with torch.no_grad():
                image_feat = model.encode_image(image_input)
                text_feat = model.encode_text(text_input)
            
            # Normalize features
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
            
            # Combine image and text features
            combined_feat = torch.cat([image_feat, text_feat], dim=1)
            
            # Add to lists
            image_features.append(combined_feat.cpu().numpy().flatten())
            
            # Add label (1 for "Biased", 0 for "Not Biased")
            label = 1 if item["bias_label"] == "Biased" else 0
            labels.append(label)
            
        except Exception as e:
            logger.error(f"Error processing {item['image_id']}: {e}")
    
    # Convert lists to numpy arrays
    X = np.array(image_features)
    y = np.array(labels)
    
    logger.info(f"Extracted features for {len(X)} samples")
    
    return X, y

def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    train_split: float,
    random_seed: int,
    output_path: str
) -> LogisticRegression:
    """Train a logistic regression classifier on CLIP features."""
    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=train_split, random_state=random_seed
    )
    
    logger.info(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
    
    # Train logistic regression classifier
    classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        random_state=random_seed
    )
    classifier.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred = classifier.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    logger.info(f"Validation accuracy: {accuracy:.4f}")
    logger.info(f"Classification report:\n{classification_report(y_val, y_pred)}")
    
    # Save the classifier
    import joblib
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(classifier, output_path)
    logger.info(f"Saved classifier to {output_path}")
    
    return classifier

def predict_with_classifier(
    classifier,
    X: np.ndarray
) -> np.ndarray:
    """Make predictions with the trained classifier."""
    return classifier.predict(X)

def generate_output(
    data: List[Dict[str, Any]],
    predictions: np.ndarray,
    output_file: str
):
    """Generate output JSONL file with predictions."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        for i, item in enumerate(data):
            if i < len(predictions):
                result = {
                    "image_id": item["image_id"],
                    "caption": item["caption"],
                    "bias_label": "Biased" if predictions[i] == 1 else "Not Biased",
                    "reasoning": "Prediction made by CLIP embeddings + LogisticRegression classifier"
                }
                f.write(json.dumps(result) + '\n')
    
    logger.info(f"Saved predictions to {output_file}")

def main():
    """Main function to run the CLIP baseline."""
    args = setup_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    # Load data
    data = load_data(args.input)
    
    # Import CLIP here to avoid import errors for those without the package
    import clip
    
    # Load CLIP model
    model, preprocess = load_clip_model(args.clip_model)
    
    # Extract features
    X, y = extract_clip_features(data, model, preprocess, args.image_dir)
    
    # Train classifier
    classifier = train_classifier(
        X, y, args.train_split, args.random_seed, args.classifier_output
    )
    
    # Make predictions (on the entire dataset for simplicity)
    predictions = predict_with_classifier(classifier, X)
    
    # Generate output
    generate_output(data, predictions, args.output)

if __name__ == "__main__":
    main()
