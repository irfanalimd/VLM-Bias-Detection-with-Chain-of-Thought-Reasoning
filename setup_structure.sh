#!/bin/bash

# Create directories
mkdir -p vlm-bias-detection/{config,data,src,scripts,notebooks,evaluation,tests,docs,examples}

# Create root files
touch vlm-bias-detection/{README.md,LICENSE,requirements.txt,setup.py,.gitignore,.env.example}

# Config files
touch vlm-bias-detection/config/{train_config.yaml,inference_config.yaml}

# Data
touch vlm-bias-detection/data/{README.md,sample_data.jsonl,prepare_data.py}

# Source code
touch vlm-bias-detection/src/{__init__.py,dataset.py,model.py,trainer.py,utils.py}

# Scripts
touch vlm-bias-detection/scripts/{generate_cot_labels.py,train.py,infer.py,evaluate_bias.py,clip_baseline.py,llava_baseline.py}

# Notebooks
touch vlm-bias-detection/notebooks/{inference_demo.ipynb,data_exploration.ipynb}

# Evaluation
touch vlm-bias-detection/evaluation/{metrics.py,analyze_results.py}

# Tests
touch vlm-bias-detection/tests/{__init__.py,test_dataset.py,test_model.py,test_metrics.py}

# Docs
touch vlm-bias-detection/docs/{METHODOLOGY.md,RESULTS.md,API.md}

# Examples
touch vlm-bias-detection/examples/{quick_start.py,batch_inference.py}

echo "✅ Project structure created successfully!"
