# VLM-Bias-Detection-with-Chain-of-Thought-Reasoning

##  Key Features

- **Explainable Bias Detection**: CoT reasoning provides step-by-step explanations
- **Multi-dimensional Analysis**: Covers gender, race, age, and occupational biases
- **High Performance**: 78.33% accuracy with 4.1/5 human clarity rating
- **Efficient Training**: Uses LoRA adapters for 73% reduction in inference costs
- **Multiple Baselines**: Includes CLIP and LLaVA baseline implementations

##  Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset](#dataset)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

##  Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM (32GB+ recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/irfanalimd/VLM-Bias-Detection-with-Chain-of-Thought-Reasoning.git
cd VLM-Bias-Detection-with-Chain-of-Thought-Reasoning
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

##  Quick Start

### 1. Run Inference on Sample Data

```python
from src.model import BiasDetector
from PIL import Image

# Load model
detector = BiasDetector.from_pretrained("models/llamav_o1_bias_detector")

# Analyze an image-caption pair
image = Image.open("examples/sample_image.jpg")
caption = "A woman in the kitchen preparing dinner."

result = detector.analyze(image, caption)
print(f"Bias Label: {result['bias_label']}")
print(f"Reasoning: {result['reasoning']}")
```

### 2. Batch Processing

```bash
python scripts/infer.py \
  --input data/test_data.jsonl \
  --output results/predictions.jsonl \
  --model_path models/llamav_o1_bias_detector \
  --image_dir data/images
```

### 3. Interactive Demo

```bash
jupyter notebook notebooks/inference_demo.ipynb
```

##  Dataset

We use a curated subset of Flickr30K containing 3,300 image-caption pairs focused on social contexts:

- **Training Set**: 2,700 examples (81.8%)
- **Validation Set**: 300 examples (9.1%)
- **Test Set**: 300 examples (9.1%)

### Dataset Preparation

1. Download Flickr30K dataset
2. Run the preprocessing script:

```bash
python data/prepare_data.py \
  --flickr_dir /path/to/flickr30k \
  --output_dir data/processed
```

### Generate CoT Labels

Use GPT-4 to generate Chain-of-Thought reasoning:

```bash
export OPENAI_API_KEY="your-api-key"

python scripts/generate_cot_labels.py \
  --input_file data/processed/filtered_captions.jsonl \
  --output_file data/processed/train_annotations.jsonl \
  --image_dir data/images \
  --batch_size 10
```

##  Training

### Fine-tune LLaMA-V-O1

```bash
python scripts/train.py --config config/train_config.yaml
```

### Training Configuration

Key parameters in `config/train_config.yaml`:

```yaml
model_name: "omkarthawakar/LlamaV-o1"
num_epochs: 5
batch_size: 4
learning_rate: 2e-5
max_length: 512
```

### Monitor Training

```bash
tensorboard --logdir models/llamav_o1_bias_detector/logs
```

##  Inference

### Single Image Analysis

```python
from scripts.infer import analyze_single_image

result = analyze_single_image(
    image_path="examples/sample.jpg",
    caption="A male surgeon performing surgery.",
    model_path="models/llamav_o1_bias_detector"
)
```

### Batch Inference

```bash
python scripts/infer.py \
  --input data/test_data.jsonl \
  --output results/predictions.jsonl \
  --model_path models/llamav_o1_bias_detector \
  --image_dir data/images \
  --batch_size 8
```

##  Evaluation

### Run Evaluation

```bash
python scripts/evaluate_bias.py \
  --test_data data/processed/test_annotations.jsonl \
  --model_predictions results/predictions.jsonl \
  --clip_baseline results/clip_predictions.jsonl \
  --llava_baseline results/llava_predictions.jsonl \
  --output_dir evaluation/results
```

### Metrics

Our evaluation includes:

- **Classification Metrics**: Accuracy, Precision, Recall, F1-Score
- **Bias Metrics**: Gender term ratio, trait attribution analysis
- **Human Evaluation**: Clarity and interpretability ratings

##  Results

### Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | Clarity |
|-------|----------|-----------|--------|----------|---------|
| **Ours (LLaMA-V-O1 + CoT)** | **78.33%** | **72.5%** | **80.04%** | **0.76** | **4.1/5** |
| CLIP Baseline | 71.2% | 68.3% | 69.8% | 0.69 | N/A |
| LLaVA Baseline | 74.5% | 70.1% | 75.2% | 0.73 | 3.6/5 |

### Key Findings

- ✅ 98.2% retention of VQA benchmark performance
- ✅ 73% reduction in inference costs vs. full-precision models
- ✅ Reveals 22% competence score gap between demographics
- ✅ Human-interpretable reasoning with 4.1/5 clarity rating
## Architecture

![Pipeline Diagram](https://github.com/irfanalimd/VLM-Bias-Detection-with-Chain-of-Thought-Reasoning/blob/main/pipeline.jpg?raw=true)

##  Advanced Usage

### Custom Model Training

```python
from src.trainer import BiasDetectionTrainer

trainer = BiasDetectionTrainer(
    model_name="omkarthawakar/LlamaV-o1",
    train_data="data/custom_train.jsonl",
    output_dir="models/custom_model",
    learning_rate=2e-5,
    num_epochs=5
)

trainer.train()
```

### API Server

Launch a REST API server:

```bash
python examples/api_server.py --port 8000
```

```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "image_url": "https://example.com/image.jpg",
    "caption": "A woman cooking in the kitchen."
  }'
```

##  Running Tests

```bash
pytest tests/ -v --cov=src
```
