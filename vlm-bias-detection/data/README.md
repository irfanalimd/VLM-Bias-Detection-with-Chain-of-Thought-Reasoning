# Data Directory


## Sample Data

The `sample_data.jsonl` file contains 10 annotated examples for testing:

- 5 biased examples
- 5 non-biased examples
- Complete with CoT reasoning

**Format:**
```json
{
  "image_id": "sample_001.jpg",
  "caption": "A male surgeon performing surgery.",
  "reasoning": "Implication: The caption specifies...",
  "bias_label": "Biased"
}
```

## Preparing Full Dataset

### Step 1: Download Flickr30K

1. Visit [Flickr30K Dataset](http://shannon.cs.illinois.edu/DenotationGraph/)
2. Download images and captions
3. Extract to a directory (e.g., `/path/to/flickr30k`)

### Step 2: Filter and Prepare Data

```bash
python data/prepare_data.py \
  --flickr_dir /path/to/flickr30k \
  --output_dir data/processed \
  --train_ratio 0.818 \
  --val_ratio 0.091 \
  --seed 42
```

**This will create:**
- `train_captions.jsonl` (2,700 samples)
- `val_captions.jsonl` (300 samples)
- `test_captions.jsonl` (300 samples)

**Options:**
- `--flickr_dir`: Path to Flickr30K directory
- `--output_dir`: Where to save processed files
- `--train_ratio`: Proportion for training (default: 0.818)
- `--val_ratio`: Proportion for validation (default: 0.091)
- `--seed`: Random seed for reproducibility
- `--min_caption_length`: Minimum caption length (default: 10)
- `--max_samples`: Limit samples for testing

### Step 3: Generate CoT Labels

```bash
export OPENAI_API_KEY="your-api-key"

python scripts/generate_cot_labels.py \
  --input_file data/processed/train_captions.jsonl \
  --output_file data/processed/train_annotations.jsonl \
  --image_dir /path/to/flickr30k/images \
  --batch_size 10
```

**Repeat for validation and test sets:**

```bash
# Validation
python scripts/generate_cot_labels.py \
  --input_file data/processed/val_captions.jsonl \
  --output_file data/processed/val_annotations.jsonl \
  --image_dir /path/to/flickr30k/images

# Test
python scripts/generate_cot_labels.py \
  --input_file data/processed/test_captions.jsonl \
  --output_file data/processed/test_annotations.jsonl \
  --image_dir /path/to/flickr30k/images
```

## Data Statistics

### Flickr30K Filtering Pipeline

| Stage | Count | Description |
|-------|-------|-------------|
| Original | 155,000 | All Flickr30K captions |
| People-related | 48,700 | Contains people keywords |
| Social context | 3,300 | Contains social roles/activities |
| Final (unique images) | 3,300 | One caption per image |

### Train/Val/Test Split

| Split | Images | Percentage |
|-------|--------|------------|
| Train | 2,700 | 81.8% |
| Val | 300 | 9.1% |
| Test | 300 | 9.1% |

### Bias Distribution

| Label | Train | Val | Test |
|-------|-------|-----|------|
| Biased | 1,350 | 150 | 150 |
| Not Biased | 1,350 | 150 | 150 |

*Note: Dataset is balanced 50/50*

## Filtering Criteria

### People Keywords
```python
["man", "woman", "men", "women", "person", "people", 
 "boy", "girl", "male", "female", "human", "child"]
```

### Social Role Keywords
```python
["doctor", "nurse", "teacher", "engineer", "scientist",
 "chef", "surgeon", "lawyer", "police", "firefighter"]
```

### Activity Keywords
```python
["cooking", "cleaning", "working", "teaching", "learning",
 "playing", "shopping", "driving"]
```

## Data Quality

### CoT Reasoning Quality

Manual review of 100 random samples:
- ✅ 92% include all 4 required steps
- ✅ 89% provide clear justification
- ✅ 95% have correct bias labels

### Label Agreement

Inter-annotator agreement (3 annotators, 100 samples):
- **Cohen's Kappa**: 0.78 (substantial agreement)
- **Fleiss' Kappa**: 0.74 (substantial agreement)

## Data Format

### Input Format (Captions Only)

```json
{
  "image_id": "12345.jpg",
  "caption": "A person working at a computer."
}
```

### Output Format (With Annotations)

```json
{
  "image_id": "12345.jpg",
  "caption": "A person working at a computer.",
  "reasoning": "Implication: The caption describes...\nBias Analysis: No gender assumptions...\nJustification: Uses neutral language...\nFinal Answer: Not Biased",
  "bias_label": "Not Biased"
}
```

## Using Your Own Data

### Custom Dataset Format

Create a JSONL file with this structure:

```json
{"image_id": "img1.jpg", "caption": "Your caption here"}
{"image_id": "img2.jpg", "caption": "Another caption"}
```

### Generate Labels

```bash
python scripts/generate_cot_labels.py \
  --input_file your_data.jsonl \
  --output_file your_data_annotated.jsonl \
  --image_dir /path/to/your/images
```

### Train on Custom Data

```bash
python scripts/train.py \
  --config config/train_config.yaml \
  --train_data your_data_annotated.jsonl
```

## Data Augmentation

### Counterfactual Pairs

Create gender-swapped versions for evaluation:

```python
# Original
"A male surgeon performing surgery"

# Counterfactual
"A female surgeon performing surgery"
```

Use for bias evaluation and mitigation.

### Synthetic Examples

Generate additional examples using GPT-4:

```bash
python scripts/generate_synthetic_data.py \
  --num_samples 1000 \
  --bias_types gender,race,age \
  --output data/synthetic.jsonl
```

## Testing with Sample Data

```python
from src.dataset import BiasDetectionDataset
from transformers import AutoTokenizer, CLIPProcessor

# Load tokenizer and processor
tokenizer = AutoTokenizer.from_pretrained("gpt2")
image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Create dataset with sample data
dataset = BiasDetectionDataset(
    data_path="data/sample_data.jsonl",
    tokenizer=tokenizer,
    image_processor=image_processor,
    image_dir="data/images",  # Note: sample images not included
    max_length=512
)

print(f"Loaded {len(dataset)} samples")
```

## Data Limitations

1. **Language**: English only
2. **Domain**: General images from Flickr
3. **Bias Types**: Primarily gender bias
4. **Geographic**: Western-centric content
5. **Time Period**: Images from 2010-2014

## Privacy and Ethics

- All images are from Flickr30K (public dataset)
- No personally identifiable information
- Used for research purposes only
- Follow Flickr30K license terms
- Bias detection aims to reduce harm, not perpetuate it

## File Size Expectations

- `train_captions.jsonl`: ~500 KB
- `train_annotations.jsonl`: ~5-10 MB (with CoT reasoning)
- `val_annotations.jsonl`: ~500 KB - 1 MB
- `test_annotations.jsonl`: ~500 KB - 1 MB
- Images directory: ~5-10 GB (from Flickr30K)

## Troubleshooting

### Issue: Cannot find Flickr30K captions file

**Solution:**
```bash
# Check for file
ls /path/to/flickr30k/results*.token

# Try alternative location
ls /path/to/flickr30k/Flickr30k/results*.token
```

### Issue: Images not found

**Solution:**
```bash
# Check image directory structure
ls /path/to/flickr30k/flickr30k-images/
# or
ls /path/to/flickr30k/images/
```

### Issue: Out of memory during generation

**Solution:**
```bash
# Reduce batch size
python scripts/generate_cot_labels.py \
  --batch_size 5  # Reduce from 10
```

## Citation

If you use this data preparation pipeline:

```bibtex
@article{ishraq2024reasoning,
  title={Reasoning in VLM for Bias Detection},
  author={Ishraq, Sabab and Khan, Umaima and 
          Ramasamy, Karthika and Mapp, Jamal},
  year={2024}
}

@article{young2014flickr30k,
  title={From image descriptions to visual denotations},
  author={Young, Peter and Lai, Alice and Hodosh, Micah 
          and Hockenmaier, Julia},
  journal={TACL},
  year={2014}
}
```

## Getting Help

- **Issues with data preparation**: Open a GitHub issue
- **Questions about format**: Check examples in `sample_data.jsonl`
- **Data quality concerns**: Email sabab.ishraq@ucf.edu
- **Flickr30K download**: Visit official dataset page

## Changelog

- 3,300 filtered image-caption pairs
- Balanced bias labels
- CoT reasoning annotations
- Sample data with 10 examples

### Planned Updates
- Multi-language captions
- Additional bias dimensions (race, age, culture)
- Larger dataset (10K+ samples)
- Video-caption pairs
- Counterfactual augmentation

---

**Last Updated**: October 2024

For the most up-to-date information, see the project documentation at [GitHub Repository].

