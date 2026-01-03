# Contrastive Verifier for Self-Correction in Small LMs

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

> **Enable small language models to identify and correct their own reasoning errors—no GPT-4 needed.**

---

## Overview

This repository implements a self-contained system that allows small language models (2-7B parameters) to perform self-correction without external verifiers. Our approach:

1. Trains a **contrastive verifier** (BERT-base, 110M params) that scores solution quality with **97.68% accuracy**
2. Uses confidence scores to **selectively trigger refinement** (only when needed)
3. Generates **self-critiques** to identify errors
4. Produces **refined solutions** based on feedback
5. Achieves **substantial accuracy gains** with minimal overhead

### Key Results (GSM8K Test Set)

| Configuration | Baseline | With Self-Correction | Improvement |
|--------------|----------|---------------------|-------------|
| **Multi-Solution** (5 attempts) | 31.75% | **35.10%** | +10.5% relative |
| **Single-Solution** (1 attempt) | 6.52% | **12.74%** | **+95.3% relative** |

- **Safety**: 21:1 benefit-to-risk ratio (rarely degrades correct solutions)
- **Efficiency**: Only 5-94% compute overhead depending on configuration
- **Novel**: Exceeds oracle performance by generating new correct solutions

---

## Quick Start

### Installation
```bash
git clone https://github.com/yourusername/contrastive-verifier.git
cd contrastive-verifier
pip install -r requirements.txt
```

### 1. Prepare Data & Train Verifier
```bash
# Download GSM8K and generate solutions
python scripts/prepare_data.py

# Train the verifier (BERT-base)
python scripts/train_verifier.py \
  --train_data data/train_pairs.jsonl \
  --val_data data/val_pairs.jsonl \
  --output checkpoints/verifier_best.pt
```

**Expected**: 97.68% validation accuracy in ~3 hours on A40 GPU

### 2. Run Self-Correction

**Multi-Solution (best accuracy):**
```bash
python scripts/run_multi_solution.py \
  --verifier checkpoints/verifier_best.pt \
  --base_model google/gemma-2-2b-it \
  --test_data data/gsm8k_test.jsonl \
  --output results/multi_results.jsonl
```

**Single-Solution (most efficient):**
```bash
python scripts/run_single_solution.py \
  --verifier checkpoints/verifier_best.pt \
  --base_model google/gemma-2-2b-it \
  --test_data data/gsm8k_test.jsonl \
  --output results/single_results.jsonl
```

### 3. Evaluate
```bash
python scripts/evaluate.py --results results/multi_results.jsonl
```

---

## How It Works
```
Question → Generate Solution(s) → Verifier Scores
                                       ↓
                              Score < threshold?
                                       ↓
                    ┌──────────────────┴────────────────┐
                    NO                                  YES
                    ↓                                    ↓
               Use as-is                    Generate Critique
                    ↓                                    ↓
                    └──────────→ Final  ←─── Generate Refinement
                                Solution         ↓
                                            Re-score & Pick Best
```

**Key Innovation**: Contrastive verifier training using correct/incorrect solution pairs enables autonomous confidence estimation.

---

## Results Summary

### Accuracy Gains

| Method | Accuracy | vs. SCORE Baseline |
|--------|----------|-------------------|
| SCORE Random | 12.34% | Baseline |
| **Verifier Only** | 31.75% | **+157%** |
| **Verifier + Refinement** | 35.10% | **+184%** |
| Oracle (best of 5) | 32.76% | +165% |

**Exceeds oracle** by generating novel correct solutions! 🎯

### Safety & Efficiency

- **Helped**: 321 questions (wrong → correct)
- **Hurt**: 16 questions (correct → wrong)
- **Risk-Reward Ratio**: 20:1
- **Compute Overhead**: 5% (multi) / 94% (single)

---

## Usage

### Python API
```python
from src.pipeline import SelfCorrectionPipeline

# Initialize
pipeline = SelfCorrectionPipeline(
    verifier_checkpoint='checkpoints/verifier_best.pt',
    base_model='google/gemma-2-2b-it',
    threshold=0.7
)

# Process a question
result = pipeline.process(
    question="John has 5 apples. He gives 2 to Mary. How many left?",
    n_solutions=5
)

print(f"Answer: {result['final_solution']}")
print(f"Confidence: {result['final_score']:.3f}")
```

### Score Individual Solutions
```python
from src.verifier import load_verifier

verifier = load_verifier('checkpoints/verifier_best.pt')
score = verifier.score(question, solution)

# score > 1.0: High confidence
# score < 0.0: Low confidence
```

---

## Project Structure
```
contrastive-verifier/
├── data/                    # Datasets
├── src/                     # Core implementation
│   ├── verifier.py         # BERT-based verifier
│   ├── refiner.py          # Critique & refinement
│   └── pipeline.py         # Complete system
├── scripts/                 # Executable scripts
│   ├── train_verifier.py
│   ├── run_multi_solution.py
│   └── evaluate.py
├── checkpoints/            # Trained models
└── results/                # Experiment outputs
```

---

## 🔧 Requirements
```
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
tqdm>=4.65.0
numpy>=1.24.0
```

**Hardware**: GPU with 16GB+ VRAM recommended

---

## Reproduce Results
```bash
# Complete pipeline (train + evaluate)
bash scripts/reproduce_results.sh
```

**Expected runtime**: ~15 hours on A40 GPU

---

## Contributing

Contributions welcome! Areas of interest:

- Support for more datasets (MATH, StrategyQA)
- Additional base models
- Improved prompting strategies
- Production optimizations
---

## Acknowledgments

- GSM8K dataset by OpenAI
- Hugging Face Transformers
- RunPod for GPU infrastructure

---

## Contact

- **GitHub**: [@Altafbaigal997](https://github.com/Altafbaigal997)
- **Email**: altaf.baigal997@gmail.com
---
