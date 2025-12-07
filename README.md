# AnonBERT-ENR

**Hybrid RegEx + BERT approach for Named Entity Recognition and anonymization of Polish personal data.**

AnonBERT-ENR is a fine-tuned HerBERT (Polish BERT) model for identifying and anonymizing sensitive personal information in Polish text. The system combines rule-based preprocessing with transformer-based NER to achieve robust anonymization across 25+ entity types.

---

## Repository Structure

```
.
├── anonbert/                    # Main package
│   ├── __init__.py             # Package exports
│   ├── anonbert.py             # High-level API
│   ├── anonimizer.py           # Faker-based data generation
│   └── anonymizepredict.py     # NER inference pipeline
├── data/                        # Training and test data
│   ├── orig.txt                # Original text with [tag] markers
│   ├── anonymized.txt          # Text with replaced placeholders
│   ├── ner_dataset.conll       # CoNLL format training data
│   ├── ner_dataset.jsonl       # JSONL format training data
│   └── test.txt                # Sample input for testing
├── herbert-ner/                 # Fine-tuned model weights (generated)
├── herbert-tuning.py           # Training script with Optuna hyperparameter search
├── prepare_dataset.py          # Dataset creation from orig.txt + anonymized.txt
├── main.py                     # Example usage script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

**Pre-trained model**: Available on Hugging Face at [`Matela7/AnonBert-ENR`](https://huggingface.co/Matela7/AnonBert-ENR)

---

## How It Works

### 1. **Dataset Preparation**
The `prepare_dataset.py` script aligns two text files:
- **orig.txt**: Original text with placeholder tags like `[name]`, `[email]`, `[pesel]`
- **anonymized.txt**: The same text with real values replacing the placeholders

The alignment algorithm uses heuristics (anchoring, punctuation patterns, name/surname detection) to map tokens to their correct BIO tags, producing:
- `ner_dataset.conll`: CoNLL format (token\ttag per line)
- `ner_dataset.jsonl`: JSON lines format

### 2. **Model Training**
`herbert-tuning.py` fine-tunes the `allegro/herbert-base-cased` model:
- Supports 25+ entity types (NAME, SURNAME, EMAIL, PESEL, etc.)
- Uses Optuna for hyperparameter optimization (20 trials)
- Implements stratified train/val/test split (60/20/20)
- Evaluates using seqeval F1 score

### 3. **Anonymization Pipeline**
`anonymizepredict.py` provides two-stage anonymization:
- **Stage 1**: Replace detected entities with tags (`[name]`, `[email]`, etc.)
- **Stage 2**: Fill tags with realistic fake data using Faker library

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/anonbert-enr.git
cd anonbert-enr

# Install dependencies
pip install -r requirements.txt

# (Optional) Install as package
pip install -e .
```

**Note**: The model requires PyTorch. Ensure you have the appropriate version for your system (CPU/CUDA).

---

## 🚀 Usage

### Quick Start

```python
from anonbert import anonymize_file, anonymize_with_fakefill
from pathlib import Path

# Anonymize with placeholder tags
input_file = Path("data/test.txt")
anonymize_file(input_file)  # Creates test_anon.txt

# Anonymize with realistic fake data
anonymize_with_fakefill(input_file)  # Creates test_anon_fake.txt
```

### Command Line

```bash
python main.py
```

### Training Your Own Model

```bash
# 1. Prepare your dataset
python prepare_dataset.py

# 2. Train the model (requires GPU recommended)
python herbert-tuning.py
```

---

## 📊 Supported Entity Types

| Tag | Description | Example |
|-----|-------------|---------|
| `NAME` | First name | Jan |
| `SURNAME` | Last name | Kowalski |
| `EMAIL` | Email address | jan@example.pl |
| `PHONE` | Phone number | +48 123 456 789 |
| `PESEL` | National ID | 12345678901 |
| `ADDRESS` | Street address | ul. Marszałkowska 1 |
| `CITY` | City name | Warszawa |
| `COMPANY` | Company name | Allegro |
| `DATE` | General date | 2024-01-15 |
| `DATE_BIRTH` | Date of birth | 1990-05-20 |
| ... | *+15 more* | ... |

Full list in `herbert-tuning.py` (`LABEL_LIST`).

---

## 🔧 Configuration

### Using a Different Model

```python
from anonbert import anonymize_file

# Use local model
anonymize_file("input.txt", model_dir="./herbert-ner")

# Use different HF model
anonymize_file("input.txt", model_dir="username/model-name")
```

### Faker Localization

```python
from anonbert.anonimizer import Anonimizer

anon = Anonimizer(locale='en_US')  # Change locale
anon.ReadText('input.txt')
output = anon.FakeFillAll()
```

---

## Authors

- Michał Matela
- Oskar Bartoszyński
- Nicolas Graeb
- Justyna Starszczak
- Dawid Stefański

---

## License

[Add your license here]

---

## Acknowledgments

- HerBERT model: [Allegro/HerBERT](https://github.com/allegro/HerBERT)
- Hugging Face Transformers library
- Faker library for synthetic data generation