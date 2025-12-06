from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import numpy as np
from seqeval.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from collections import Counter


model_name = "allegro/herbert-base-cased"

# Etykiety
LABEL_LIST = [
    "O",
    "B-NAME", "I-NAME",
    "B-SURNAME", "I-SURNAME",
    "B-AGE", "I-AGE",
    "B-DATE_BIRTH", "I-DATE_BIRTH",
    "B-DATE", "I-DATE",
    "B-SEX", "I-SEX",
    "B-RELIGION", "I-RELIGION",
    "B-POLITICAL", "I-POLITICAL",
    "B-ETHNICITY", "I-ETHNICITY",
    "B-ORIENTATION", "I-ORIENTATION",
    "B-HEALTH", "I-HEALTH",
    "B-RELATIVE", "I-RELATIVE",
    "B-CITY", "I-CITY",
    "B-ADDRESS", "I-ADDRESS",
    "B-EMAIL", "I-EMAIL",
    "B-PHONE", "I-PHONE",
    "B-PESEL", "I-PESEL",
    "B-DOCUMENT", "I-DOCUMENT",
    "B-COMPANY", "I-COMPANY",
    "B-SCHOOL", "I-SCHOOL",
    "B-JOB", "I-JOB",
    "B-BANK_ACCOUNT", "I-BANK_ACCOUNT",
    "B-CREDIT_CARD", "I-CREDIT_CARD",
    "B-USERNAME", "I-USERNAME",
    "B-SECRET", "I-SECRET",
    "B-PII", "I-PII"
]

NUM_LABELS = len(LABEL_LIST)
label2id = {l: i for i, l in enumerate(LABEL_LIST)}
id2label = {i: l for i, l in enumerate(LABEL_LIST)}

print(f"Zdefiniowano {NUM_LABELS} etykiet")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id
)


def load_conll_file(filepath):
    examples = []
    current_tokens = []
    current_labels = []
    unknown_labels = set()
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                if current_tokens:
                    examples.append({
                        'tokens': current_tokens,
                        'ner_tags': current_labels
                    })
                    current_tokens = []
                    current_labels = []
            else:
                parts = line.split('\t')
                if len(parts) >= 2:
                    token = parts[0]
                    tag = parts[1]
                    
                    if tag not in label2id:
                        unknown_labels.add(tag)
                        tag = 'O'
                    
                    current_tokens.append(token)
                    current_labels.append(tag)
    
    if current_tokens:
        examples.append({
            'tokens': current_tokens,
            'ner_tags': current_labels
        })
    
    if unknown_labels:
        print(f"\n⚠️  Znaleziono nieznane etykiety (zamieniono na 'O'): {', '.join(sorted(unknown_labels))}")
    
    return examples


def get_sentence_label_distribution(example):
    label_counts = Counter()
    for tag in example['ner_tags']:
        if tag.startswith('B-'):
            label_counts[tag] += 1

    if label_counts:
        return label_counts.most_common(1)[0][0]
    return 'O'


def prepare_stratify_labels(labels, min_count=2):
    label_counts = Counter(labels)
    stratify_safe = []
    
    for label in labels:
        if label_counts[label] < min_count:
            stratify_safe.append('RARE_CLASS')
        else:
            stratify_safe.append(label)
    
    return stratify_safe


# -------------------------
# Wczytywanie i podział danych
# -------------------------

print("Wczytywanie danych z ner_dataset.conll...")
all_examples = load_conll_file("data/ner_dataset.conll")
print(f"Wczytano {len(all_examples)} przykładów (zdań)")

stratify_labels = [get_sentence_label_distribution(ex) for ex in all_examples]
stratify_safe = prepare_stratify_labels(stratify_labels, min_count=2)

print("\nWykonywanie podziału train/val/test (80/10/10) ze stratyfikacją...")

# train = 80%, temp = 20%
train_examples, temp_examples, train_stratify, temp_stratify = train_test_split(
    all_examples,
    stratify_safe,
    test_size=0.2,
    random_state=42,
    stratify=stratify_safe
)

# temp → val + test po 10%
val_examples, test_examples = train_test_split(
    temp_examples,
    test_size=0.5,
    random_state=42,
    stratify=temp_stratify
)

print(f"Zbiór treningowy: {len(train_examples)}")
print(f"Zbiór walidacyjny: {len(val_examples)}")
print(f"Zbiór testowy:     {len(test_examples)}")


# -------------------------
# Dataset HuggingFace
# -------------------------

train_dataset = Dataset.from_list(train_examples)
val_dataset = Dataset.from_list(val_examples)
test_dataset = Dataset.from_list(test_examples)


def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    new_labels = []

    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label2id.get(labels[word_idx], label2id['O']))
            else:
                tag = labels[word_idx]
                if tag.startswith("B-"):
                    tag = "I-" + tag[2:]
                label_ids.append(label2id.get(tag, label2id['O']))
            previous_word_idx = word_idx

        new_labels.append(label_ids)

    tokenized["labels"] = new_labels
    return tokenized


print("\nTokenizacja danych...")
train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)
test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)


# -------------------------
# Trening
# -------------------------

args = TrainingArguments(
    output_dir="./herbert-ner",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    save_strategy="no",
    load_best_model_at_end=False,
    logging_steps=50,
    metric_for_best_model="f1",
    greater_is_better=True
)

data_collator = DataCollatorForTokenClassification(tokenizer)

def compute_metrics(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    true_predictions = []

    for pred_seq, label_seq in zip(predictions, labels):
        curr_true = []
        curr_pred = []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                curr_true.append(id2label[l])
                curr_pred.append(id2label[p])
        true_labels.append(curr_true)
        true_predictions.append(curr_pred)

    f1 = f1_score(true_labels, true_predictions)
    return {"f1": f1}


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\nRozpoczynanie treningu...")
trainer.train()

print("\nEvaluacja na zbiorze testowym:")
metrics = trainer.evaluate(test_dataset)
print(metrics)

print("\nZapis modelu oraz tokenizera...")
trainer.save_model("./herbert-ner")
tokenizer.save_pretrained("./herbert-ner")

print("\n✅ Training finished!")
print(f"📁 Model zapisany w: ./herbert-ner")

import json
with open("./herbert-ner/label_mapping.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, indent=2)

print("📝 Zapisano mapowanie etykiet w: ./herbert-ner/label_mapping.json")