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
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter
from transformers import TrainerCallback
import optuna


model_name = "allegro/herbert-base-cased"

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


# --------------------------------------------
# WCZYTYWANIE DANYCH
# --------------------------------------------

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


def stratified_shuffle_split(examples, labels, test_size, random_state):
    if len(examples) <= 1:
        return list(examples), [], list(labels), []

    indices = np.arange(len(examples))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    try:
        train_idx, test_idx = next(splitter.split(indices, labels))
    except ValueError as exc:
        print(f"⚠️  {exc} – fallback na losowy podział bez stratyfikacji.")
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)
        split_point = int(round(len(indices) * (1 - test_size)))
        max_split = len(indices) - 1
        if max_split <= 0:
            split_point = 0
        else:
            split_point = min(max(split_point, 1), max_split)
        train_idx, test_idx = indices[:split_point], indices[split_point:]

    train_examples = [examples[i] for i in train_idx]
    test_examples = [examples[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]

    return train_examples, test_examples, train_labels, test_labels


print("Wczytywanie danych z ner_dataset.conll...")
all_examples = load_conll_file("data/ner_dataset.conll")
print(f"Wczytano {len(all_examples)} przykładów")

stratify_labels = [get_sentence_label_distribution(ex) for ex in all_examples]
stratify_safe = prepare_stratify_labels(stratify_labels)

# PODZIAŁ 80/10/10
train_examples, temp_examples, train_stratify, temp_stratify = stratified_shuffle_split(
    all_examples, stratify_safe, test_size=0.4, random_state=42
)

temp_stratify_safe = prepare_stratify_labels(temp_stratify)
val_examples, test_examples, _, _ = stratified_shuffle_split(
    temp_examples, temp_stratify_safe, test_size=0.5, random_state=42
)

print(f"Train: {len(train_examples)}")
print(f"Val:   {len(val_examples)}")
print(f"Test:  {len(test_examples)}")


# --------------------------------------------
# DATASET + TOKENIZACJA
# --------------------------------------------

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


train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)
test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)


# --------------------------------------------
# METRYKI
# --------------------------------------------

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

    return {"f1": f1_score(true_labels, true_predictions)}


# --------------------------------------------
# HYPERPARAMETER SEARCH
# --------------------------------------------

def optuna_hp_space(trial):
    return {
        "learning_rate": trial.suggest_float("learning_rate", 2e-5, 7e-5, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 3, 10),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [4, 8, 16]),
    }


def compute_objective(metrics):
    return metrics["eval_f1"]


# ✔ tworzymy bazowe args do searcha
base_args = TrainingArguments(
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


print("\nRozpoczynam hyperparameter search (Optuna)...")
def model_init():
    return AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id
    )

trainer = Trainer(
    model_init=model_init,
    args=base_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
best_run = trainer.hyperparameter_search(
    direction="maximize",
    hp_space=optuna_hp_space,
    compute_objective=compute_objective,
    n_trials=20,
    backend="optuna",
)

print("\nNajlepsze hiperparametry znalezione przez Optuna:")
print(best_run.hyperparameters)


# --------------------------------------------
# FINALNY TRENING NA BEST PARAMS
# --------------------------------------------

best_args = TrainingArguments(
    output_dir="./herbert-ner-best",
    evaluation_strategy="epoch",
    save_strategy="no",
    load_best_model_at_end=False,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=50,
    **best_run.hyperparameters
)

trainer = Trainer(
    model=model,
    args=best_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("\nRozpoczynanie finalnego treningu...")
trainer.train()


print("\nEvaluacja na zbiorze testowym:")
metrics = trainer.evaluate(test_dataset)
print(metrics)


# --------------------------------------------
# ZAPIS MODELU
# --------------------------------------------

print("\nZapis modelu oraz tokenizera...")
trainer.save_model("./herbert-ner")
tokenizer.save_pretrained("./herbert-ner")

import json
with open("./herbert-ner/label_mapping.json", "w") as f:
    json.dump({"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}, f, indent=2)

print("📝 Zapisano mapowanie etykiet w: ./herbert-ner/label_mapping.json")
print("\n✅ Training finished!")