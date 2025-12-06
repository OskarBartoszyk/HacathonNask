from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import load_dataset
import numpy as np
from seqeval.metrics import f1_score, classification_report

# ===========================
# CONFIG
# ===========================
model_name = "allegro/herbert-base-cased"

LABEL_LIST = [
    "O",

    # 1. Dane identyfikacyjne osobowe
    "B-name", "I-name",
    "B-surname", "I-surname",
    "B-age", "I-age",
    "B-date-of-birth", "I-date-of-birth",
    "B-date", "I-date",
    "B-sex", "I-sex",
    "B-religion", "I-religion",
    "B-political-view", "I-political-view",
    "B-ethnicity", "I-ethnicity",
    "B-sexual-orientation", "I-sexual-orientation",
    "B-health", "I-health",
    "B-relative", "I-relative",

    # 2. Dane kontaktowe i lokalizacyjne
    "B-city", "I-city",
    "B-address", "I-address",
    "B-email", "I-email",
    "B-phone", "I-phone",

    # 3. Dokumenty i ID
    "B-pesel", "I-pesel",
    "B-document-number", "I-document-number",

    # 4. Dane zawodowe i edukacyjne
    "B-company", "I-company",
    "B-school-name", "I-school-name",
    "B-job-title", "I-job-title",

    # 5. Finansowe
    "B-bank-account", "I-bank-account",
    "B-credit-card-number", "I-credit-card-number",

    # 6. Identyfikatory cyfrowe i loginy
    "B-username", "I-username",
    "B-secret", "I-secret"
]

NUM_LABELS = len(LABEL_LIST)
label2id = {l: i for i, l in enumerate(LABEL_LIST)}
id2label = {i: l for i, l in enumerate(LABEL_LIST)}

# ===========================
# TOKENIZER + MODEL
# ===========================
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id
)

# ===========================
# LOADING DATA (ConLL)
# ===========================
dataset = load_dataset("text", data_files={
    "train": "train.conll",
    "test": "test.conll"
})

def parse_conll(example):
    tokens = []
    labels = []

    for line in example["text"].split("\n"):
        if not line.strip():
            continue

        try:
            token, tag = line.split()
        except:
            continue

        tokens.append(token)
        labels.append(tag)

    return {"tokens": tokens, "ner_tags": labels}


dataset = dataset.map(parse_conll)

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
                label_ids.append(label2id[labels[word_idx]])
            else:
                # I-tag if continuation
                tag = labels[word_idx]
                if tag.startswith("B-"):
                    tag = "I-" + tag[2:]
                label_ids.append(label2id[tag])
            previous_word_idx = word_idx

        new_labels.append(label_ids)

    tokenized["labels"] = new_labels
    return tokenized


tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)

train_dataset = tokenized_dataset["train"]
test_dataset = tokenized_dataset["test"]

args = TrainingArguments(
    output_dir="./herbert-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    logging_steps=50
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

    return {
        "f1": f1_score(true_labels, true_predictions),
    }


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

print("Training finished!")