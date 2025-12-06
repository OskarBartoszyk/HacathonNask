from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import load_dataset, Dataset
import numpy as np
from seqeval.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from collections import Counter


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


tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=NUM_LABELS,
    id2label=id2label,
    label2id=label2id
)


def load_conll_file(filepath):
    """Wczytuje plik CoNLL i zwraca listę przykładów (zdań)."""
    examples = []
    current_tokens = []
    current_labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            if not line:  # pusta linia = koniec zdania
                if current_tokens:
                    examples.append({
                        'tokens': current_tokens,
                        'ner_tags': current_labels
                    })
                    current_tokens = []
                    current_labels = []
            else:
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        token = parts[0]
                        tag = parts[1]
                        current_tokens.append(token)
                        current_labels.append(tag)
                except:
                    continue
        
        # dodaj ostatnie zdanie jeśli nie było pustej linii na końcu
        if current_tokens:
            examples.append({
                'tokens': current_tokens,
                'ner_tags': current_labels
            })
    
    return examples


def get_sentence_label_distribution(example):
    """
    Zwraca główną etykietę dla zdania (poza 'O').
    Używane do stratyfikacji - szukamy najważniejszej encji w zdaniu.
    Grupuje rzadkie klasy do kategorii 'OTHER' aby umożliwić stratyfikację.
    """
    # Zlicz wszystkie etykiety B- (początek encji)
    label_counts = Counter()
    for tag in example['ner_tags']:
        if tag.startswith('B-'):
            label_counts[tag] = label_counts.get(tag, 0) + 1
    
    # Jeśli są jakieś encje, zwróć najczęstszą
    if label_counts:
        return label_counts.most_common(1)[0][0]
    
    # Jeśli tylko 'O', zwróć 'O'
    return 'O'


def prepare_stratify_labels(labels, min_count=2):
    """
    Przygotowuje etykiety do stratyfikacji.
    Dla klas z < min_count przykładów przypisuje specjalną etykietę 'RARE_CLASS'.
    Dla pozostałych klas zwraca oryginalną etykietę.
    """
    label_counts = Counter(labels)
    stratify_safe = []
    
    for label in labels:
        if label_counts[label] < min_count:
            # Klasy z < 2 przykładami grupujemy do 'RARE_CLASS' tylko na potrzeby stratyfikacji
            stratify_safe.append('RARE_CLASS')
        else:
            stratify_safe.append(label)
    
    return stratify_safe


# Wczytaj dane z pojedynczego pliku
print("Wczytywanie danych z ner_dataset.conll...")
all_examples = load_conll_file("ner_dataset.conll")
print(f"Wczytano {len(all_examples)} przykładów (zdań)")

# Stwórz etykiety stratyfikacji dla każdego przykładu
stratify_labels = [get_sentence_label_distribution(ex) for ex in all_examples]

# Wyświetl statystyki
print("\nRozkład klas w całym zbiorze:")
label_dist = Counter(stratify_labels)
for label, count in label_dist.most_common():
    print(f"  {label}: {count} zdań ({count/len(all_examples)*100:.1f}%)")

# Przygotuj etykiety do stratyfikacji (rzadkie klasy = None)
stratify_safe = prepare_stratify_labels(stratify_labels, min_count=2)

# Zlicz ile klas może być stratyfikowanych
if stratify_safe:
    stratifiable = sum(1 for l in stratify_safe if l is not None)
    print(f"\nKlasy możliwe do stratyfikacji: {stratifiable}/{len(stratify_safe)} zdań")
    rare_classes = [label for label, count in label_dist.items() if count < 2]
    if rare_classes:
        print(f"Klasy z <2 przykładami (dzielone losowo): {', '.join(rare_classes)}")

# Wykonaj podział train/test (80/20)
print("\nWykonywanie podziału train/test (80/20) ze stratyfikacją dla częstych klas...")
train_examples, test_examples = train_test_split(
    all_examples,
    test_size=0.2,
    random_state=42,
    stratify=stratify_safe
)

print(f"Zbiór treningowy: {len(train_examples)} przykładów")
print(f"Zbiór testowy: {len(test_examples)} przykładów")

# Pobierz etykiety dla zbiorów train i test
train_labels = [get_sentence_label_distribution(ex) for ex in train_examples]
test_labels = [get_sentence_label_distribution(ex) for ex in test_examples]

# Wyświetl rozkład w zbiorach train i test
print("\nRozkład klas w zbiorze treningowym:")
train_dist = Counter(train_labels)
for label, count in train_dist.most_common():
    print(f"  {label}: {count} zdań ({count/len(train_examples)*100:.1f}%)")

print("\nRozkład klas w zbiorze testowym:")
test_dist = Counter(test_labels)
for label, count in test_dist.most_common():
    print(f"  {label}: {count} zdań ({count/len(test_examples)*100:.1f}%)")

# Sprawdź, które klasy są obecne w train/test
all_unique_labels = set(label_dist.keys())
train_unique_labels = set(train_dist.keys())
test_unique_labels = set(test_dist.keys())

missing_in_train = all_unique_labels - train_unique_labels
missing_in_test = all_unique_labels - test_unique_labels

if missing_in_test:
    print(f"\n⚠️  Klasy nieobecne w zbiorze testowym: {', '.join(missing_in_test)}")
if missing_in_train:
    print(f"\n⚠️  Klasy nieobecne w zbiorze treningowym: {', '.join(missing_in_train)}")

# Konwertuj do formatu Dataset
train_dataset = Dataset.from_list(train_examples)
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


print("\nTokenizacja danych...")
train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
test_dataset = test_dataset.map(tokenize_and_align_labels, batched=True)

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

print("\nRozpoczynanie treningu...")
trainer.train()

print("\nTraining finished!")
print(f"Model zapisany w: ./herbert-ner")