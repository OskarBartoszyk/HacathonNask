#!/usr/bin/env python3
"""Infer NER tags with the fine-tuned HerBERT model hosted on Hugging Face."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from prepare_dataset import simple_tokenize

PLACEHOLDER_MAP = {
    "NAME": "[name]",
    "SURNAME": "[surname]",
    "AGE": "[age]",
    "DATE_BIRTH": "[date-of-birth]",
    "DATE": "[date]",
    "SEX": "[sex]",
    "RELIGION": "[religion]",
    "POLITICAL": "[political-view]",
    "ETHNICITY": "[ethnicity]",
    "ORIENTATION": "[sexual-orientation]",
    "HEALTH": "[health]",
    "RELATIVE": "[relative]",
    "CITY": "[city]",
    "ADDRESS": "[address]",
    "EMAIL": "[email]",
    "PHONE": "[phone]",
    "PESEL": "[pesel]",
    "DOCUMENT": "[document-number]",
    "COMPANY": "[company]",
    "SCHOOL": "[school-name]",
    "JOB": "[job-title]",
    "BANK_ACCOUNT": "[bank-account]",
    "CREDIT_CARD": "[credit-card-number]",
    "USERNAME": "[username]",
    "SECRET": "[secret]",
    "PII": "[pii]",
}
def _resolve_id2label(model: AutoModelForTokenClassification, local_dir: Path | None) -> dict[int, str]:
    if local_dir:
        mapping_path = local_dir / "label_mapping.json"
        if mapping_path.exists():
            with mapping_path.open("r", encoding="utf-8") as f:
                stored = json.load(f)
            return {int(k): v for k, v in stored["id2label"].items()}

    config_labels = getattr(model.config, "id2label", None)
    if config_labels:
        return {int(k): v for k, v in config_labels.items()}

    raise ValueError(
        "Model nie zawiera mapowania id2label. Upewnij się, że repozytorium Hugging Face zostało poprawnie zapisane."
    )


def predict_tags(
    tokenizer: AutoTokenizer,
    model: AutoModelForTokenClassification,
    tokens: Sequence[str],
    id2label: dict[int, str],
) -> List[str]:
    if not tokens:
        return []
    encoded = tokenizer(
        list(tokens),
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
    )
    with torch.no_grad():
        outputs = model(**encoded)
    predictions = outputs.logits.argmax(dim=-1)[0].tolist()
    word_ids = encoded.word_ids()
    tags: List[str] = []
    prev_word = None
    for idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if word_id != prev_word:
            label_idx = predictions[idx]
            tags.append(id2label[label_idx])
        prev_word = word_id
    return tags


def tokens_to_placeholder_text(tokens: Sequence[str], tags: Sequence[str]) -> str:
    compressed: List[str] = []
    i = 0
    while i < len(tokens):
        tag = tags[i]
        if tag.startswith("B-"):
            label_name = tag[2:]
            placeholder = PLACEHOLDER_MAP.get(label_name, "[pii]")
            compressed.append(placeholder)
            i += 1
            while i < len(tokens) and tags[i].startswith("I-"):
                i += 1
            continue
        compressed.append(tokens[i])
        i += 1
    return detokenize(compressed)


def detokenize(tokens: Iterable[str]) -> str:
    text = " ".join(tokens)
    patterns = [
        (r"\s+([.,;:!?%])", r"\1"),
        (r"\(\s+", "("),
        (r"\s+\)", ")"),
        (r"\s+([\]\}])", r"\1"),
        (r"([\[{])\s+", r"\1"),
        (r"\s+'/", "'"),
        (r"/\s+", "/"),
        (r"\s+\-", "-"),
        (r"\s+„", " „"),
        (r"„\s+", "„"),
        (r"\s+”", "”"),
        (r"\s+'", "'"),
        (r"'\s+", "'"),
    ]
    for pattern, repl in patterns:
        text = re.sub(pattern, repl, text)
    return text.strip()


def write_conll(output_path: Path, tokens_with_tags: List[List[Tuple[str, str]]]) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for sentence in tokens_with_tags:
            if not sentence:
                continue
            for token, tag in sentence:
                f.write(f"{token}\t{tag}\n")
            f.write("\n")


def anonymize_file(
    input_path: Path | str,
    model_dir: Path | str = "Matela7/AnonBert-ENR",
    output_path: Path | str | None = None,
    conll_path: Path | str | None = None,
) -> tuple[Path, Path]:
    """Tag a text file and write anonymized text plus optional CoNLL with labels."""

    input_path = Path(input_path)
    model_dir = Path(model_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku wejściowego: {input_path}")

    local_model_dir: Path | None = None
    if model_dir.exists():
        local_model_dir = model_dir
        source = model_dir
    else:
        # Pozwól użytkownikowi podać identyfikator repozytorium jako string.
        source = str(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForTokenClassification.from_pretrained(source)
    model.eval()
    id2label = _resolve_id2label(model, local_model_dir)

    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_anon{input_path.suffix or '.txt'}")
    else:
        output_path = Path(output_path)

    if conll_path is None:
        conll_path = input_path.with_name(f"{input_path.stem}_tags.conll")
    else:
        conll_path = Path(conll_path)

    anonymized_lines: List[str] = []
    conll_sequences: List[List[Tuple[str, str]]] = []

    with input_path.open("r", encoding="utf-8") as src:
        for line in src:
            stripped = line.strip()
            if not stripped:
                anonymized_lines.append("")
                conll_sequences.append([])
                continue
            tokens = simple_tokenize(stripped)
            tags = predict_tags(tokenizer, model, tokens, id2label)
            conll_sequences.append(list(zip(tokens, tags)))
            anonymized_lines.append(tokens_to_placeholder_text(tokens, tags))

    with output_path.open("w", encoding="utf-8") as dst:
        for line in anonymized_lines:
            dst.write(line + "\n")

    if conll_sequences:
        write_conll(conll_path, conll_sequences)

    print(f"✓ Zapisano tekst z zamienionymi etykietami do: {output_path}")
    if conll_sequences:
        print(f"✓ Zapisano tokeny i tagi w formacie CoNLL do: {conll_path}")

    return output_path, conll_path


if __name__ == "__main__":
    INPUT_PATH = Path("data/test.txt")  # zmień na swój plik
    MODEL_DIR: Path | str = "Matela7/AnonBert-ENR"
    anonymize_file(input_path=INPUT_PATH, model_dir=MODEL_DIR)