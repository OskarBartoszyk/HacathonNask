#!/usr/bin/env python3
"""Infer NER tags with the fine-tuned HerBERT model and optionally anonymize text."""

from __future__ import annotations

import argparse
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


def load_label_mapping(model_dir: Path) -> Tuple[dict[int, str], dict[str, int]]:
    mapping_path = model_dir / "label_mapping.json"
    if mapping_path.exists():
        with mapping_path.open("r", encoding="utf-8") as f:
            stored = json.load(f)
        id2label = {int(k): v for k, v in stored["id2label"].items()}
        label2id = {k: int(v) for k, v in stored["label2id"].items()}
    else:
        raise FileNotFoundError(
            f"Nie znaleziono {mapping_path}. Upewnij się, że po treningu zapisano mapowanie etykiet."
        )
    return id2label, label2id


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Taguj i anonimizuj tekst za pomocą modelu HerBERT NER.")
    parser.add_argument("input", type=Path, help="Plik z tekstem do oznaczenia (jeden akapit na linię).")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("./herbert-ner"),
        help="Katalog z zapisanym modelem oraz label_mapping.json.",
    )
    parser.add_argument("--output", type=Path, help="Plik wyjściowy z tekstem po anonimizacji.")
    parser.add_argument(
        "--conll-output",
        type=Path,
        help="Opcjonalny plik w formacie CoNLL z tokenami i etykietami.",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku wejściowego: {args.input}")

    id2label, _ = load_label_mapping(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.eval()

    anonymized_lines: List[str] = []
    conll_sequences: List[List[Tuple[str, str]]] = []

    with args.input.open("r", encoding="utf-8") as src:
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

    if args.output:
        with args.output.open("w", encoding="utf-8") as dst:
            for line in anonymized_lines:
                dst.write(line + "\n")
    else:
        for line in anonymized_lines:
            print(line)

    if args.conll_output:
        write_conll(args.conll_output, conll_sequences)


if __name__ == "__main__":
    main()
