#!/usr/bin/env python3
"""
Skrypt do przygotowania datasetu NER w formacie JSONL i CoNLL
z oryginalnego pliku z etykietami [name], [city], itp.
i zanonimizowanego pliku z wartościami.
"""

import json
import re
from typing import List, Tuple


def read_file(filepath: str) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()


def extract_sentences(text: str) -> List[str]:
    """Podziel tekst na zdania/akapity."""
    sentences = []
    current = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            if current:
                sentences.append(' '.join(current))
                current = []
        else:
            current.append(line)
    if current:
        sentences.append(' '.join(current))
    return sentences


def simple_tokenize(text: str) -> List[str]:
    """
    Prosta tokenizacja kompatybilna z HerBERTem.
    WAŻNE: nie rozbijamy nawiasów kwadratowych, żeby [name] pozostało jednym tokenem.
    """
    # Usuń nadmiarowe spacje
    text = re.sub(r'\s+', ' ', text).strip()
    # Rozdziel typowe znaki interpunkcyjne, ale NIE [] (kwadratowe)
    # zostawiamy też cudzysłowy i myślniki w miarę prostym podejściu
    text = re.sub(r'([.,;:!?(){}„"\'\/\-\—])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split(' ')

def align_and_tag(orig_text: str, anon_text: str, debug: bool = False) -> List[Tuple[str, str]]:
    """Dopasuj oryginał do zanonimizowanej wersji i stwórz tagi BIO."""
    label_map = {
        '[name]': 'NAME',
        '[surname]': 'SURNAME',
        '[age]': 'AGE',
        '[date-of-birth]': 'DATE_BIRTH',
        '[date]': 'DATE',
        '[sex]': 'SEX',
        '[religion]': 'RELIGION',
        '[political-view]': 'POLITICAL',
        '[ethnicity]': 'ETHNICITY',
        '[sexual-orientation]': 'ORIENTATION',
        '[health]': 'HEALTH',
        '[relative]': 'RELATIVE',
        '[city]': 'CITY',
        '[address]': 'ADDRESS',
        '[email]': 'EMAIL',
        '[phone]': 'PHONE',
        '[pesel]': 'PESEL',
        '[document-number]': 'DOCUMENT',
        '[company]': 'COMPANY',
        '[school-name]': 'SCHOOL',
        '[job-title]': 'JOB',
        '[bank-account]': 'BANK_ACCOUNT',
        '[credit-card-number]': 'CREDIT_CARD',
        '[username]': 'USERNAME',
        '[secret]': 'SECRET',
    }

    orig_tokens = simple_tokenize(orig_text)
    anon_tokens = simple_tokenize(anon_text)

    if debug:
        print("ORIG TOKENS:", orig_tokens)
        print("ANON TOKENS:", anon_tokens)

    result = []
    orig_idx = 0
    anon_idx = 0

    while orig_idx < len(orig_tokens) and anon_idx < len(anon_tokens):
        orig_token = orig_tokens[orig_idx]
        anon_token = anon_tokens[anon_idx]

        # Jeśli tokeny są identyczne -> O
        if orig_token == anon_token:
            result.append((anon_token, 'O'))
            orig_idx += 1
            anon_idx += 1
            continue

        # Jeżeli ORIG token to etykieta w formacie [label]
        if orig_token.startswith('[') and orig_token.endswith(']'):
            label = label_map.get(orig_token, 'PII')
            entity_tokens = []

            # Konsumuj anon tokeny dopóki nie napotkamy tokenu, który odpowiada
            # następnemu tokenowi w oryginale (lub do końca)
            next_orig = orig_tokens[orig_idx + 1] if orig_idx + 1 < len(orig_tokens) else None

            while anon_idx < len(anon_tokens):
                if next_orig is not None and anon_tokens[anon_idx] == next_orig:
                    break
                # jeśli kolejny orig też jest etykietą - zatrzymaj po dodaniu bieżącego anon (to pokrywa back-to-back labels)
                if next_orig is not None and next_orig.startswith('[') and next_orig.endswith(']'):
                    # ale jeśli anon token jest równy następnej etykiecie (mało prawdopodobne), przerwij
                    if anon_tokens[anon_idx] == next_orig:
                        break
                    # dodaj bieżący anon i zakończ (przy back-to-back label)
                    entity_tokens.append(anon_tokens[anon_idx])
                    anon_idx += 1
                    break

                entity_tokens.append(anon_tokens[anon_idx])
                anon_idx += 1

            # Tagowanie BIO
            if entity_tokens:
                result.append((entity_tokens[0], f'B-{label}'))
                for token in entity_tokens[1:]:
                    result.append((token, f'I-{label}'))
            # przesuwamy orig do następnego tokenu (etykieta jako jeden token)
            orig_idx += 1
            continue

        # Inna niezgodność: spróbuj heurystyki - jeśli anon_token występuje gdzieś dalej w orig -> przesuwamy orig
        # (proste przeskakiwanie, żeby uniknąć utknięcia)
        if anon_token in orig_tokens[orig_idx+1:]:
            result.append((anon_token, 'O'))
            anon_idx += 1
            continue

        # fallback: oznacz anon jako O i przesuwaj oba wskaźniki jeśli wyglądają podobnie
        result.append((anon_token, 'O'))
        anon_idx += 1

    # Pozostałe anon tokeny -> O
    while anon_idx < len(anon_tokens):
        result.append((anon_tokens[anon_idx], 'O'))
        anon_idx += 1

    return result


def create_dataset(orig_file: str, anon_file: str, output_file: str, debug: bool = False):
    orig_text = read_file(orig_file)
    anon_text = read_file(anon_file)

    orig_sentences = extract_sentences(orig_text)
    anon_sentences = extract_sentences(anon_text)

    if len(orig_sentences) != len(anon_sentences):
        print(f"UWAGA: różna liczba zdań: orig={len(orig_sentences)}, anon={len(anon_sentences)}")

    dataset = []

    for i, (orig_sent, anon_sent) in enumerate(zip(orig_sentences, anon_sentences)):
        try:
            tagged = align_and_tag(orig_sent, anon_sent, debug=debug)
            if not tagged:
                continue

            tokens = [t[0] for t in tagged]
            tags = [t[1] for t in tagged]

            if len(tokens) < 3:
                continue

            dataset.append({'id': i, 'tokens': tokens, 'ner_tags': tags})
        except Exception as e:
            print(f"Błąd zdania {i}: {e}")
            continue

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✓ Utworzono dataset: {len(dataset)} przykładów")
    print(f"✓ Zapisano do: {output_file}")

    if dataset:
        all_tags = [tag for item in dataset for tag in item['ner_tags']]
        unique_tags = sorted(set(all_tags))
        print(f"\n📊 Statystyki: {len(unique_tags)} unikalnych tagów")
        for tag in unique_tags:
            if tag != 'O':
                count = all_tags.count(tag)
                print(f"  {tag}: {count}")


def create_conll_format(orig_file: str, anon_file: str, output_file: str):
    orig_text = read_file(orig_file)
    anon_text = read_file(anon_file)

    orig_sentences = extract_sentences(orig_text)
    anon_sentences = extract_sentences(anon_text)

    with open(output_file, 'w', encoding='utf-8') as f:
        for orig_sent, anon_sent in zip(orig_sentences, anon_sentences):
            try:
                tagged = align_and_tag(orig_sent, anon_sent)
                if not tagged or len(tagged) < 3:
                    continue
                for token, tag in tagged:
                    f.write(f"{token}\t{tag}\n")
                f.write("\n")
            except Exception:
                continue

    print(f"✓ Utworzono dataset CoNLL: {output_file}")


if __name__ == "__main__":
    orig_file = "data/orig.txt"          # plik z etykietami [name], [city], ...
    anon_file = "data/anonymized.txt"    # zanonimizowana wersja
    output_jsonl = "data/ner_dataset.jsonl"
    output_conll = "data/ner_dataset.conll"

    print("🚀 Tworzenie datasetu dla HerBERTa...")
    print(f"📂 Oryginalny (z etykietami): {orig_file}")
    print(f"📂 Zanonimizowany (wartości): {anon_file}\n")

    create_dataset(orig_file, anon_file, output_jsonl, debug=False)
    print()
    create_conll_format(orig_file, anon_file, output_conll)
    print("\n✅ Gotowe!")