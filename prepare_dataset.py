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
    """Podziel tekst na zdania/akapity. Każda linia to osobne zdanie."""
    sentences = []
    for line in text.split('\n'):
        line = line.strip()
        if line:  # Ignoruj puste linie
            sentences.append(line)
    return sentences


def simple_tokenize(text: str) -> List[str]:
    """Lekka tokenizacja zgodna z HerBERTem z ochroną wrażliwych ciągów."""
    text = re.sub(r'\s+', ' ', text).strip()

    placeholder_counter = 0
    replacements = {}

    def protect(pattern: str, prefix: str, source: str) -> str:
        nonlocal placeholder_counter
        matches = list(re.finditer(pattern, source))
        for match in reversed(matches):
            placeholder = f'__{prefix}_{placeholder_counter}__'
            replacements[placeholder] = match.group(0)
            placeholder_counter += 1
            start, end = match.span()
            source = source[:start] + placeholder + source[end:]
        return source

    # Zachowaj etykiety i wrażliwe ciągi w całości
    text = protect(r'\[[^\]]+\]', 'LABEL', text)
    text = protect(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', 'EMAIL', text)
    text = protect(r'https?://\S+', 'URL', text)

    # Rozdziel typowe znaki interpunkcyjne, ale bez naruszania placeholderów
    text = re.sub(r'([.,;:!?(){}„"\'\/\-\—])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split(' ')

    return [replacements.get(token, token) for token in tokens if token]

def is_common_word(token: str) -> bool:
    """Sprawdź czy token to zwyczajny wyraz (nie jest rzadkim słowem kluczowym)."""
    # Słowa, które często pojawiają się jako słowa kluczowe
    common_words = {'w', 'z', 'na', 'do', 'od', 'i', 'a', 'o', 'że', 'to', 'jak', 
                    'gdy', 'lub', 'ale', 'jeśli', 'po', 'przed', 'przez', 'dla',
                    'jest', 'są', 'by', 'był', 'będzie', 'mam', 'ma'}
    return token.lower() in common_words


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
    all_labels = set(label_map.values())

    orig_tokens = simple_tokenize(orig_text)
    anon_tokens = simple_tokenize(anon_text)

    if debug:
        print("ORIG TOKENS:", orig_tokens)
        print("ANON TOKENS:", anon_tokens)

    result = []
    orig_idx = 0
    anon_idx = 0

    punctuation_tokens = {'.', ',', ';', ':', '!', '?', '-', '(', ')', '/', '–', '—'}
    punctuation_friendly_labels = all_labels | {'PII'}
    stop_words = {
        'w', 'z', 'na', 'do', 'od', 'i', 'a', 'o', 'że', 'to', 'jak', 'gdy', 'lub',
        'ale', 'jeśli', 'po', 'przed', 'przez', 'dla', 'u', 'ze', 'pod', 'nad'
    }
    anchor_window = 10

    def find_anchor_position(sequence: List[str], start_idx: int, anchor_tokens: List[str]) -> int | None:
        """Zwróć indeks pierwszego dopasowania anchor_tokens w sequence zaczynając od start_idx."""
        if not anchor_tokens:
            return None
        anchor_len = len(anchor_tokens)
        max_start = len(sequence) - anchor_len
        if max_start < start_idx:
            return None
        for idx in range(start_idx, max_start + 1):
            if sequence[idx:idx + anchor_len] == anchor_tokens:
                return idx
        return None

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
            entity_tokens: List[str] = []

            # Zbuduj listę tokenów do następnej etykiety, by ustalić kotwicę
            anchor_tokens_raw: List[str] = []
            scan_idx = orig_idx + 1
            while scan_idx < len(orig_tokens):
                candidate = orig_tokens[scan_idx]
                if candidate.startswith('[') and candidate.endswith(']'):
                    break
                anchor_tokens_raw.append(candidate)
                scan_idx += 1

            next_label_token = orig_tokens[scan_idx] if scan_idx < len(orig_tokens) else None
            next_orig = None
            for candidate in anchor_tokens_raw:
                if candidate not in punctuation_tokens:
                    next_orig = candidate
                    break

            # Przygotuj możliwe kotwice do wyszukania w tekście zanonimizowanym
            anchor_candidates: List[List[str]] = []
            if anchor_tokens_raw:
                anchor_candidates.append(anchor_tokens_raw)
                anchor_candidates.append(anchor_tokens_raw[:anchor_window])
            cleaned_anchor = [tok for tok in anchor_tokens_raw if tok not in punctuation_tokens]
            if cleaned_anchor:
                anchor_candidates.append(cleaned_anchor)
                anchor_candidates.append(cleaned_anchor[:anchor_window])
            cleaned_no_stop = [tok for tok in cleaned_anchor if tok.lower() not in stop_words]
            if cleaned_no_stop:
                anchor_candidates.append(cleaned_no_stop)
                anchor_candidates.append(cleaned_no_stop[:anchor_window])

            # Usuń duplikaty zachowując kolejność
            deduped_candidates: List[List[str]] = []
            seen_signatures: set[tuple[str, ...]] = set()
            for candidate in anchor_candidates:
                usable = [tok for tok in candidate if tok]
                signature = tuple(usable)
                if not usable or signature in seen_signatures:
                    continue
                seen_signatures.add(signature)
                deduped_candidates.append(usable)
            anchor_candidates = deduped_candidates

            anchor_idx = None
            for candidate in anchor_candidates:
                pos = find_anchor_position(anon_tokens, anon_idx, candidate)
                if pos is not None:
                    anchor_idx = pos
                    break

            if anchor_idx is not None:
                entity_tokens = anon_tokens[anon_idx:anchor_idx]
                anon_idx = anchor_idx
            else:
                while anon_idx < len(anon_tokens):
                    current_anon = anon_tokens[anon_idx]

                    # Jeśli następny token w oryginale to etykieta (np. brak wartości), przerywamy
                    if next_label_token and next_label_token.startswith('[') and next_label_token.endswith(']'):
                        break

                    # Jeśli dotarliśmy do spodziewanego tokenu po encji, zatrzymaj
                    if next_orig is not None and current_anon == next_orig:
                        break

                    # Stop words nie przerywają już encji – pozwalamy im występować w środku

                    # Obsługa interpunkcji
                    if current_anon in punctuation_tokens and entity_tokens:
                        if current_anon == ',' and label in ['NAME', 'SURNAME', 'EMAIL']:
                            break
                        if label in punctuation_friendly_labels:
                            entity_tokens.append(current_anon)
                            anon_idx += 1
                            if anon_idx < len(anon_tokens) and anon_tokens[anon_idx][0].isdigit():
                                continue
                            if next_orig is not None and anon_idx < len(anon_tokens):
                                if (anon_tokens[anon_idx] == next_orig or
                                        anon_tokens[anon_idx].lower() in stop_words):
                                    break
                            continue
                        anon_idx += 1
                        break

                    entity_tokens.append(current_anon)
                    anon_idx += 1

            # Tagowanie BIO
            if entity_tokens:
                result.append((entity_tokens[0], f'B-{label}'))
                for token in entity_tokens[1:]:
                    result.append((token, f'I-{label}'))

            orig_idx += 1
            continue

        # Inna niezgodność: spróbuj heurystyki - jeśli anon_token występuje gdzieś dalej w orig -> przesuwamy orig
        if anon_token in orig_tokens[orig_idx+1:]:
            result.append((anon_token, 'O'))
            anon_idx += 1
            continue

        # fallback: oznacz anon jako O i przesuwaj oba wskaźniki
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