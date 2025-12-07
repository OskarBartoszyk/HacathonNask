# Data Preprocessing and Dataset Creation

## Overview

This document describes the comprehensive data preprocessing pipeline developed for creating a Named Entity Recognition (NER) dataset for Polish sensitive information detection. The pipeline transforms paired documents (original with PII labels and anonymized with replacement values) into training-ready datasets in both JSONL and CoNLL formats.

## Data Sources

### Input Files

1. **Original Text (`data/orig.txt`)**
   - Contains text with PII placeholders marked with brackets (e.g., `[name]`, `[city]`, `[email]`)
   - ~3,337 lines of Polish administrative and technical documents
   - Includes building permits, legal decisions, technical specifications, and administrative correspondence

2. **Anonymized Text (`data/anonymized.txt`)**
   - Contains the same documents with PII placeholders replaced by synthetic values
   - Maintains identical document structure (line-by-line correspondence)
   - Example: `[name]` → `Apolonia`, `[city]` → `Oleśnica`, `[phone]` → `574 777 072`

### Output Formats

1. **JSONL Format (`data/ner_dataset.jsonl`)**
   - Each line contains one training example as JSON
   - Structure: `{"id": int, "tokens": [str], "ner_tags": [str]}`
   - Suitable for transformer-based models (HerBERT, BERT)

2. **CoNLL Format (`data/ner_dataset.conll`)**
   - Token-tag pairs separated by tabs
   - Sentences separated by blank lines
   - Standard format for NER tasks

## Entity Labels

The system recognizes 24 distinct PII categories, mapped from bracket notation to uppercase abbreviations:

| Bracket Notation | NER Tag | Category |
|-----------------|---------|----------|
| `[name]` | `NAME` | First name |
| `[surname]` | `SURNAME` | Last name |
| `[age]` | `AGE` | Age |
| `[date-of-birth]` | `DATE_BIRTH` | Date of birth |
| `[date]` | `DATE` | General date |
| `[sex]` | `SEX` | Gender |
| `[religion]` | `RELIGION` | Religious affiliation |
| `[political-view]` | `POLITICAL` | Political views |
| `[ethnicity]` | `ETHNICITY` | Ethnic origin |
| `[sexual-orientation]` | `ORIENTATION` | Sexual orientation |
| `[health]` | `HEALTH` | Health information |
| `[relative]` | `RELATIVE` | Family relation |
| `[city]` | `CITY` | City name |
| `[address]` | `ADDRESS` | Street address |
| `[email]` | `EMAIL` | Email address |
| `[phone]` | `PHONE` | Phone number |
| `[pesel]` | `PESEL` | Polish national ID |
| `[document-number]` | `DOCUMENT` | Document number |
| `[company]` | `COMPANY` | Company name |
| `[school-name]` | `SCHOOL` | School name |
| `[job-title]` | `JOB` | Job title |
| `[bank-account]` | `BANK_ACCOUNT` | Bank account number |
| `[credit-card-number]` | `CREDIT_CARD` | Credit card number |
| `[username]` | `USERNAME` | Username |
| `[secret]` | `SECRET` | Secret information |

## Preprocessing Pipeline

### 1. Sentence Extraction

```python
def extract_sentences(text: str) -> List[str]
```

- Splits input text into individual sentences/lines
- Each line is treated as a separate training example
- Preserves document structure and context
- Filters out empty lines

### 2. Tokenization

The tokenization process is critical for aligning with HerBERT's subword tokenizer and preserving entity boundaries.

#### Protected Pattern Preservation

Before splitting, the tokenizer protects sensitive patterns:

1. **Label Markers**: `[name]`, `[city]`, etc. kept intact
2. **Email Addresses**: Full email preserved as single token
3. **URLs**: HTTP/HTTPS URLs protected from splitting

```python
# Protected patterns are temporarily replaced with placeholders
# Example: user@example.com → __EMAIL_0__
```

#### Punctuation Handling

- Common punctuation marks are separated: `. , ; : ! ? ( ) { } " ' / - – —`
- Whitespace normalized to single spaces
- Protected patterns restored after splitting

#### Polish-Specific Considerations

The tokenizer handles:
- Polish diacritics: `ą ć ę ł ń ó ś ź ż`
- Multi-word entities (e.g., compound surnames)
- Hyphenated words common in Polish names

### 3. Alignment and Tagging

This is the most sophisticated component, implementing a multi-strategy alignment algorithm.

```python
def align_and_tag(orig_text: str, anon_text: str) -> List[Tuple[str, str]]
```

#### Strategy 1: Exact Token Matching

- When tokens match exactly, they receive the `O` (Outside) tag
- Fast path for non-PII content
- Handles ~80% of tokens

#### Strategy 2: Anchor-Based Alignment

When encountering a PII label (e.g., `[name]`), the system:

1. **Identifies anchors**: Looks ahead to find non-PII tokens following the label
2. **Creates anchor candidates**: Multiple versions with varying strictness:
   - Full sequence of following tokens
   - Sequence without punctuation
   - Sequence without stop words
   - Limited window (5 tokens) for performance

3. **Searches for anchor**: Scans anonymized text for anchor position
4. **Extracts entity**: All tokens between current position and anchor

Example:
```
Original: "Sprawę prowadzi [name] [surname] Tel [phone]"
Anonymized: "Sprawę prowadzi Apolonia Kościesza Tel 574 777 072"
Anchor: "Tel" → Entity tokens: ["Apolonia"]
```

#### Strategy 3: Heuristic-Based Extraction

When anchors fail, linguistic heuristics are applied:

**Polish Name Detection**:
- Common first names dictionary (~100 names)
- Capitalization patterns (Title Case)
- Vowel distribution analysis
- Name length constraints

**Polish Surname Detection**:
- Suffix matching (primary): `-ski`, `-ska`, `-cki`, `-cka`, `-wicz`, `-owicz`
- Extended suffixes (60+ patterns): `-czyk`, `-ak`, `-iak`, `-arz`, `-nik`
- Hyphenated surnames (e.g., `Kowalski-Nowak`)
- Uppercase acronym surnames
- Morphological patterns

**Entity Boundary Detection**:
- Punctuation marks signal boundaries (except for email/names with commas)
- Next PII label indicates end
- Stop word filtering
- Context-aware extraction

#### Strategy 4: Name-Surname Promotion

A special rule promotes consecutive `SURNAME` tags to `NAME` when:
1. First token looks like a first name
2. Second token looks like a surname
3. Total word count ≤ 3 (prevents over-matching)

Example:
```
Before: [B-SURNAME] [B-SURNAME]
After:  [B-NAME] [I-NAME]
```

### 4. BIO Tagging Scheme

Entities use the BIO (Begin-Inside-Outside) format:

- **B-{LABEL}**: Beginning of entity (e.g., `B-NAME`)
- **I-{LABEL}**: Inside/continuation of entity (e.g., `I-NAME`)
- **O**: Outside any entity

Example:
```
Token:  ["Jan", "Kowalski", "mieszka", "w", "Warszawie"]
Tag:    ["B-NAME", "I-NAME", "O", "O", "B-CITY"]
```

### 5. Quality Filtering

Examples are filtered based on:
- **Minimum length**: At least 3 tokens
- **Successful alignment**: All tokens must be processed
- **Error handling**: Malformed examples skipped with logging

## Dataset Statistics

After processing ~3,337 sentences:

### Entity Distribution

The dataset contains various PII types with the following approximate distribution:

- **Names & Surnames**: Most frequent (administrative documents)
- **Locations**: Cities, addresses (construction/legal docs)
- **Dates**: Various date formats (legal decisions, permits)
- **Organizations**: Companies, schools (business correspondence)
- **Identifiers**: Phone, email, document numbers
- **Sensitive categories**: Health, political views (less frequent)

### Format Breakdown

**JSONL Dataset**:
- Structured JSON objects for easy parsing
- Compatible with HuggingFace datasets
- Includes sentence IDs for tracking

**CoNLL Dataset**:
- Tab-separated format
- Empty lines separate sentences
- Compatible with standard NER evaluation tools

## Implementation Details

### Performance Considerations

1. **Anchor Window**: Limited to 5 tokens for efficiency
2. **Pattern Caching**: Regex patterns compiled once
3. **Stop Words**: Small set of high-frequency Polish words
4. **Deduplication**: Anchor candidates deduplicated by signature

### Error Handling

```python
try:
    tagged = align_and_tag(orig_sent, anon_sent)
except Exception as e:
    print(f"Error in sentence {i}: {e}")
    continue
```

- Sentence-level error isolation
- Failed sentences logged but don't stop processing
- Graceful degradation for edge cases

### Execution Time

The pipeline processes the entire dataset in approximately:
- JSONL generation: ~2-3 minutes
- CoNLL generation: ~2-3 minutes
- Total: ~5-6 minutes for full dataset

## Usage

### Running the Pipeline

```bash
python prepare_dataset.py
```

### Debug Mode

For detailed alignment information:

```python
create_dataset(orig_file, anon_file, output_file, debug=True)
```

This prints:
- Original tokens
- Anonymized tokens
- Step-by-step alignment decisions

## Output Examples

### JSONL Example

```json
{
  "id": 0,
  "tokens": ["Sprawę", "prowadzi", "Apolonia", "Kościesza", "Tel", "574", "777", "072"],
  "ner_tags": ["O", "O", "B-NAME", "I-NAME", "O", "B-PHONE", "I-PHONE", "I-PHONE"]
}
```

### CoNLL Example

```
Sprawę	O
prowadzi	O
Apolonia	B-NAME
Kościesza	I-NAME
Tel	O
574	B-PHONE
777	I-PHONE
072	I-PHONE

```

## Challenges and Solutions

### Challenge 1: Token Alignment Drift

**Problem**: Original and anonymized texts don't tokenize identically
**Solution**: Multi-strategy alignment with anchoring and fallback heuristics

### Challenge 2: Multi-Token Entities

**Problem**: Names, addresses span multiple tokens
**Solution**: Anchor-based extraction finds entity boundaries

### Challenge 3: Polish Morphology

**Problem**: Complex inflection, surnames with many variants
**Solution**: Extensive suffix database, pattern recognition

### Challenge 4: Name vs. Surname Disambiguation

**Problem**: Consecutive personal names hard to categorize
**Solution**: Look-ahead promotion rules based on Polish name patterns

### Challenge 5: Missing Anchors

**Problem**: Consecutive PII labels with no stable anchors
**Solution**: Heuristic-based extraction with linguistic rules

## Future Improvements

1. **Subword Alignment**: Direct alignment with HerBERT's WordPiece tokenizer
2. **Active Learning**: Flag uncertain alignments for manual review
3. **Cross-Validation**: Systematic validation against manually annotated subset
4. **Augmentation**: Synthetic variations of entities for data diversity
5. **Multilingual**: Extend patterns for multilingual PII detection

## References

- HerBERT: [Allegro/herbert-base-cased](https://huggingface.co/allegro/herbert-base-cased)
- CoNLL Format: [CoNLL-2003 Shared Task](https://www.aclweb.org/anthology/W03-0419/)
- BIO Tagging: Ramshaw & Marcus (1995)
