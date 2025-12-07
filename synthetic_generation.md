# Synthetic Data Generation

## Overview

This document describes the synthetic data generation mechanism used to replace PII placeholders (tags like `[name]`, `[city]`, `[email]`) with realistic, fake values. This process is essential for creating training data that maintains natural language patterns while protecting real personal information.

## Generation Mechanism

### Core Library: Faker

The synthetic data generation is powered by the **Faker** library, a widely-used Python package for generating fake but realistic data.

**Key advantages:**
- Locale-specific generation (`pl_PL` for Polish)
- Consistent API across different data types
- Realistic patterns and formats
- No external API dependencies

### Implementation

```python
from faker import Faker

class Anonimizer:
    def __init__(self, locale='pl_PL'):
        self.faker = Faker(locale)
```

The `Anonimizer` class wraps Faker functionality and provides specialized methods for each PII category.

## Data Sources

### 1. Built-in Faker Providers

Faker includes locale-specific providers for:

- **Polish names**: `faker.first_name()`, `faker.last_name()`
- **Polish cities**: `faker.city()`
- **Polish addresses**: `faker.address()`
- **Phone numbers**: `faker.phone_number()` (Polish format)
- **Email addresses**: `faker.email()`
- **Dates**: `faker.date_of_birth()`, `faker.date_between()`
- **Company names**: `faker.company()`
- **Job titles**: `faker.job()`

### 2. Custom Generators

For Polish-specific identifiers not in Faker:

#### PESEL Number
```python
def _generate_pesel(self):
    """Generates random 11-digit PESEL (Polish national ID)"""
    return ''.join([str(random.randint(0, 9)) for _ in range(11)])
```

Note: This is a simplified version that generates valid-looking numbers but doesn't implement the full PESEL checksum algorithm.

#### Document Number
```python
def _generate_document_number(self):
    """Generates ID document number (e.g., ABC123456)"""
    letters = ''.join([chr(random.randint(65, 90)) for _ in range(3)])
    numbers = ''.join([str(random.randint(0, 9)) for _ in range(6)])
    return f"{letters}{numbers}"
```

#### Bank Account (IBAN)
```python
def _generate_bank_account(self):
    """Generates Polish IBAN (26 digits after PL prefix)"""
    numbers = ''.join([str(random.randint(0, 9)) for _ in range(26)])
    return f"PL{numbers}"
```

### 3. Predefined Lists

For categorical data with limited options:

```python
# Gender
random.choice(['Mężczyzna', 'Kobieta', 'Inna'])

# Religion
random.choice(['Katolicyzm', 'Protestantyzm', 'Islam', 'Judaizm', 
               'Buddyzm', 'Hinduizm', 'Ateizm', 'Agnostycyzm'])

# Political views
random.choice(['Lewicowe', 'Centrolewicowe', 'Centrowe', 
               'Centroprawicowe', 'Prawicowe', 'Libertariańskie', 'Apolityczne'])

# Ethnicity
random.choice(['Polska', 'Ukraińska', 'Białoruska', 'Romska', 
               'Niemiecka', 'Rosyjska', 'Litewska'])

# Sexual orientation
random.choice(['Heteroseksualna', 'Homoseksualna', 'Biseksualna', 
               'Aseksualna', 'Panseksualna'])

# Health status
random.choice(['Dobry', 'Bardzo dobry', 'Średni', 
               'Wymaga leczenia', 'Chroniczna choroba'])
```

## Tag Mapping

Complete mapping of PII tags to generation functions:

| Tag | Generation Method | Example Output |
|-----|------------------|----------------|
| `[name]` | `faker.first_name()` | `Apolonia` |
| `[surname]` | `faker.last_name()` | `Kościesza` |
| `[age]` | `random.randint(18, 90)` | `55` |
| `[date-of-birth]` | `faker.date_of_birth()` | `1968-03-15` |
| `[date]` | `faker.date_between()` | `2023-11-20` |
| `[sex]` | `random.choice([...])` | `Kobieta` |
| `[religion]` | `random.choice([...])` | `Katolicyzm` |
| `[political-view]` | `random.choice([...])` | `Centrowe` |
| `[ethnicity]` | `random.choice([...])` | `Polska` |
| `[sexual-orientation]` | `random.choice([...])` | `Heteroseksualna` |
| `[health]` | `random.choice([...])` | `Dobry` |
| `[relative]` | `f"{faker.first_name()} {faker.last_name()}"` | `Jan Kowalski` |
| `[city]` | `faker.city()` | `Oleśnica` |
| `[address]` | `faker.address()` | `al. Słonecznikowa 27, 10-776 Elbląg` |
| `[email]` | `faker.email()` | `user@example.com` |
| `[phone]` | `faker.phone_number()` | `574 777 072` |
| `[pesel]` | `_generate_pesel()` | `55020212345` |
| `[document-number]` | `_generate_document_number()` | `ABC123456` |
| `[company]` | `faker.company()` | `Gabinety Wiese` |
| `[school-name]` | Custom template | `Liceum Kowalski` |
| `[job-title]` | `faker.job()` | `Inżynier` |
| `[bank-account]` | `_generate_bank_account()` | `PL12345678901234567890123456` |
| `[credit-card-number]` | `faker.credit_card_number()` | `4532-1234-5678-9012` |
| `[username]` | `faker.user_name()` | `jan.kowalski` |
| `[secret]` | `faker.password()` | `aB3$xY9#kL2m` |

## The Polish Inflection Challenge

### The Problem

Polish is a highly inflected language with 7 grammatical cases. The same word changes form based on its grammatical role in the sentence.

**Critical example:**
```
Context: "Mieszkam w [city]"
Nominative (dictionary form): Radom
Locative (after "w"): Radomiu  ✓ CORRECT
```

If we naively replace `[city]` with the nominative form `Radom`, we get:
```
"Mieszkam w Radom" ❌ GRAMMATICALLY INCORRECT
```

The correct form requires the locative case:
```
"Mieszkam w Radomiu" ✓ GRAMMATICALLY CORRECT
```

### Our Approach: Acceptance of Limitations

**Current implementation: No inflection handling**

Our solution generates words in their **nominative (dictionary) form** without grammatical case adaptation. This is a conscious trade-off:

#### Why we don't handle inflection:

1. **Complexity**: Polish inflection rules are extremely complex
   - Different patterns for each noun type
   - Gender-specific endings
   - Irregular forms
   - Context-dependent modifications

2. **Library limitations**: Faker generates nominative forms only
   - No built-in case inflection for Polish
   - Would require external morphological analyzers (e.g., Morfeusz, spaCy)

3. **Model robustness**: Our NER model (HerBERT) is trained to handle:
   - Real-world text with occasional errors
   - Multiple grammatical forms of the same entity
   - Context from surrounding words

4. **Task focus**: NER is about **entity recognition**, not grammatical correctness
   - The model learns patterns from context
   - Entity boundaries matter more than exact inflection
   - Real documents also contain grammatical errors

### Example Scenarios

#### Scenario 1: City Names
```
Input:  "Decyzja została wydana w [city]"
Output: "Decyzja została wydana w Oleśnica"
Correct: "Decyzja została wydana w Oleśnicy"
Status: ⚠️ GRAMMATICALLY IMPERFECT
```

#### Scenario 2: Names in Different Cases
```
Input:  "Sprawę prowadzi [name] [surname]"
Output: "Sprawę prowadzi Apolonia Kościesza"
Status: ✓ CORRECT (nominative is appropriate here)

Input:  "Spotkanie z [name] [surname]"
Output: "Spotkanie z Jan Kowalski"
Correct: "Spotkanie z Janem Kowalskim"
Status: ⚠️ GRAMMATICALLY IMPERFECT
```

#### Scenario 3: Dates and Numbers
```
Input:  "Data urodzenia: [date-of-birth]"
Output: "Data urodzenia: 1968-03-15"
Status: ✓ CORRECT (no inflection needed)
```

### Future Solutions (Not Implemented)

If inflection were to be added, potential approaches:

1. **Morphological Libraries**
   ```python
   import morfeusz2
   # Analyze context and apply appropriate case
   ```

2. **Rule-Based Templates**
   ```python
   context_patterns = {
       r'w ([city])': lambda c: locative_case(c),
       r'z ([city])': lambda c: instrumental_case(c),
   }
   ```

3. **Neural Language Models**
   ```python
   # Use GPT/T5 to generate contextually correct forms
   from transformers import pipeline
   filler = pipeline("fill-mask", model="allegro/herbert-large")
   ```

4. **Hybrid Approach**
   - Use regex to detect prepositions
   - Apply case transformations based on preposition type
   - Fall back to nominative for ambiguous cases

## Data Realism and Consistency

### Input Independence

**Current implementation: Full randomization**

Each tag replacement is **independent** and **random**:

```python
while re.search(pattern, result):
    result = re.sub(pattern, generator(), result, count=1)
```

**Characteristics:**
- ✓ Maximum privacy protection (no link to original data)
- ✓ High diversity in generated dataset
- ✓ Prevents memorization of specific individuals
- ⚠️ No consistency between related fields
- ⚠️ No preservation of original data characteristics

### Example: Full Randomization

Original with tags:
```
"Kierownik [name] [surname] (wiek: [age]) mieszka w [city]"
```

Generated (multiple runs produce different results):
```
Run 1: "Kierownik Jan Kowalski (wiek: 45) mieszka w Warszawa"
Run 2: "Kierownik Maria Nowak (wiek: 62) mieszka w Kraków"
Run 3: "Kierownik Piotr Wiśniewski (wiek: 29) mieszka w Gdańsk"
```

### No Cross-Field Consistency

The system does **not** maintain logical relationships:

```
Input:  "[name] [surname] (wiek: [age], płeć: [sex])"
Output: "Apolonia Kowalski (wiek: 55, płeć: Mężczyzna)"
```

Issues:
- Female first name + masculine surname ⚠️
- Age/gender combinations may be statistically unusual
- No historical date consistency (birth date vs. age)

### Benefits of This Approach

1. **Privacy-preserving**: No correlation with real data
2. **Diverse training data**: Wide variety of combinations
3. **Simple implementation**: No complex state management
4. **Reproducible**: Easy to regenerate on demand

### Potential Improvements (Not Implemented)

1. **Consistent Person Generator**
   ```python
   def generate_consistent_person():
       gender = random.choice(['M', 'F'])
       age = random.randint(18, 90)
       if gender == 'M':
           name = faker.first_name_male()
           surname = faker.last_name_male()
       else:
           name = faker.first_name_female()
           surname = faker.last_name_female()
       return {"name": name, "surname": surname, "age": age, "gender": gender}
   ```

2. **Date Consistency**
   ```python
   birth_date = faker.date_of_birth(minimum_age=age, maximum_age=age+1)
   ```

3. **Geographic Consistency**
   ```python
   city = faker.city()
   address = faker.address_in_city(city)  # Hypothetical
   phone = faker.phone_number_in_city(city)  # Area codes match
   ```

## Quality Assurance

### Format Validation

Generated data follows expected formats:

- **Phone numbers**: Polish formats (`+48 XXX XXX XXX`, `XX XXX XX XX`)
- **Dates**: ISO format (`YYYY-MM-DD`)
- **Email**: Valid RFC format (`user@domain.com`)
- **PESEL**: 11 digits
- **IBAN**: `PL` + 26 digits

### Realistic Distributions

- **Names**: From Polish name frequency distributions
- **Cities**: Actual Polish city names
- **Ages**: Reasonable range (18-90 years)
- **Dates**: Recent 5-year window by default

## Example Transformations

### Before (with tags):
```
Sprawę prowadzi [name] [surname] Tel [phone] pok 005
```

### After (anonymized):
```
Sprawę prowadzi Apolonia Kościesza Tel 574 777 072 pok 005
```

### Before (complex example):
```
Decyzję Burmistrza Miasta i Gminy [city] z dnia [date] o środowiskowych 
uwarunkowaniach zgody na realizację przedsięwzięcia
```

### After (anonymized):
```
Decyzję Burmistrza Miasta i Gminy Wieluń z dnia 2005-10-12 o środowiskowych 
uwarunkowaniach zgody na realizację przedsięwzięcia
```

## Generation Statistics

From the test dataset (`data/test_anon_fake.txt`):

- **Total lines processed**: 3,337
- **Unique first names generated**: ~100+
- **Unique surnames generated**: ~150+
- **Unique cities**: ~50+
- **Phone numbers**: All unique (random generation)
- **PESEL numbers**: All unique (random generation)

## Comparison: Tag-based vs. Synthetic

| Aspect | Tag-based (`[city]`) | Synthetic (`Warszawa`) |
|--------|---------------------|----------------------|
| Privacy | ✓ Maximum | ✓ Maximum |
| Readability | ⚠️ Technical | ✓ Natural |
| NER Training | ✓ Clear labels | ✓ Realistic context |
| Grammar | N/A | ⚠️ No inflection |
| Consistency | N/A | ⚠️ Independent fields |
| Diversity | Limited | ✓ High |

## Usage

### Basic Usage
```python
from anonbert import anonimizer

# Create anonymizer with Polish locale
anon = anonimizer.Anonimizer(locale='pl_PL')

# Read text with tags
anon.ReadText('data/orig.txt')

# Generate synthetic data
synthetic_text = anon.FakeFillAll()
```

### Processing Pipeline
```python
# Used in dataset creation
anonimze_with_fakefill(input_path)
# Generates: input_path with "_anon_fake.txt" suffix
```

## Limitations and Trade-offs

### Accepted Limitations

1. **No grammatical case handling** - generates nominative forms only
2. **No cross-field consistency** - each tag replaced independently
3. **No context awareness** - doesn't adapt to surrounding text
4. **Simplified PESEL** - doesn't implement checksum validation
5. **Generic addresses** - may not match real street patterns

### Why These Are Acceptable

For NER model training:
- **Context matters more than perfection**: The model learns from surrounding words
- **Real data has errors too**: Documents often contain typos and grammatical mistakes
- **Entity boundaries are key**: Correct tokenization is more important than inflection
- **Diversity improves generalization**: Random variation prevents overfitting

## Conclusion

Our synthetic data generation prioritizes:
1. ✓ **Privacy**: Complete disconnection from real data
2. ✓ **Simplicity**: Maintainable, understandable code
3. ✓ **Diversity**: Wide variety for robust model training
4. ✓ **Realism**: Plausible formats and patterns

While it sacrifices:
1. ⚠️ **Grammatical perfection**: No inflection handling
2. ⚠️ **Logical consistency**: Independent field generation
3. ⚠️ **Context adaptation**: No sentence-level understanding

This balance is appropriate for our NER training objective, where the model's ability to recognize entity patterns from context is more valuable than perfect grammatical correctness in synthetic training data.

## Future Enhancements

Potential improvements for future iterations:

1. **Basic inflection support** for most common prepositions
2. **Gender-aware name generation** to avoid mismatches
3. **Consistent person profiles** across related fields
4. **Context-aware generation** using language models
5. **Statistical validation** against real Polish text distributions
