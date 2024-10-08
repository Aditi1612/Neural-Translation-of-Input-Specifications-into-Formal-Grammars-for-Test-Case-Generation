# Counting Context-Free Grammars

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
source setup_env.sh
```

## General
```
python scripts/preprocess/sample_testcase_from_dataset.py train
python scripts/preprocess/sample_testcase_from_dataset.py test
```

## Validation of Ground-Truth Grammars
```
python scripts/validate/validate_ground_truth.py
```


## Case study 1: Generation

Human-labelled grammars that fail to generate test cases with current
implementation.
```
python scripts/print/generation_failed.py
```

## Case study 2: Parsing

For test cases generated from human-labelled grammars, those that
human-labelled grammars fail to parse.
```
python scripts/print/parsing_failed.py
```
