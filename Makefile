.PHONY: \
	all \
	help \
	prepare-dataset \
	test-ccfg clean-saved \
	test-human-labeled-data \
	test-human-labeled-data-test \
	test-human-labeled-data-train \
	validate-bard-grammar \
	validate-bard-grammar-1-shot \
	validate-bard-grammar-5-shot

all:

data/raw:
	python scripts/download_dataset.py

data/unlabeled: data/raw
	python scripts/filter_python_dataset.py

data/solutions: data/unlabeled
	python scripts/generate_python3_solutions.py

test-human-labeled-data: test-human-labeled-data-train test-human-labeled-data-test  ## Test the human-labeled data

test-human-labeled-data-train:  ## Test the human-labeled train data
	python validate_labeling.py \
		--labeled-data data/labeled/train.jsonl \
		--testcase data/unlabeled/code_contests_train_python.jsonl

test-human-labeled-data-test:  ## Test the human-labeled test data
	python validate_labeling.py \
		--labeled-data data/labeled/test.jsonl \
		--testcase data/unlabeled/code_contests_train_python.jsonl

validate-bard-grammar: validate-bard-grammar-1-shot validate-bard-grammar-5-shot  ## Test the bard grammar

validate-bard-grammar-1-shot:  ## Test the bard grammar with 1-shot
	python validate_labeling.py \
		--labeled-data data/bard_labeled/test_1_shot.jsonl \
		--testcase data/unlabeled/code_contests_test_python.jsonl

validate-bard-grammar-5-shot:  ## Test the bard grammar with 5-shot
	python validate_labeling.py \
		--labeled-data data/bard_labeled/test_5_shot.jsonl \
		--testcase data/unlabeled/code_contests_test_python.jsonl

prepare-dataset:  ## Prepare the dataset
	python scripts/download_dataset.py \
	&& python scripts/filter_python_dataset.py

clean-saved:  ## Clean the saved files except the last checkpoint
	sh scripts/clean_saved.sh

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | sort
