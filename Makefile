.PHONY: \
	all \
	help \
	prepare-dataset \
	test-ccfg clean-saved \
	test-human-labeled-data \
	test-human-labeled-data-test \
	test-human-labeled-data-train

all:

data/raw:
	python scripts/download_dataset.py

data/unlabeled: data/raw
	python scripts/filter_python_dataset.py

data/solutions: data/unlabeled
	python scripts/generate_python3_solutions.py

test-human-labeled-data: \  ## Test the human-labeled data
		test-human-labeled-data-train test-human-labeled-data-test

test-human-labeled-data-train:  ## Test the human-labeled train data
	python test_labeling.py \
		--labeled-data data/labeled/train.jsonl \
		--testcase data/unlabeled/code_contests_train_python.jsonl

test-human-labeled-data-test:  ## Test the human-labeled test data
	python test_labeling.py \
		--labeled-data data/labeled/test.jsonl \
		--testcase data/unlabeled/code_contests_train_python.jsonl

prepare-dataset:  ## Prepare the dataset
	python scripts/download_dataset.py \
	&& python scripts/filter_python_dataset.py

clean-saved:  ## Clean the saved files except the last checkpoint
	sh scripts/clean_saved.sh

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | sort
