.PHONY: \
	all prepare-dataset test-ccfg clean-saved \
	test-human-labeled-data \
	test-human-labeled-data-train \
	test-human-labeled-data-test

all:

data/raw:
	python scripts/download_dataset.py

data/unlabeled: data/raw
	python scripts/filter_python_dataset.py

data/solutions: data/unlabeled
	python scripts/generate_python3_solutions.py

test-human-labeled-data: \
		test-human-labeled-data-train test-human-labeled-data-test

test-human-labeled-data-train:
	python test_labeling.py \
		--labeled-data data/labeled/train.jsonl \
		--testcase data/unlabeled/code_contests_train_python.jsonl

test-human-labeled-data-test:
	python test_labeling.py \
		--labeled-data data/labeled/test.jsonl \
		--testcase data/unlabeled/code_contests_train_python.jsonl

prepare-dataset:
	python scripts/download_dataset.py \
	&& python scripts/filter_python_dataset.py

clean-saved:
	sh scripts/clean_saved.sh
