.PHONY: all prepare-dataset test-ccfg clean-saved

all:

data/raw:
	python scripts/download_dataset.py

data/unlabeled: data/raw
	python scripts/filter_python_dataset.py

data/solutions: data/unlabeled
	python scripts/generate_python3_solutions.py


prepare-dataset:
	python scripts/download_dataset.py \
	&& python scripts/filter_python_dataset.py

clean-saved:
	sh scripts/clean_saved.sh
