all:

prepare-dataset:
	python scripts/download_dataset.py \
	&& python scripts/filter_python_dataset.py

test-ccfg:
	python -m counting_context_free_grammar.test_counting_context_fee_grammar
