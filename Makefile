DATA:=${DATA_DIR}
VENV:=.venv
PYTHON:=$(VENV)/bin/python
MODEL:=saved/checkpoint.pth

# Prevent build non-existing prerequisites
.SECONDARY:

data/%.jsonl: raw-data/%.jsonl
	mkdir -p $(dir $@)
	ln -fs $(abspath $<) $@

prepare-dataset:  ## Create ${SOLUTIONS_DIR}
	python scripts/download_dataset.py \
	&& python scripts/filter_python_dataset.py \
	&& python scripts/generate_python3_solutions.py \
	&& python scripts/filter_labeled_test_dataset.py

default: all

SUFFIXES_GRAMMAR = \
	ccfg-t5/beam-1/test.jsonl \
	ccfg-t5/beam-10/test.jsonl \
	ccfg-t5/beam-100/test.jsonl \
	gemini/1-shot/test.jsonl \
	gemini/5-shot/test.jsonl \
	gpt/1-shot/test.jsonl \
	gpt/5-shot/test.jsonl \
	ground-truth/test.jsonl \
	ground-truth/train.jsonl
GRAMMAR = $(SUFFIXES_GRAMMAR:%=$(DATA)/grammar/%)
include makefiles/grammar.mk

SUFFIXES_TESTCASE_GRAMMAR = $(SUFFIXES_GRAMMAR:%=grammar/%)
SUFFIXES_TESTCASE = \
	$(SUFFIXES_TESTCASE_GRAMMAR) \
	code-contest/private/test.jsonl \
	code-contest/public/test.jsonl \
	code-contest/generated/test.jsonl \
	direct/gemini/test.jsonl \
	direct/gpt/test.jsonl \
	fuzzing/private/test.jsonl \
	fuzzing/public/test.jsonl \
	fuzzing/private/train.jsonl \
	fuzzing/public/train.jsonl
TESTCASE = $(SUFFIXES_TESTCASE:%=$(DATA)/testcase/%)
include makefiles/testcase.mk

GENERATION_RESULT = $(SUFFIXES_TESTCASE:%=$(DATA)/generation-result/%)
include makefiles/generation_result.mk

PARSING_RESULT = $(SUFFIXES_GRAMMAR:%=$(DATA)/parsing-result/%)
include makefiles/parsing_result.mk

EXECUTION_RESULT = $(SUFFIXES_TESTCASE:%=$(DATA)/execution/%)
include makefiles/execution_result.mk

SUMMARY_RESULT = $(SUFFIXES_TESTCASE:%=$(DATA)/execution-summary/%)
include makefiles/summary_result.mk

COVERAGE_RESULT = $(SUFFIXES_TESTCASE:%=$(DATA)/coverage/%)
include makefiles/coverage_result.mk

all: \
	$(SUMMARY_RESULT) \
	$(GENERATION_RESULT) \
	$(PARSING_RESULT) \
	$(COVERAGE_RESULT)
