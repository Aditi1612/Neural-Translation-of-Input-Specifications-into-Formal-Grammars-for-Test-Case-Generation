VENV=.venv
PYTHON=$(VENV)/bin/python

model=saved/checkpoint.pth

.SECONDARY:

data/grammar-candidate/ccfg-t5/beam-%/test.jsonl: data/grammar/ground-truth/test.jsonl
	mkdir -p $(dir $@)
	$(PYTHON) scripts/generate_grammar_candidates.py \
		--model-pth saved/checkpoint.pth \
		--data $< \
		--output $@ \
		--num-beams $*

data/grammar/%.jsonl: data/grammar-candidate/%.jsonl
	mkdir -p $(dir $@)
	$(PYTHON) scripts/generate_grammar_from_candidates.py \
		--data $< \
		--output $@

data/testcase/%-extreme.jsonl: data/grammar/%.jsonl
	mkdir -p $(dir $@)
	$(PYTHON) scripts/generate_testcase_from_grammar.py \
		--grammar-data $< \
		--output $@ \
		--extreme

data/testcase/%.jsonl: data/grammar/%.jsonl
	mkdir -p $(dir $@)
	$(PYTHON) scripts/generate_testcase_from_grammar.py \
		--grammar-data $< \
		--output $@

data/testcase/fuzzing/test.jsonl: data/testcase/code-contest/private/test.jsonl
	mkdir -p $(dir $@)
	$(PYTHON) scripts/testcase_fuzzing.py \
		--input $< \
		--output $@

data/testcase/fuzzing/test-public.jsonl: data/testcase/code-contest/public/test.jsonl
	mkdir -p $(dir $@)
	$(PYTHON) scripts/testcase_fuzzing.py \
		--input $< \
		--output $@

data/execution/%.jsonl: data/testcase/%.jsonl
	mkdir -p $(dir $@)
	$(PYTHON) scripts/generate_execution.py $< $@

data/generation-result/%.jsonl: data/testcase/%.jsonl \
		data/grammar/ground-truth/test.jsonl
	mkdir -p $(dir $@)
	$(PYTHON) scripts/generate_generation_result.py \
		$< data/grammar/ground-truth/test.jsonl $@

data/parsing-result/%-extreme.jsonl: data/grammar/%.jsonl \
		data/testcase/ground-truth/test-extreme.jsonl
	mkdir -p $(dir $@)
	$(PYTHON) scripts/generate_parsing_result.py \
		$< data/testcase/ground-truth/test-extreme.jsonl $@

data/parsing-result/%.jsonl: data/grammar/%.jsonl \
		data/testcase/ground-truth/test.jsonl
	mkdir -p $(dir $@)
	$(PYTHON) scripts/generate_parsing_result.py \
		$< data/testcase/ground-truth/test.jsonl $@

suffixes = ccfg-t5/beam-1/test.jsonl \
	ccfg-t5/beam-10/test.jsonl \
	ccfg-t5/beam-100/test.jsonl \
	code-contest/private/test.jsonl \
	code-contest/public/test.jsonl \
	code-contest/generated/test.jsonl \
	fuzzing/test.jsonl \
	fuzzing/test-public.jsonl \
	gemini/1-shot/test.jsonl \
	gemini/5-shot/test.jsonl \
	gemini/direct/test.jsonl \
	gpt/1-shot/test.jsonl \
	gpt/5-shot/test.jsonl \
	gpt/direct/test.jsonl \
	ground-truth/test.jsonl

generation_results = $(foreach suffix,$(suffixes),data/generation-result/$(suffix))
parsing_results = $(foreach suffix,$(suffixes),data/parsing-result/$(suffix))
executions = $(foreach suffix,$(suffixes),data/execution/$(suffix))

suffixes_extreme = ccfg-t5/beam-1/test-extreme.jsonl \
	ccfg-t5/beam-10/test-extreme.jsonl \
	ccfg-t5/beam-100/test-extreme.jsonl \
	gemini/1-shot/test-extreme.jsonl \
	gemini/5-shot/test-extreme.jsonl \
	gpt/1-shot/test-extreme.jsonl \
	gpt/5-shot/test-extreme.jsonl \
	ground-truth/test-extreme.jsonl

generation_extreme_results = $(foreach suffix,$(suffixes_extreme),data/generation-result/$(suffix))
parsing_extreme_results = $(foreach suffix,$(suffixes_extreme),data/parsing-result/$(suffix))
extreme_executions = $(foreach suffix,$(suffixes_extreme),data/execution/$(suffix))

executions-long = data/execution/ccfg-t5/beam-1/test-timeout-10.jsonl \
	data/execution/ccfg-t5/beam-10/test-timeout-10.jsonl \
	data/execution/ccfg-t5/beam-100/test-timeout-10.jsonl \
	data/execution/code-contest/private/test-timeout-10.jsonl \
	data/execution/code-contest/public/test-timeout-10.jsonl \
	data/execution/code-contest/generated/test-timeout-10.jsonl \
	data/execution/fuzzing/test-timeout-10.jsonl \
	data/execution/fuzzing/test-public-timeout-10.jsonl \
	data/execution/gemini/1-shot/test-timeout-10.jsonl \
	data/execution/gemini/5-shot/test-timeout-10.jsonl \
	data/execution/gemini/direct/test-timeout-10.jsonl \
	data/execution/gpt/1-shot/test-timeout-10.jsonl \
	data/execution/gpt/5-shot/test-timeout-10.jsonl \
	data/execution/gpt/direct/test-timeout-10.jsonl \
	data/execution/ground-truth/test-timeout-10.jsonl

data/execution/%/test-timeout-10.jsonl: data/testcase/%/test.jsonl
	mkdir -p $(dir $@)
	$(PYTHON) scripts/generate_execution.py $< $@

data/testcase/%/test-exetreme.jsonl: data/grammar/%/test.jsonl
	mkdir -p $(dir $@)
	$(PYTHON) scripts/generate_testcase_from_grammar.py \
		--grammar-data $< \
		--output $@ \
		--timeout 10

all-generation-results:
	$(MAKE) -k $(generation_results)

all-executions:
	$(MAKE) -k $(executions)

all-executions-long:
	$(MAKE) -k $(executions-long)


all-extreme-executions:
	$(MAKE) -k $(extreme_executions)

all-extreme-parsing-results:
	$(MAKE) -k $(parsing_extreme_results)

all-extreme-generation-results:
	$(MAKE) -k $(generation_extreme_results)

all-extreme: all-extreme-executions all-extreme-generation-results all-extreme-parsing-results
