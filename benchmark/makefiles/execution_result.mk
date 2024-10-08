$(DATA)/execution/%.jsonl: $(DATA)/testcase/%.jsonl
	mkdir -p $(dir $@)
	$(PYTHON) scripts/generate/execution.py \
		--testcase $< \
		--output $@

execution: $(EXECUTION_RESULT)
