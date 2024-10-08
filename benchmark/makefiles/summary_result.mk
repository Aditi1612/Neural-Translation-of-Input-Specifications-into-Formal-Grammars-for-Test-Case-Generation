$(DATA)/execution-summary/%.jsonl: $(DATA)/execution/%.jsonl
	mkdir -p $(dir $@)
	$(PYTHON) scripts/generate/execution_summary.py \
		--execution $< \
		--output $@

execution-summary: $(SUMMARY_RESULT)
