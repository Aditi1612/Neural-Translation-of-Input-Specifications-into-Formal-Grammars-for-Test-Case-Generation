$(DATA)/coverage/%.jsonl: $(DATA)/testcase/%.jsonl
	mkdir -p $(dir $@)
	$(PYTHON) scripts/generate/coverage.py $< $@

coverage: $(COVERAGE_RESULT)
