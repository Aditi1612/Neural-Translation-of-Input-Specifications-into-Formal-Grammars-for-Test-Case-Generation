${GROUND_TRUTH_TESTCASE_DIR}:
	mkdir -p $@

$(DATA)/parsing-result/%.jsonl: \
		$(DATA)/grammar/%.jsonl \
		| ${GROUND_TRUTH_TESTCASE_DIR}
	mkdir -p $(dir $@)
	$(PYTHON) scripts/generate/generation_result.py \
		--grammar $< \
		--testcase "$|/$(not $@)" \
		--output $@

parsing-result: $(PARSING_RESULT)

clean-parsing-result:
	rm -rf $(DATA)/parsing-result
