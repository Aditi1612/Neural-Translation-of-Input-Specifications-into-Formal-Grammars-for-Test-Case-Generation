$(DATA)/generation-result/%.jsonl: \
		$(DATA)/testcase/%.jsonl \
		| ${GROUND_TRUTH_GRAMMAR_DIR}
	mkdir -p $(dir $@)
	$(PYTHON) scripts/generate/generation_result.py \
		--grammar "$|/$(notdir $@)" \
		--testcase $< \
		--output $@

generation-result: $(GENERATION_RESULT)

clean-generation-result:
	rm -rf $(DATA)/generation-result
