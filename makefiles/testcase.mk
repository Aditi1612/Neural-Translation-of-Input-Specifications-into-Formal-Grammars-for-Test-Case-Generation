$(DATA)/testcase/grammar/%.jsonl: $(DATA)/grammar/%.jsonl
	mkdir -p $(dir $@)
	$(PYTHON) scripts/generate/testcase_from_grammar.py \
		--grammar-data $< \
		--output $@

$(DATA)/testcase/fuzzing/%.jsonl: $(DATA)/testcase/code-contest/%.jsonl
	mkdir -p $(dir $@)
	$(PYTHON) scripts/generate/testcase_fuzzing.py \
		--input $< \
		--output $@

testcase: $(TESTCASE)

clean-testcase-grammar:
	rm -rf $(DATA)/testcase/grammar

clean-testcase-fuzzing:
	rm -rf $(DATA)/testcase/fuzzing
