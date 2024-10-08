MODEL:=saved/checkpoint.pth

$(DATA)/grammar-candidate/ccfg-t5/beam-%/test.jsonl: \
		$(DATA)/grammar/ground-truth/test.jsonl
	mkdir -p $(dir $@)
	$(PYTHON) scripts/generate/grammar_candidates.py \
		--model-pth $(MODEL) \
		--data $< \
		--output $@ \
		--num-beams $*

$(DATA)/grammar/%.jsonl: $(DATA)/grammar-candidate/%.jsonl
	mkdir -p $(dir $@)
	$(PYTHON) scripts/generate/grammar_from_candidates.py \
		--data $< \
		--output $@

grammar: $(GRAMMAR)
