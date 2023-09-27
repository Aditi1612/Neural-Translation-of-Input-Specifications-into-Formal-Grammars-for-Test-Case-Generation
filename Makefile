.PHONY: \

	all \
	help \
	prepare-dataset \
	test-ccfg clean-saved \
	test-human-labeled-data \
	test-human-labeled-data-test \
	test-human-labeled-data-train \
	validate-bard-grammar \
	validate-bard-grammar-1-shot \
	validate-bard-grammar-5-shot

model_with_pseudo_labeling_path = saved/0927.wPL/best-checkpoint-epoch55.pth
model_without_pseudo_labeling_path = saved/0925.without_pseudo_labeling/best-checkpoint-epoch90.pth
model_with_base_pl_path = saved/0927.train_base_pl/checkpoint-epoch60.pth

all:

################################################################################

prepare-dataset:  ## Prepare the dataset
	python scripts/download_dataset.py \
	&& python scripts/filter_python_dataset.py \
	&& python scripts/generate_python3_solutions.py \
	&& python scripts/filter_labeled_test_dataset.py

data/raw:
	python scripts/download_dataset.py

data/unlabeled: data/raw
	python scripts/filter_python_dataset.py

data/solutions: data/unlabeled
	python scripts/generate_python3_solutions.py

results:
	mkdir -p results

################################################################################
# Grammar generation with CodeT5-based model
# FIXME: Hard-coded
################################################################################

label-with-model-without-pseudo-labeling-labeled-test: | results  ## Label the data with the model
	for i in 1 10 100; do \
		python label_with_model.py \
			--model-pth $(model_without_pseudo_labeling_path) \
			--output results/ccfg_without_pl_labeled_test_beam_$${i}.jsonl \
			--unlabeled-data data/unlabeled/code_contests_train_python_with_test_label.jsonl\
			--config configs/labeling_config_beam_$${i}.json; \
	done

label-with-model-with-pseudo-labeling-labeled-test: | results  ## Label the data with the model
	for i in 1 10 100; do \
		python label_with_model.py \
			--model-pth $(model_with_pseudo_labeling_path) \
			--output results/ccfg_with_pl_labeled_test_beam_$${i}.jsonl \
			--unlabeled-data data/unlabeled/code_contests_train_python_with_test_label.jsonl\
			--config configs/labeling_config_beam_$${i}.json; \
	done

label-test-with-model-with-base-pl: | results  ## Label the data with the base pl model
	for i in 1 10 100; do \
		python label_with_model.py \
			--model-pth $(model_with_base_pl_path) \
			--output results/ccfg_with_base_pl_labeled_test_beam_$${i}.jsonl \
			--unlabeled-data data/unlabeled/code_contests_train_python_with_test_label.jsonl\
			--config configs/labeling_config_beam_$${i}.json; \
	done

################################################################################
# Syntax evaluation
################################################################################

validate-syntactic-equivalence-model-without-pl:  ## Validate the syntactic equivalence without pseudo labeling
	python test.py --model-pth $(model_without_pseudo_labeling_path)

validate-syntactic-equivalence-model-with-pl:  ## Validate the syntactic equivalence with pseudo labeling
	python test.py --model-pth $(model_with_pseudo_labeling_path)

validate-syntactic-equivalence-model-with-base-pl:  ## Validate the syntactic equivalence with pseudo labeling
	python test.py --model-pth $(model_with_base_pl_path)

validate-syntactic-equivalence-bard-1-shot:  ## Validate the syntactic equivalence of bard grammar with 1-shot
	python test_large_language_model.py \
		--model-labeled-data=data/bard_labeled/test_1_shot.jsonl

validate-syntactic-equivalence-bard-5-shot:  ## Validate the syntactic equivalence of bard grammar with 5-shot
	python test_large_language_model.py \
		--model-labeled-data=data/bard_labeled/test_5_shot.jsonl

validate-syntactic-equivalence-gpt-1-shot:  ## Validate the syntactic equivalence of gpt grammar with 1-shot
	python test_large_language_model.py \
		--model-labeled-data=data/gpt_labeled/test_1_shot.jsonl

validate-syntactic-equivalence-gpt-5-shot:  ## Validate the syntactic equivalence of gpt grammar with 5-shot
	python test_large_language_model.py \
		--model-labeled-data=data/gpt_labeled/test_5_shot.jsonl

################################################################################
# Semantic evaluation
################################################################################

validate-model-labeling-without-pl-labeled-test:  ## Validate the model labeling without pseudo labeling
	for i in 1 10 100; do \
		python validate_labeling.py \
			--labeled-data results/ccfg_without_pl_labeled_test_beam_$${i}.jsonl \
			--testcase data/unlabeled/code_contests_train_python_with_test_label.jsonl; \
	done

validate-model-labeling-with-pl-labeled-test:  ## Validate the model labeling without pseudo labeling
	for i in 1 10 100; do \
		python validate_labeling.py \
			--labeled-data results/ccfg_with_pl_labeled_test_beam_$${i}.jsonl \
			--testcase data/unlabeled/code_contests_train_python_with_test_label.jsonl; \
	done

validate-bard-grammar: validate-bard-grammar-1-shot validate-bard-grammar-5-shot  ## Test the bard grammar

validate-bard-grammar-1-shot:  ## Test the bard grammar with 1-shot
	python validate_labeling.py \
		--labeled-data data/bard_labeled/test_1_shot.jsonl \
		--testcase data/unlabeled/code_contests_train_python_with_test_label.jsonl

validate-bard-grammar-5-shot:  ## Test the bard grammar with 5-shot
	python validate_labeling.py \
		--labeled-data data/bard_labeled/test_5_shot.jsonl \
		--testcase data/unlabeled/code_contests_train_python_with_test_label.jsonl

validate-gpt-grammar-1-shot:  ## Test the gpt grammar with 1-shot
	python validate_labeling.py \
		--labeled-data data/gpt_labeled/test_1_shot.jsonl \
		--testcase data/unlabeled/code_contests_train_python_with_test_label.jsonl

validate-gpt-grammar-5-shot:  ## Test the gpt grammar with 5-shot
	python validate_labeling.py \
		--labeled-data data/gpt_labeled/test_5_shot.jsonl \
		--testcase data/unlabeled/code_contests_train_python_with_test_label.jsonl

################################################################################
# Generate testcase
################################################################################

generate-testcase-model-without-pseudo-lebeling:  ## Generate the testcase with the model without pseudo labeling
	for i in 1 10 100; do \
		python generate_testcase_with_grammar.py \
			--labeled-data results/ccfg_without_pl_labeled_test_beam_$${i}.jsonl \
			--output results/ccfg_without_pl_labeled_test_beam_$${i}_testcase.jsonl; \
	done

generate-testcase-bard-grammar-1-shot:  ## Generate the testcase with bard 1-shot grammar
	python generate_testcase_with_grammar.py \
		--labeled-data data/bard_labeled/test_1_shot.jsonl \
		--output results/bard_test_1_shot_testcase.jsonl

generate-testcase-bard-grammar-5-shot:  ## Generate the testcase with bard 5-shot grammar
	python generate_testcase_with_grammar.py \
		--labeled-data data/bard_labeled/test_5_shot.jsonl \
		--output results/bard_test_5_shot_testcase.jsonl

generate-testcase-gpt-grammar-1-shot:  ## Generate the testcase with gpt 1-shot grammar
	python generate_testcase_with_grammar.py \
		--labeled-data data/gpt_labeled/test_1_shot.jsonl \
		--output results/gpt_test_1_shot_testcase.jsonl

generate-testcase-gpt-grammar-5-shot:  ## Generate the testcase with gpt 5-shot grammar
	python generate_testcase_with_grammar.py \
		--labeled-data data/gpt_labeled/test_5_shot.jsonl \
		--output results/gpt_test_5_shot_testcase.jsonl

################################################################################
# Effectiveness Evaluation
################################################################################

validate-testcase-model-without-pseudo-labeling:  ## Validate the model generated testcase
	for i in 1 10 100; do \
		python validate_testcase.py \
			--testcase results/ccfg_without_pl_labeled_test_beam_$${i}_testcase.jsonl \
			--type model_generated; \
	done

validate-testcase-fuzzing:  ## Validate the fuzzing testcase
	python validate_testcase.py \
		--testcase data/testcase/fuzzing.jsonl \
		--type fuzzing

validate_testcase_codecontest_targets = \
	validate-testcase-codecontest-public \
	validate-testcase-codecontest-private \
	validate-testcase-codecontest-generated

validate-testcase-codecontest: $(validate_testcase_codecontest_targets) ## Validate the codecontest testcase

validate-testcase-codecontest-generated:  ## Validate the codecontest generated testcase
	python validate_testcase.py \
		--testcase data/unlabeled/code_contests_train_python_with_test_label.jsonl \
		--type codecontests_generated

validate-testcase-codecontest-public:  ## Validate the codecontest public testcase
	python validate_testcase.py \
		--testcase data/unlabeled/code_contests_train_python_with_test_label.jsonl \
		--type codecontests_public

validate-testcase-codecontest-private:  ## Validate the codecontest private testcase
	python validate_testcase.py \
		--testcase data/unlabeled/code_contests_train_python_with_test_label.jsonl \
		--type codecontests_private

validate-testcase-bard-grammar-1-shot:  ## Validate the bard-grammar 1-shot testcase
	python validate_testcase.py \
		--testcase results/bard_test_1_shot_testcase.jsonl \
		--type model_generated

validate-testcase-bard-grammar-5-shot:  ## Validate the bard-grammar 5-shot testcase
	python validate_testcase.py \
		--testcase results/bard_test_5_shot_testcase.jsonl \
		--type model_generated

validate-testcase-gpt-grammar-1-shot:  ## Validate the gpt-grammar 1-shot testcase
	python validate_testcase.py \
		--testcase results/gpt_test_1_shot_testcase.jsonl \
		--type model_generated

validate-testcase-gpt-grammar-5-shot:  ## Validate the gpt-grammar 5-shot testcase
	python validate_testcase.py \
		--testcase results/gpt_test_5_shot_testcase.jsonl \
		--type model_generated


################################################################################
# XXX: Utility
################################################################################

clean-saved:  ## Clean the saved files except the last checkpoint
	sh scripts/clean_saved.sh

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | sort

test-human-labeled-data: test-human-labeled-data-train test-human-labeled-data-test  ## Test the human-labeled data

test-human-labeled-data-train:  ## Test the human-labeled train data
	python validate_labeling.py \
		--labeled-data data/labeled/train.jsonl \
		--testcase data/unlabeled/code_contests_train_python.jsonl

test-human-labeled-data-test:  ## Test the human-labeled test data
	python validate_labeling.py \
		--labeled-data data/labeled/test.jsonl \
		--testcase data/unlabeled/code_contests_train_python.jsonl

################################################################################
# XXX: Unused
################################################################################

label-unlabeled-test-with-model: | results
	for i in 1 10 100; do \
		python label_with_model.py \
			--model-pth saved/0925.without_pseudo_labeling/best-checkpoint-epoch90.pth \
			--output results/ccfg_without_pl_unlabeled_test_beam_$${i}.jsonl \
			--unlabeled-data data/unlabeled/code_contests_test_python.jsonl\
			--config configs/labeling_config_beam_$${i}.json; \
	done

label-with-model-with-pseudo-labeling-valid: | results
	for i in 1 10 100; do \
		python label_with_model.py \
			--model-pth $(model_with_pseudo_labeling_path) \
			--output results/ccfg_without_pl_valid_beam_$${i}.jsonl \
			--config configs/labeling_config_beam_$${i}.json; \
	done

label-with-model-without-pseudo-labeling-valid: | results
	for i in 1 10 100; do \
		python label_with_model.py \
			--model-pth $(model_without_pseudo_labeling_path) \
			--output results/ccfg_without_pl_valid_beam_$${i}.jsonl \
			--config configs/labeling_config_beam_$${i}.json; \
	done
