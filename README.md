# Neural Translation of Input Specifications into Formal Grammars for Test Case Generation

by Anonymous

This repo provides the source code & data of our work.

## Abstract

> Test cases are crucial for ensuring the program's correctness and evaluating
> performance in programming. The high diversity of test cases within
> constraints is necessary to distinguish between correct and incorrect
> answers. Automated source code generation is currently a popular area due to
> the inefficiency of manually generating test cases. Recent attempts involve
> generating conditional cases from problem descriptions using deep-learning
> models that learn from source code. However, this task requires a combination
> of complex skills such as extracting syntactic and logical constraints for a
> given test case from a problem, and generating test cases that satisfy the
> constraints. In this work, we introduce a modified context-free grammar that
> explicitly represents the syntactical and logical constraints embedded within
> programming problems. Our innovative framework for automated test case
> generation separates restriction extraction from test case generation,
> simplifying the task for the model. We compare diverse methods for neural
> translation of input specifications into formal grammars.

## Usage

```
$ make help
clean-saved:  ## Clean the saved files except the last checkpoint
generate-testcase-model-without-pseudo-lebeling-extreme:  ## Generate the testcase with the model without pseudo labeling in extreme distribution
generate-testcase-model-without-pseudo-lebeling:  ## Generate the testcase with the model without pseudo labeling
help:  ## Show this help message
label-test-with-model-with-base-pl: | results  ## Label the data with the base pl model
label-test-with-model-with-complete-pl: | results  ## Label the data with the complete pl model
label-test-with-model-with-correct-pl: | results  ## Label the data with the correct pl model
label-with-model-without-pseudo-labeling-labeled-test: | results  ## Label the data with the model
label-with-model-with-pseudo-labeling-labeled-test: | results  ## Label the data with the model
prepare-dataset:  ## Prepare the dataset
test-human-labeled-data: test-human-labeled-data-train test-human-labeled-data-test  ## Test the human-labeled data
test-human-labeled-data-test:  ## Test the human-labeled test data
test-human-labeled-data-train:  ## Test the human-labeled train data
validate-large-language-model-grammar: $(validate_large_lagnague_model_grammar_targets) ## Validate the large language model grammar
validate-model-labeling-without-pl-labeled-test:  ## Validate the model labeling without pseudo labeling
validate-model-labeling-with-pl-labeled-test:  ## Validate the model labeling with pseudo labeling
validate-model-with-base-pl-test:  ## Validate the model labeling with base-pl
validate-model-with-complete-pl-test:  ## Validate the model labeling with complete-pl
validate-model-with-correct-pl-test:  ## Validate the model labeling with correct-pl
validate-model-with-generatable-pl-test:  ## Validate the model labeling with generatable-pl
validate-syntactic-equivalence-model-with-base-pl:  ## Validate the syntactic equivalence with base pseudo labeling
validate-syntactic-equivalence-model-with-complete-pl-earlier:  ## Validate the syntactic equivalence with complete pseudo labeling earlier
validate-syntactic-equivalence-model-with-complete-pl:  ## Validate the syntactic equivalence with complete pseudo labeling
validate-syntactic-equivalence-model-with-correct-pl:  ## Validate the syntactic equivalence with correct pseudo labeling
validate-syntactic-equivalence-model-with-generatable-pl:  ## Validate the syntactic equivalence with generatable pseudo labeling
validate-syntactic-equivalence-model-without-pl:  ## Validate the syntactic equivalence without pseudo labeling
validate-syntactic-equivalence-model-with-pl:  ## Validate the syntactic equivalence with pseudo labeling
validate-testcase: $(validate_testcase_targets) ## Validate the testcase
validate-testcase-bard-zero-shot:  ## Validate the bard generated testcase zero-shot
validate-testcase-codecontests: $(validate_testcase_codecontests_targets) ## Validate the codecontest testcase
validate-testcase-codecontests-generated:  ## Validate the codecontests generated testcase
validate-testcase-codecontests-private:  ## Validate the codecontest private testcase
validate-testcase-codecontests-public:  ## Validate the codecontest public testcase
validate-testcase-fine-tuning:  ## Validate the fine-tuning testcase
validate-testcase-fuzzing:  ## Validate the fuzzing testcase
validate-testcase-gpt-zero-shot:  ## Validate the gpt generated testcase zero-shot
validate-testcase-large-language-model: $(validate_testcase_large_language_model_targets) ## Validate the large language model testcase
validate-testcase-large-language-model-grammar: $(validate_testcase_large_language_model_grammar_targets) ## Validate the large language model grammar
validate-testcase-model-without-pseudo-labeling-extreme:  ## Validate the model generated testcase in extreme distribution
validate-testcase-model-without-pseudo-labeling:  ## Validate the model generated testcase
```

## Reproducing the results

```
$ python train.py -h
usage: train.py [-h] [--loss-path LOSS_PATH] [--config CONFIG]

optional arguments:
  -h, --help            show this help message and exit
  --loss-path LOSS_PATH
  --config CONFIG
```

## Citation

Not available.

## License

All source code is made available under a BSD 3-clause license. You can freely
use and modify the code, without warranty, so long as you provide attribution
to the authors. See `LICENSE.md` for the full license text.
