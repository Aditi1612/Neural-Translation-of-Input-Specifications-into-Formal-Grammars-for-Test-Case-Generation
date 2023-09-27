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
generate-testcase-model-without-pseudo-lebeling:  ## Generate the testcase with the model without pseudo labeling
help:  ## Show this help message
label-test-with-model-with-base-pl: | results  ## Label the data with the base pl model
label-with-model-without-pseudo-labeling-labeled-test: | results  ## Label the data with the model
label-with-model-with-pseudo-labeling-labeled-test: | results  ## Label the data with the model
prepare-dataset:  ## Prepare the dataset
test-human-labeled-data: test-human-labeled-data-train test-human-labeled-data-test  ## Test the human-labeled data
test-human-labeled-data-test:  ## Test the human-labeled test data
test-human-labeled-data-train:  ## Test the human-labeled train data
validate-bard-grammar: validate-bard-grammar-1-shot validate-bard-grammar-5-shot  ## Test the bard grammar
validate-model-labeling-without-pl-labeled-test:  ## Validate the model labeling without pseudo labeling
validate-model-labeling-with-pl-labeled-test:  ## Validate the model labeling without pseudo labeling
validate-syntactic-equivalence-model-with-base-pl:  ## Validate the syntactic equivalence with pseudo labeling
validate-syntactic-equivalence-model-without-pl:  ## Validate the syntactic equivalence without pseudo labeling
validate-syntactic-equivalence-model-with-pl:  ## Validate the syntactic equivalence with pseudo labeling
validate-testcase-codecontest: $(validate_testcase_codecontest_targets) ## Validate the codecontest testcase
validate-testcase-codecontest-generated:  ## Validate the codecontest generated testcase
validate-testcase-codecontest-private:  ## Validate the codecontest private testcase
validate-testcase-codecontest-public:  ## Validate the codecontest public testcase
validate-testcase-fuzzing:  ## Validate the fuzzing testcase
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
