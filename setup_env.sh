PYTHONPATH="$(pwd)/src:$(pwd)/scripts"
export PYTHONPATH

RAW_DATA_DIR="./raw-data"
export RAW_DATA_DIR

INTERMEDIATE_DIR="${RAW_DATA_DIR}/intermediate"
export INTERMEDIATE_DIR

DATA_DIR="./data"
export DATA_DIR

LEVEL_DIR="${RAW_DATA_DIR}/level"
export LEVEL_DIR

# Path to the correct/incorrect solutions of the code contest
SOLUTIONS_DIR="${DATA_DIR}/solutions"
CORRECT_SOLUTIONS_DIR="${SOLUTIONS_DIR}/solutions"
export CORRECT_SOLUTIONS_DIR
INCORRECT_SOLUTIONS_DIR="${SOLUTIONS_DIR}/incorrect_solutions"
export INCORRECT_SOLUTIONS_DIR

# Data related to the code contest testcases
PUBLIC_TESTCASE_DIR="${DATA_DIR}/testcase/code-contest/public"
export PUBLIC_TESTCASE_DIR
PRIVATE_TESTCASE_DIR="${DATA_DIR}/testcase/code-contest/private"
export PRIVATE_TESTCASE_DIR

# Data related to the ground truth grammars
GROUND_TRUTH_GRAMMAR_DIR="${RAW_DATA_DIR}/grammar/ground-truth"
export GROUND_TRUTH_GRAMMAR_DIR
GROUND_TRUTH_TESTCASE_DIR="${DATA_DIR}/testcase/grammar/ground-truth"
export GROUND_TRUTH_TESTCASE_DIR
GROUND_TRUTH_GENERATION_RESULT="${DATA_DIR}/generation-result/grammar/ground-truth/test.jsonl"
export GROUND_TRUTH_GENERATION_RESULT
GROUND_TRUTH_PARSING_RESULT="${DATA_DIR}/parsing-result/ground-truth/test.jsonl"
export GROUND_TRUTH_PARSING_RESULT
GROUND_TRUTH_EXECUTION_SUMMARY="${DATA_DIR}/execution-summary/grammar/ground-truth/test.jsonl"
export GROUND_TRUTH_EXECUTION_SUMMARY

# Data related to the dataset testcases
PUBLIC_GENERATION_RESULT="${DATA_DIR}/generation-result/code-contest/public/test.jsonl"
export PUBLIC_GENERATION_RESULT
PRIVATE_GENERATION_RESULT="${DATA_DIR}/generation-result/code-contest/private/test.jsonl"
export PRIVATE_GENERATION_RESULT
