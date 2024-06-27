shopt -s globstar

for file in data/execution/**/*.jsonl; do
# for file in data/execution/**/test-timeout-10.jsonl; do
  echo "$file"
  testcase="${file/execution/testcase}"
  python scripts/compute_execution_validity.py "$file" "$testcase" \
    --filter1 "data/generation-result/ground-truth/test-extreme.jsonl" \
    --filter2 "data/generation-result/code-contest/private/test.jsonl"
  echo
done
