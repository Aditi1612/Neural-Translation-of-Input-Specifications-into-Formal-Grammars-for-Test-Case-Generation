shopt -s globstar

for file in data/execution/**/*.jsonl; do
  echo "$file"
  generation=${file/execution/generation-result}
  testcase=${file/execution/testcase}
  python scripts/compute_effectiveness.py "$file" "$generation" "$testcase" \
    --filter1 "data/generation-result/ground-truth/test-extreme.jsonl" \
    --filter2 "data/generation-result/code-contest/private/test.jsonl"
  echo
done
