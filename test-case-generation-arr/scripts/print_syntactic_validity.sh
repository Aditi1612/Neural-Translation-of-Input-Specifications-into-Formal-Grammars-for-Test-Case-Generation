shopt -s globstar

for file in data/generation-result/**/*.jsonl; do
  echo "$file"
  python scripts/compute_syntactic_validity.py \
    "$file" \
    --filter1 "data/generation-result/ground-truth/test-extreme.jsonl" \
    --filter2 "data/generation-result/code-contest/private/test.jsonl"
  echo
done
