shopt -s globstar

for file in ${DATA_DIR}/generation-result/**/*.jsonl; do
  echo "$file"
  python scripts/compute_syntactic_validity.py \
    "$file" \
    --filter1 "${DATA_DIR}/generation-result/ground-truth/test-extreme.jsonl" \
    --filter2 "${DATA_DIR}/generation-result/code-contest/private/test.jsonl"
  echo
done
