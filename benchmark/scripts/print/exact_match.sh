shopt -s globstar

for file in ${DATA_DIR}/grammar/**/*.jsonl; do
  echo "$file"
  python scripts/compute_exact_match.py \
    "$file" ${DATA_DIR}/grammar/ground-truth/test.jsonl \
    --filter1 "${DATA_DIR}/generation-result/ground-truth/test-extreme.jsonl" \
    --filter2 "${DATA_DIR}/generation-result/code-contest/private/test.jsonl"
  echo
done
