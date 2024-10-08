shopt -s globstar

for file in ${DATA_DIR}/parsing-result/**/*.jsonl; do
  echo "$file"
  generation=${file/parsing-result/generation-result}
  python scripts/compute_generality.py  "$file" \
    "$generation" \
    --filter1 "${DATA_DIR}/generation-result/ground-truth/test-extreme.jsonl" \
    --filter2 "${DATA_DIR}/generation-result/code-contest/private/test.jsonl"
  echo
done
