shopt -s globstar

for file in data/parsing-result/**/*.jsonl; do
  echo "$file"
  generation=${file/parsing-result/generation-result}
  python scripts/compute_generality.py  "$file" \
    "$generation" \
    --filter1 "data/generation-result/ground-truth/test-extreme.jsonl" \
    --filter2 "data/generation-result/code-contest/private/test.jsonl"
  echo
done
