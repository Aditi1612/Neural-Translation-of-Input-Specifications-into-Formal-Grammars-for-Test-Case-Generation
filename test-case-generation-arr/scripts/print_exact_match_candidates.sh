shopt -s globstar

for file in data/grammar-candidate/**/*.jsonl; do
  echo "$file"
  python scripts/compute_exact_match_candidate.py \
    "$file" data/grammar/ground-truth/test.jsonl \
    --filter1 "data/generation-result/ground-truth/test-extreme.jsonl" \
    --filter2 "data/generation-result/code-contest/private/test.jsonl"
  echo
done
