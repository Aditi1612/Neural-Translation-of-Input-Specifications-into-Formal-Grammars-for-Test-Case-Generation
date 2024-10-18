shopt -s globstar

for file in ${DATA_DIR}/parsing-result/**/*.jsonl; do
  echo "$file"
  python scripts/compute/generality.py \
    --generation-result "${file/parsing-result/generation-result\/grammar}" \
    --parsing-result "$file"
  echo
done
