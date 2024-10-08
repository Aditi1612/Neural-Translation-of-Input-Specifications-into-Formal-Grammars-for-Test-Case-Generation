shopt -s globstar

for file in ${DATA_DIR}/generation-result/**/*.jsonl; do
  echo "$file"
  python scripts/compute/validity.py \
    --generation-result "$file"
  echo
done
