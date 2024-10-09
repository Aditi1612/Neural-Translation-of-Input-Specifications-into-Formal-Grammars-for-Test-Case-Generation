shopt -s globstar

for file in data/**/*.jsonl; do
  target="${file/data/filtered-data}"
  filter="${GROUND_TRUTH_GRAMMAR_DIR}/$(basename $file)"
  mkdir -p "$(dirname "$target")"
  python scripts/misc/filter_data.py \
    --input "$file" \
    --output "$target" \
    --filter "$filter"
done
