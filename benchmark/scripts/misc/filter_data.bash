shopt -s globstar

for file in raw-data/**/*.jsonl; do
  target="${file/raw-data/filtered-raw-data}"
  mkdir -p "$(dirname "$target")"
  python filter_data.py \
    --input "$file" \
    --output "$target" \
    --filter new_test_testcase_extracted.jsonl
done
