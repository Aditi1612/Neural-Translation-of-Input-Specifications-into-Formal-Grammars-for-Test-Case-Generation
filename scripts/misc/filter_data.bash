shopt -s globstar

for file in data/**/test.jsonl; do
  target="${file/data/filtered-data}"
  echo "Filtering $file to $target"
  filter="./raw-data/grammar/ground-truth/test.jsonl"
  mkdir -p "$(dirname "$target")"
  python scripts/misc/filter_data.py \
    --input "$file" \
    --output "$target" \
    --filter "$filter"
done
