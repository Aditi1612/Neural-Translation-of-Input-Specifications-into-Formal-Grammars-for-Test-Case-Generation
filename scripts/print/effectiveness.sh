shopt -s globstar

for file in ${DATA_DIR}/execution-summary/**/*.jsonl; do
  echo "$file"
  generation=${file/execution-summary/generation-result}
  python scripts/compute/effectiveness_element.py \
    --execution-summary "$file" \
    --generation-result "$generation"
  python scripts/compute/effectiveness_set.py \
    --execution-summary "$file" \
    --generation-result "$generation"
  echo
done
