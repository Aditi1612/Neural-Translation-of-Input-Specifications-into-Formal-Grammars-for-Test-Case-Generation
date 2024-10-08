shopt -s globstar

for file in ${DATA_DIR}/execution-summary/**/*.jsonl; do
  echo "$file"
  generation=${file/execution-summary/generation-result}
  echo "element-wise"
  python scripts/compute/elementwise_effectiveness_level.py \
    --execution-summary "$file" \
    --generation-result "$generation"
  echo "set"
  python scripts/compute/set_effectiveness_level.py \
    --execution-summary "$file" \
    --generation-result "$generation"
  echo
done
