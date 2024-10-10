shopt -s globstar

for file in ${DATA_DIR}/execution-summary/grammar/**/*.jsonl; do
  echo "$file"
  generation=${file/execution-summary/generation-result}
  for degree in 0 1 2; do
    echo "$degree"
    python scripts/compute/set_effectiveness_degree.py \
      --execution-summary "$file" \
      --generation-result "$generation" \
      --degree "$degree"
    echo
  done
done

