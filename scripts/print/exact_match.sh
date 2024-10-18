shopt -s globstar

for file in ${RAW_DATA_DIR}/grammar/**/*.jsonl; do
  echo "$file"
  python scripts/compute/exact_match.py --grammar "$file"
  echo
done
