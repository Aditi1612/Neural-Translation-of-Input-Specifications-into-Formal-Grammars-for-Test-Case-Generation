mkdir -p ./table/ccfg
for f in results/ccfg/*.jsonl; do
  echo "./table/ccfg/$(basename "$f" .jsonl).tex"
  python3 validate_labeling.py \
    --labeled-data "$f" \
    --testcase "data/unlabeled/code_contests_train_python.jsonl" \
    > "./table/ccfg/$(basename "$f" .jsonl).tex"
done
