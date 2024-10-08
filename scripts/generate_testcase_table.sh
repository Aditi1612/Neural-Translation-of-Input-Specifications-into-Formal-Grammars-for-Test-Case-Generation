mkdir -p ./table/testcase
for f in results/testcase/*.jsonl; do
  echo "./table/testcase/$(basename "$f" .jsonl).tex"
  python3 validate_testcase.py --testcase "$f" --type model_generated \
    > "./table/testcase/$(basename "$f" .jsonl).tex"
done
