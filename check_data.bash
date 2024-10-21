shopt -s globstar

for f in ${DATA_DIR}/{execution-summary,generation-result,grammar,parsing-result}/**/test.jsonl ${RAW_DATA_DIR}/{grammar,level,testcase}/**/test.jsonl; do
  echo -n $f
  filename=$(basename -- "$f" ".jsonl")
  if jq '.["name"]' -r $f | diff - "raw-data/problem-list/${filename}.txt" > /dev/null 2>&1; then
    echo " - OK"
  else
    echo " - Failed"
  fi
done;
