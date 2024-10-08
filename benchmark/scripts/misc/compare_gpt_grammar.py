import jsonlines
from utils import sanitize

file1  = jsonlines.open("data/generation-result/grammar/gpt/1-shot/test.jsonl", "r")
file2  = jsonlines.open("data/generation-result/grammar/gpt/5-shot/test.jsonl", "r")

for f1, f2 in sanitize(zip(file1, file2)):
    if (len(f1["results"]) == 0) != (len(f2["results"]) == 0):
        print(f1["name"])
