import jsonlines

with jsonlines.open('data/train_input_spec_sample.jsonl') as f:
    for problem in f:
        correct = len(problem['solutions'])
        incorrect = len(problem['incorrect_solutions'])
        print('%-3d + %-3d = %-3d' %(correct, incorrect, correct + incorrect))