# problem = {"name": {"origin": "p02689 AtCoder Beginner Contest 166 - Peaks", "name": "p02689", "index": 62}, "description": {"description": "There are N observatories in AtCoder Hill, called Obs. 1, Obs. 2, ..., Obs. N. The elevation of Obs. i is H_i. There are also M roads, each connecting two different observatories. Road j connects Obs. A_j and Obs. B_j.\n\nObs. i is said to be good when its elevation is higher than those of all observatories that can be reached from Obs. i using just one road. Note that Obs. i is also good when no observatory can be reached from Obs. i using just one road.\n\nHow many good observatories are there?\n\nConstraints\n\n* 2 \\leq N \\leq 10^5\n* 1 \\leq M \\leq 10^5\n* 1 \\leq H_i \\leq 10^9\n* 1 \\leq A_i,B_i \\leq N\n* A_i \\neq B_i\n* Multiple roads may connect the same pair of observatories.\n* All values in input are integers.\n\nInput\n\nInput is given from Standard Input in the following format:\n\n\nN M\nH_1 H_2 ... H_N\nA_1 B_1\nA_2 B_2\n:\nA_M B_M", "spec": "Constraints\n\n* 2 \\leq N \\leq 10^5\n* 1 \\leq M \\leq 10^5\n* 1 \\leq H_i \\leq 10^9\n* 1 \\leq A_i,B_i \\leq N\n* A_i \\neq B_i\n* Multiple roads may connect the same pair of observatories.\n* All values in input are integers.\n\nInput\n\nInput is given from Standard Input in the following format:\n\n\nN M\nH_1 H_2 ... H_N\nA_1 B_1\nA_2 B_2\n:\nA_M B_M", "origin": "There are N observatories in AtCoder Hill, called Obs. 1, Obs. 2, ..., Obs. N. The elevation of Obs. i is H_i. There are also M roads, each connecting two different observatories. Road j connects Obs. A_j and Obs. B_j.\n\nObs. i is said to be good when its elevation is higher than those of all observatories that can be reached from Obs. i using just one road. Note that Obs. i is also good when no observatory can be reached from Obs. i using just one road.\n\nHow many good observatories are there?\n\nConstraints\n\n* 2 \\leq N \\leq 10^5\n* 1 \\leq M \\leq 10^5\n* 1 \\leq H_i \\leq 10^9\n* 1 \\leq A_i,B_i \\leq N\n* A_i \\neq B_i\n* Multiple roads may connect the same pair of observatories.\n* All values in input are integers.\n\nInput\n\nInput is given from Standard Input in the following format:\n\n\nN M\nH_1 H_2 ... H_N\nA_1 B_1\nA_2 B_2\n:\nA_M B_M\n\n\nOutput\n\nPrint the number of good observatories.\n\nExamples\n\nInput\n\n4 3\n1 2 3 4\n1 3\n2 3\n2 4\n\n\nOutput\n\n2\n\n\nInput\n\n6 5\n8 6 9 1 2 1\n1 3\n4 2\n4 3\n4 6\n4 6\n\n\nOutput\n\n3"}, "public_tests": {"input": ["6 5\n8 6 9 1 2 1\n1 3\n4 2\n4 3\n4 6\n4 6", "4 3\n1 2 3 4\n1 3\n2 3\n2 4"], "output": ["3", "2"]}, "private_tests": {"input": [], "output": []}, "spec": {"grammer": ["<S>->[N] <s> [M] <n> <T_N> <n> <L_M>", "<T_i>-><T_i-1> <s> H_i", "<T_1>->H_i", "<L_i>-><L_i-1> <n> A_i <s> B_i", "<L_1>->A_i <s> B_i"], "constraints": ["2<=N<=10^5", "1<=M<=10^5", "1<=H_i<=10^9", "1<=A_i<=N", "1<=B_i<=N", "A_i!=B_i"], "spec": "Constraints\n\n* 2 \\leq N \\leq 10^5\n* 1 \\leq M \\leq 10^5\n* 1 \\leq H_i \\leq 10^9\n* 1 \\leq A_i,B_i \\leq N\n* A_i \\neq B_i\n* Multiple roads may connect the same pair of observatories.\n* All values in input are integers.\n\nInput\n\nInput is given from Standard Input in the following format:\n\n\nN M\nH_1 H_2 ... H_N\nA_1 B_1\nA_2 B_2\n:\nA_M B_M"}}

from discriminator import discriminator 
from generator import test_case_generator
import sys

try:
    import jsonlines
    name = sys.argv[1]
    if name.isdigit(): 
        name_type = 'index'
        name = int(name)
    else: name_type = 'name'
    with jsonlines.open('data/train_grammer.jsonl') as f:
        for p in f:
            if p['name'][name_type] == name:
                problem = p
                break
    print(problem['spec']['grammer'])
    print(problem['spec']['constraints'])
except Exception as e:
    print(e)
    pass

try:
    test_mode = sys.argv[2]
except:
    test_mode = ''

parser = discriminator(test_mode)
generator = test_case_generator(test_mode)

name = problem['name']['name']
index = problem['name']['index']

grammer = problem['spec']['grammer']
const = problem['spec']['constraints']

test_case = problem['public_tests']['input'][0]
print(test_case)
print('-----')
try:
    res = generator(grammer, const)
    print('generator: ', res, sep='\n')
except Exception as e:
    print(e)
try:
    res = parser(grammer, const, test_case)
    print('parser: ', res)
except Exception as e:
    print(e)
