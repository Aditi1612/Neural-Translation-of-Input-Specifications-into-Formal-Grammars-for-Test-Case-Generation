from typing import Any
import jsonlines
import re
import random
import math
import string

class fuzzing():
    def __init__(self):
        self.fuzzing_rate = 0.3
        ...
        
    def __call__(self, test_cases:[str]) -> [[str],]:
        res = []
        for idx, test_case in enumerate(test_cases):
            test_case = test_case.rstrip()
            print(idx)
            tokens_origin = re.split(r'[ \n]', test_case)
            spliters = re.findall(r'[ \n]', test_case)
            num_of_tokens = len(tokens_origin)
            num_of_sample = math.ceil(num_of_tokens*self.fuzzing_rate)
            weight = [1]*len(tokens_origin)
            # 첫 token이 정수라면 test case의 수를 나타내는 경우가 대부분이므로 선택하지 않는다.
            weight[0] = 0 if re.fullmatch(r'[0-9]*', tokens_origin[0]) and num_of_tokens > 1 else 1
            mutateds = []
            for _ in range(10):
                tokens = tokens_origin[:]
                targets = random.choices(range(num_of_tokens), weights=weight, k=num_of_sample)
                mutated = ''
                for target_idx in targets:
                    target = tokens[target_idx]
                    # print(target)
                    if re.fullmatch(r'[-+]?([0-9]+\.[0-9]+|[0-9]+)', target):
                        tokens[target_idx] = str(int(target) + (1 if random.random() >= 0.5 else -1))
                    elif re.fullmatch(r'[a-zA-Z]*', target):
                        str_list = list(self.get_str_list(target))
                        remove_index = random.choice(range(len(target)))
                        str_list.remove(target[remove_index])
                        tokens[target_idx] = target[:remove_index-1] + random.choice(str_list) + target[remove_index:]
                    else:
                        change = list(target)
                        print(target)
                        while(''.join(change) == target and len(target) > 1 and len(set(change)) > 1):
                            random.shuffle(change)
                        tokens[target_idx] = ''.join(change)
                        
                mutated += tokens[0]
                for token, spliter in zip(tokens[1:], spliters):
                    mutated += spliter
                    mutated += token
                mutateds.append(mutated)
            res.append(mutateds)
            
        return res

    def get_str_list(self, token:str):
        if token == token.lower():
            return string.ascii_lowercase
        elif token == token.upper():
            return string.ascii_uppercase
        else:
            return string.ascii_letters
            
if __name__ == "__main__":
    fuzzer = fuzzing()
    test_case = ['1 6 6 2 1 1\n', '5 1 4 4 2 1\n', '4 1 7 4 1 2\n', '3 1\n', '4 3\n', '10\n', '1\n', '3 3\nWWW\nBWW\nWWW\n', '5 6\nWWBBBW\nWWBBBW\nWWBBBW\nWWWWWW\nWWWWWW\n', '6\n1\n2\n3\n4\n5\n6\n', 'aabb\nabab\n', 'aaba\nabaa\n', '###.\n....\n####\n']
    print(fuzzer(test_case))
    pass
