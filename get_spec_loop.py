import jsonlines
import re

def main(**kwargs):
    with jsonlines.open('data/train_input_spec_sample2.jsonl') as f:
        sample_idx = kwargs['idx']
        for idx, problem in enumerate(f):
            if idx < sample_idx: continue
            if idx >= sample_idx + 5: 
                order = 'r'
                while(order == 'r'):
                   order = input('...wait...\n')
                   if order == 'q': exit()
                sample_idx = idx
                ...
            name = re.split(r'[. ]', problem['name'])[0]
            
            print(f'{idx} -',name + '\n')
            print('constraints: ')
            input_const = str(problem['input_constraints'])
            input_const = input_const.replace('\\ldots', '...').replace('\\cdots', '...').replace('\\leq', '<=').replace('\\times', 'X').replace('\\neq', '!=').replace('\\ ', ' ')
            
            print(input_const + '\n')
            print('\nspec: ')
            print(problem['input_spec'])
            
            input_sample = problem['public_tests']['input'][0]
            
            print('\ninput_sample')
            print(input_sample)
            
            print('---------------------------------\n')


if __name__ == '__main__':
    import sys
    idx = int(sys.argv[1])
    main(idx = idx)