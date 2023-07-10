
def print_info(**kwargs):
    problem = kwargs['problem']
    info = kwargs['info']
    if info != None:
        if info == 'test_case':
            print('public TC:')
            print(problem['public_tests'])

            print('\nprivate TC:')
            print(problem['private_tests'])
            return

        print(f'{info}:')
        print(problem[info])
        return

    print('description:')
    print(problem['description'])

    print('\ninput_spec:')
    print(problem['input_spec'])

    print('\ngrammer:')
    print(problem['grammer'])

    print('\npublic TC:')
    print(problem['public_tests'])

    print('\nprivate TC:')
    print(problem['private_tests'])



if __name__ == "__main__":
    import jsonlines
    import sys


    target = sys.argv[1]
    info = None

    try:
        info = sys.argv[2]
        if info.lower() == 'd':
            info = 'description'
        elif info.lower() == 's':
            info = 'input_spec'
        elif info.lower() == 'g':
            info = 'grammer'
        elif info.lower() == 'p':
            info = 'public_tests'
        elif info.lower() == 'h':
            # hidden test case
            info = 'private_tests'
        elif info.lower() == 't':
            info = 'test_case'
        else:
            info = None
    except:
        pass

    seperate_point = 500
    # dataset = None
    dataset1 = 'input_grammer0_with_spec'
    dataset2 = 'testcase_with_spec'

    if target.isnumeric():
        target = int(target)
        dataset = dataset1 if target < seperate_point else dataset2

        with jsonlines.open(f'data/{dataset}.jsonl') as f:
            for problem in f:
                name, idx = problem['name'].split(' - ')
                idx = int(idx)
                if idx != target: continue
                print_info(problem=problem, info=info)
                exit()

    else:
        with jsonlines.open(f'data/{dataset1}.jsonl') as f:
            for problem in f:
                name, idx = problem['name'].split(' - ')
                if name != target: continue
                print_info(problem=problem, info=info)
                exit()

        with jsonlines.open(f'data/{dataset2}.jsonl') as f:
            for problem in f:
                name, idx = problem['name'].split(' - ')
                if name != target: continue
                print_info(problem=problem, info=info)
                exit()








