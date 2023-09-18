import re

class discriminator():

    def __init__(self, generate_mode=None) -> None:

        self.sep_token = '\t'
        self.new_line_token = '<n>'
        self.space_token = '<s>'
        self.start_token = '<S>'
        self.derivate_token = '->'
        self.blink_token = 'ε'

        self.RE_INTEGER = re.compile(r'-?[0-9]+ *')
        self.RE_NONTERMINAL = re.compile(r'(<[^<]*>)')
        self.RE_STRING = re.compile(r'\[[^\[]*\]\{.*\}')

        self.const_dict = {}
        self.variable_dict = {}
        self.derivation_dict = {}
        self.permutation_variable = []
        self.compare_dict = {}
        self.flag = {"queue": [], "variable": [], "deriv_idx": [], 'test_case': [], 'counter': []}
        self.test_case = ''

        self.derivation_queue = [self.start_token]

        self.generate_mode = 'generate' if generate_mode == None else generate_mode

    def __call__(self, grammer: list, constraints: list, test_case: str) -> bool:
        self.__init__(self.generate_mode)
        self.derivation_queue = [self.start_token]
        self.make_derivate_dict(grammer)
        self.make_constraints_dict(constraints)
        self.test_case = test_case.strip()
        # print(self.const_dict)
        # print(self.derivation_dict)
        # print(self.compare_dict)
        return self.parsing()

    pass

    def parsing(self):
        # self.derivation_queue = [self.start_token]
        # print(1)

        while self.derivation_queue:
            # print(self.derivation_queue)
            # print(self.test_case, '\n')
            # print(self.variable_dict)
            curr_variable = self.derivation_queue[0]
            # print(self.derivation_queue)
            # print(self.flag)

            if curr_variable == self.start_token:
                # self.derivate(curr_variable)
                self.derivation_queue = self.derivation_dict[curr_variable][0].split(' ')
                continue
            if curr_variable == self.blink_token:
                del self.derivation_queue[0]
                continue
            if curr_variable == self.space_token:
                if self.test_case[0] == ' ':
                    self.test_case = self.test_case[1:]
                    del self.derivation_queue[0]
                    continue
                else:
                    raise("Error")
            if curr_variable == self.new_line_token:
                if self.test_case[0] == '\n':
                    self.test_case = self.test_case[1:]
                    del self.derivation_queue[0]
                    continue
                else:
                    raise("Error")
            
            if self.RE_STRING.fullmatch(curr_variable):
                curr_sep = self.get_sep_token()
                if curr_sep == -1:
                    if self.go_flag_point() : continue
                    if self.generate_mode == "test":
                        raise Exception(f"Error1: can't find seperate token\n\t{curr_variable}")

                    return False

                self.test_case = self.test_case.split(curr_sep)
                curr_token = self.test_case[0]
                self.test_case = curr_sep.join(self.test_case[1:])

                str_len = len(curr_token)
                len_var = curr_variable.split(']{')[1][:-1]
                if len_var in self.const_dict:
                    start, end = self.get_range(len_var)
                    if not start <= str_len <= end:
                        if self.go_flag_point() : continue
                        if self.generate_mode == "test":
                            raise Exception(f"Error2-1: length of string is out of range\n\texpected: {start}<=len<={end}\n\treal: {str_len}")
                        return False
                    if len_var in self.variable_dict:
                        if self.variable_dict[len_var] != str_len:
                            excepted = self.variable_dict[len_var]
                            if self.go_flag_point() : continue
                            if self.generate_mode == "test":
                                raise Exception(f"Error2-2: length of string is different\n\texpected: {excepted}\n\treal: {str_len}")
                            return False
                    else:
                        self.variable_dict[len_var] = str_len
                    curr_variable = curr_variable.split('{')[0] + '{'
                    curr_variable = f'{curr_variable}{str_len}' + '}'
                elif ',' in len_var:
                    start, end = len_var.split(',')
                    start = self.get_value(start)
                    end = self.get_value(end)
                    curr_variable = curr_variable.split('{')[0] + '{'
                    curr_variable = f'{curr_variable}{start},{end}' + '}'

                if not re.match(curr_variable, curr_token):
                    if self.go_flag_point() : continue

                    if self.generate_mode == "test":
                        raise Exception(
                            "Error2: string does not matched\n"
                            + f"\tvariable: {curr_variable}\n"
                            + f"\ttest case value: {curr_token}")
                    return False

                del self.derivation_queue[:2]
                continue

            # 기타 terminal 생성
            elif not self.RE_NONTERMINAL.fullmatch(curr_variable):
                curr_sep = self.get_sep_token()
                if curr_sep == -1:
                    if self.go_flag_point() : continue
                    if self.generate_mode == "test":
                        raise Exception(f"Error3: Invalid value - can't find seperate token\n\t{self.derivation_queue}\n\t{self.test_case}")
                    return False

                self.test_case = self.test_case.split(curr_sep)
                curr_token = self.test_case[0]
                self.test_case = curr_sep.join(self.test_case[1:])

                del self.derivation_queue[:2]

                # a_i 형태
                if re.match(r'.*_.*', curr_variable):
                    # print('c', curr_variable)
                    variable, counter = curr_variable.split('_')
                    variable += '_i'
                    counter = int(counter)

                    if variable in self.derivation_dict:
                        if curr_token not in self.derivation_dict[variable]:
                            derivate_list = self.derivation_dict[variable]
                            if self.go_flag_point() : continue
                            if self.generate_mode == "test":
                                raise Exception(f"Error4: Invalid derivation\n\t{curr_token} not in {derivate_list}")
                            return False
                    else:
                        start, end = self.get_range(variable, counter)
                        if not start <= int(curr_token) <= end:
                            if self.go_flag_point() : continue
                            if self.generate_mode == "test":
                                raise Exception(f"Error5: Number is out of value: {variable}\n\texpectd: {start} ~ {end}\n\treal   : {curr_token}")
                            return False
                    if variable not in self.variable_dict or curr_variable in self.variable_dict[variable]:
                        self.variable_dict[variable] = {}

                    if self.RE_INTEGER.fullmatch(curr_token):
                        curr_token = int(curr_token)
                    self.variable_dict[variable][curr_variable] = curr_token

                    continue

                elif curr_variable in self.derivation_dict:
                    if curr_token not in self.derivation_dict[curr_variable]:
                        derivate_list = self.derivation_dict[curr_variable]
                        if self.go_flag_point() : continue
                        if self.generate_mode == "test":
                            raise Exception(f"Error6: Invalid derivation\n\t{curr_token} not in {derivate_list}")

                        return False

                    continue
                else:
                    # N, [N]의 형태
                    if re.match(r'\[[^\[]*\]', curr_variable):
                        curr_variable = curr_variable[1:-1]

                    # reset compare dict
                    if curr_variable in self.variable_dict and curr_variable in self.compare_dict:
                        target = self.compare_dict[curr_variable]['target']
                        self.variable_dict.pop(curr_variable)
                        self.variable_dict.pop(target)

                    if curr_variable not in self.const_dict:
                        # print(curr_variable)

                        if curr_variable in self.derivation_dict:
                            if curr_token not in self.derivation_dict[curr_variable]:
                                if self.go_flag_point() : continue
                                derivate_list = self.derivation_dict[curr_variable]
                                if self.generate_mode == "test":
                                    raise Exception(f"Error7: Invalid derivation\b\t{curr_variable} not in {derivate_list}")
                                return False
                        elif not curr_variable == curr_token:
                            # print(7)
                            if self.go_flag_point() : continue

                            if self.generate_mode == "test":
                                raise Exception(f"Error8: Invalid value\n\texpectd: {curr_token}\n\treal   : {curr_variable}")

                            return False
                        else:
                            # 여기 구현
                            ...

                        continue

                    start, end = self.get_range(curr_variable)
                    if not start <= int(curr_token) <= end:
                        # print(8)
                        if self.go_flag_point() : continue

                        if self.generate_mode == "test":
                            raise Exception(f"Error9: variable is out of range\n\texpected: {start}<={curr_variable}<={end}\n\treal: {curr_token}")
                        return False
                    self.variable_dict[curr_variable] = int(curr_token)

                continue

            # <T_i> 형태
            if re.match(r'<[^_]*_[^_]*>', curr_variable):
                nonterminal , counter = curr_variable.split('_')
                # <T_N>
                if ',' in counter:

                    ...
                if not self.RE_INTEGER.fullmatch(counter[:-1]):
                    # print(self.variable_dict)
                    # print(curr_variable)
                    if '-' in counter or '+' in counter or '*' in counter or re.findall(r'[0-9]+[a-zA-Z]+', counter):
                        counter = self.get_value(counter[:-1])
                    else:
                        counter = self.variable_dict[counter[:-1]]
                    counter = f'{counter}>'
                    curr_variable = nonterminal + '_' + counter
                # <T_1> 등의 상황
                if curr_variable in self.derivation_dict:
                    counter = int(counter[:-1])
                else:
                    curr_variable = nonterminal + '_i>'
                    counter = int(counter[:-1])
            if counter < 0:
                if self.generate_mode == "test":
                    raise Exception(f"Error10: counter have negative value")
                return False
            # derivate
            try:
                self.derivate(curr_variable, counter)
            except:
                self.derivate(curr_variable)

        self.test_case = self.test_case.strip()

        if self.test_case == '': return True

        if self.generate_mode == "test":
            raise Exception("Error11: not finish")

        return False

    def derivate(self, curr_variable, counter = None, deriv_idx = None):
        if len(self.derivation_dict[curr_variable]) > 1:
            self.flag['queue'].append(self.derivation_queue[:])
            self.flag['variable'].append(curr_variable)
            self.flag['deriv_idx'].append(0 if deriv_idx == None else deriv_idx)
            self.flag['test_case'].append(self.test_case[:])
            self.flag['counter'].append(counter)
        # print('1', len(self.derivation_dict[curr_variable]), curr_variable)
        
        next_variable = self.derivation_dict[curr_variable][0 if deriv_idx == None else deriv_idx]
        curr_list = []

        for variable in next_variable.split(' '):
            # <T_i-1>이나 a_i가 생성되었을 때 counter가 필요함
            if re.match(r'<.*_i-1>', variable):
                nonterminal = variable.split('_')[0]
                variable = f'{nonterminal}_{counter-1}>'
            elif re.match(r'<.*_i>', variable):
                nonterminal = variable.split('_')[0]
                variable = f'{nonterminal}_{counter}>'
            elif re.fullmatch(r'[^_<]*_.*', variable):
                nonterminal = variable.split('_')[0]
                variable = f'{nonterminal}_{counter}'
            curr_list.append(variable)

        # derivation이 진행된 이후 완료된 token 제거
        del self.derivation_queue[0]

        # DFS로 진행하기 위해 제거된 variable에서 생성된 variable들을 queue의 앞에 배치함
        curr_list.extend(self.derivation_queue)
        self.derivation_queue = curr_list

    def go_flag_point(self):
        while True:
            if not self.flag['variable']: return False

            # print('g', self.flag)

            queue = self.flag['queue'].pop()
            variable = self.flag['variable'].pop()
            idx = self.flag['deriv_idx'].pop() + 1
            test_case = self.flag['test_case'].pop()
            counter = self.flag['counter'].pop()

            if len(self.derivation_dict[variable]) <= idx:
                continue

            self.derivation_queue = queue
            self.test_case = test_case
            self.derivate(variable, counter, idx)
            return True

        return False

    def get_range(self, variable, counter=None):
        '''
        * terminal variable이 생성할 정수의 범위를 반환하는 함수
        '''
        # print('v', variable)

        curr_token_const = self.const_dict[variable]
        include1 = curr_token_const['include1']
        include2 = curr_token_const['include2']
        addition = 0

        start = self.get_value(curr_token_const['start'])  
        start += 0 if include1 else 1

        end = self.get_value(curr_token_const['end'])  

        # constraint가 "variable < end"의 형태이면
        # "variable <= end-1"과 같다
        end -= 0 if include2 else 1

        derivate_range = [start, end]
        # print('d', type(derivate_range[0]))

        if variable in self.compare_dict:
            target = self.compare_dict[variable]['target']

            if target not in self.variable_dict: 
                # print(derivate_range[0], derivate_range[1])
                return derivate_range[0], derivate_range[1]

            range_index = 0 if self.compare_dict[variable]['symbol'] == '<' else 1
            # print('d1', type(derivate_range[0]))
            # print('i', range_index)
            if self.compare_dict[variable]['type'] == 'same_variable':
                if len(self.variable_dict[variable]) != 0:
                    target = variable.split('_')[0] + f'_{counter-1}'
                    derivate_range[range_index] = self.variable_dict[variable][target]
                    derivate_range[range_index] += 0 if self.compare_dict[variable]['include'] else 1
                else:
                    derivate_range[range_index] += 0 if include1 else 1
            else:
                # print('d2', type(derivate_range[0]))
                target = self.compare_dict[variable]['target']
                if re.match(r'.*_.*', target):
                    # print('d3', type(derivate_range[0]))
                    symbol = target.split('_')[0]
                    symbol += f'_{counter}'
                    derivate_range[range_index] = self.variable_dict[target][symbol]
                    derivate_range[range_index] += 0 if self.compare_dict[variable]['include'] else 1
                else:
                    derivate_range[range_index] = self.variable_dict[target]
                    derivate_range[range_index] += 0 if self.compare_dict[variable]['include'] else 1

        # print(derivate_range[0], derivate_range[1])
        return derivate_range[0], derivate_range[1]

    def get_sep_token(self):
        if len(self.derivation_queue) <= 1:
            return '___'
        if self.derivation_queue[1] == self.new_line_token:
            return '\n'
        elif self.derivation_queue[1] == self.space_token:
            return ' '
        else:
            return -1

    def get_value(self, num):
        negative = False
        # print('n', num)
        if num[0] == '-':
            negative = True
            num = num[1:]

        num = num.replace('min', '$').replace('max', '&')

        targets = re.findall(r'[0-9]+[a-zA-Z]+', num)
        for target in targets:
            change = re.findall(r'[a-zA-Z]+', target)[0]
            num = num.replace(target, target.replace(change, f'*{change}'), 1)        
        
        values = [x.strip() for x in re.split('[-*/+^\(\)$&,]', num)]
        operators = [x.strip() for x in re.findall('[-*/+^\(\)$&,]', num)]
        # print(values)

        # values_num = []
        for idx, value in enumerate(values):
            if not value:
                continue
            if re.fullmatch(r'[-+]?[0-9]*', value):
                continue
            elif re.fullmatch(r'[-+]?[0-9]*\.[0-9]*', value): 
                continue
            else:
                if re.fullmatch(r'[a-zA-Z]*_[0-9]*', value):
                    # print('vv', value)
                    variable, counter = value.split('_')
                    values[idx] = str(self.variable_dict[f'{variable}_i'][value])
                else:
                    values[idx] = str(self.variable_dict[value])

        num = ''

        for value, operator in zip(values, operators):
            if operator == '^':
                operator = '**'
            elif operator == '$':
                operator = 'min'
            elif operator == '&':
                operator = 'max'
            num += (value + operator)
        num += values[-1]

        targets = re.findall(r'[0-9]+\(', num)

        for target in targets:
            change = target[:-1] + '*' + target[-1]
            num = num.replace(target, change)

        return eval(num) * (-1 if negative else 1)

    def is_terminal(self, token):
        return not self.RE_NONTERMINAL.fullmatch(token)

    def make_derivate_dict(self, grammer: list):
        for token in grammer:
            left_hand, right_hand = token.split(self.derivate_token)
            left_hand = left_hand.strip()
            right_hand = right_hand.strip()

            self.derivation_dict[left_hand] = []
            for token in right_hand.split('|'):
                token = token.strip()
                self.derivation_dict[left_hand].append(token)

    def make_constraints_dict(self, constraints: list):
        for const in constraints:
            if re.match(r'[^<]*<=?[^<]*<=?[^<]*', const):

                variables = re.split(r'[<>]=?', const)

                start, end = variables[0], variables[-1]

                del variables[0]
                del variables[-1]

                compare_symbols = re.findall(r'[<>]=?', const)

                for variable_token in variables:
                    for variable in variable_token.split(','):
                        # start, variable, end = re.split(r'<=?', const)
                        self.const_dict[variable.strip()] = {
                            'start': start.strip(), 'end': end.strip(),
                            'include1': compare_symbols[0] == '<=',
                            'include2': compare_symbols[-1] == '<='
                        }
                del compare_symbols[0]
                del compare_symbols[-1]

                if len(variables) > 1:
                    for i in range(len(variables)-1):
                        self.compare_dict[variables[i+1]] = {
                            'target': variables[i],
                            'symbol': compare_symbols[i][0],
                            'include': '=' in compare_symbols[i],
                            'type': 'different_variable'
                        }

            elif re.fullmatch(r'[^<>]*[<>]=?[^<>]*', const):

                variable1, variable2 = re.split(r'[<>]=?', const)
                variable1, variable2 = variable1.strip(), variable2.strip()
                compare_symbol = re.findall(r'[<>]=?', const)[0]
                if '=' not in compare_symbol:
                    v1_const = self.const_dict[variable1]
                    v2_const = self.const_dict[variable2]

                    if compare_symbol == '<':
                        if v1_const['end'] == v2_const['end']:
                            self.const_dict[variable1]['include2'] = False
                    else:
                        if v1_const['start'] == v2_const['start']:
                            self.const_dict[variable1]['include1'] = False


                if variable1.split('_')[0] == variable2.split('_')[0]:
                    self.compare_dict[variable1] = {
                                        'target': variable2,
                                        'symbol': compare_symbol[0],
                                        'include': '=' in compare_symbol,
                                        'type': 'same_variable'
                                        }
                else:
                    self.compare_dict[variable2] = {
                                        'target': variable1,
                                        'symbol': compare_symbol[0],
                                        'include': '=' in compare_symbol,
                                        'type': 'different_variable'
                                        }

            elif re.fullmatch(r'[^=]*!=[^=]*', const):
                variable1, variable2 = const.split('!=')
                self.permutation_variable.append(variable1)


def test():
    import jsonlines
    parser = discriminator('test')
    # file = ''

    with open('res2.txt', 'w', encoding='utf-8') as write_file:
        for file_name in ['train', 'test']:
            # print(file_name)
            with jsonlines.open(f'data/{file_name}_grammar.jsonl') as f:
                for p in f:
                    passed = True
                    # if p['name']['index'] <= 350: continue
                    print(p['name']['name'], '-', p['name']['index'])

                    test_cases = p['public_tests']['input']
                    # test_cases.extend(p['private_tests']['input'])
                    grammar = p['spec']['grammer']
                    const = p['spec']['constraints']

                    for test_case in test_cases:
                        # print(parser(grammar, const, test_case))
                        try:
                            res = parser(grammar, const, test_case)
                        except Exception as e:
                            if passed:
                                name = p['name']['name']
                                index = p['name']['index']

                                write_file.write(f'{name} - {index}')
                                write_file.write('\n')

                                print(p['name']['name'], '-', p['name']['index'])

                            write_file.write(test_case + '\n')

                            passed = False
                            # break
                            write_file.write(e.__str__())
                            write_file.write('\n')
                            print(e)

                    if not passed:
                        print()
                        write_file.write('\n')


def test_one(problem_idx):

    import jsonlines
    import re
    parser = discriminator('test')
    # file = ''

    for file_name in ['train', 'test']:
        # print(file_name)
        with jsonlines.open(f'data/{file_name}_grammar.jsonl') as f:
            for p in f:
                # print(re.split(r'[. ]', p['name']))
                if p['name']['index'] != problem_idx: continue

                # if p['name']['index'] <= 350: continue
                # print(p['name']['name'], '-', p['name']['index'])

                test_cases = p['public_tests']['input']
                # test_cases.extend(p['private_tests']['input'])
                grammar = p['spec']['grammer']
                const = p['spec']['constraints']
                print(p['name']['name'], '-', p['name']['index'])

                for test_case in test_cases:

                    print(test_case)
                    res = parser(grammar, const, test_case)
                    # print()
                    # print(parser(grammar, const, test_case))
                    try:
                        res = parser(grammar, const, test_case)
                        print(res)
                    except Exception as e:
                        print(e)
                    print()
                return


if __name__ == '__main__':
    import sys
    import jsonlines

    test()
    # test_one(913)
    exit()

    file_name = sys.argv[1]
    error_file = f"result_{file_name}_error_list.txt"
    path_file = f"result_{file_name}_pass_list.txt"
    error_reason_file = f"result_{file_name}_error_reason.txt"

    discriminator = discriminator()

    with open(path_file, 'w', encoding='utf-8') as write_file:
        write_file.write('')

    with open(error_file, 'w', encoding='utf-8') as write_file:
        write_file.write('')

    with open(error_reason_file, 'w', encoding='utf-8') as write_file:
        write_file.write('')

    with jsonlines.open(f'data/{file_name}.jsonl') as f:
        for p_idx, problem in enumerate(f, 1):
            print(p_idx)

            name, idx = problem['name'].split(' - ')
            grammer = problem['grammer']
            const = problem['constraints']
            test_case = problem['public_tests']['input'][0]
            try:
                res = discriminator(grammer, const, test_case)

                with open(path_file, 'a', encoding='utf-8') as write_file:
                    write_file.write(f'{name}, {idx}\n')

            except Exception as e:
                with open(error_file, 'a', encoding='utf-8') as write_file:
                    write_file.write(f'{name}, {idx}\n')

                with open(error_reason_file, 'a', encoding='utf-8') as write_file:
                    write_file.write(f'{idx} {name}:\n' + test_case + ('' if test_case[-1] == '\n' else '\n'))
                    write_file.write('\t' + str(e) + '\n\n')

    # "name" =  "71_A - 1"
    test_grammer =  ["<S>->[N] <n> <T_N>", "<T_i>-><T_i-1> <n> [a-z]{1,10^2}", "<T_1>->[a-z]{1,10^2}"]
    test_const = ["1<=N<=100"]
    # "public_tests": "input":
    test_cases = ["4\nword\nlocalization\ninternationalization\npneumonoultramicroscopicsilicovolcanoconiosis\n","26\na\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\ns\nt\nu\nv\nw\nx\ny\nz\n", "10\ngyartjdxxlcl\nfzsck\nuidwu\nxbymclornemdmtj\nilppyoapitawgje\ncibzc\ndrgbeu\nhezplmsdekhhbo\nfeuzlrimbqbytdu\nkgdco\n", "5\nabcdefgh\nabcdefghi\nabcdefghij\nabcdefghijk\nabcdefghijklm\n", "20\nlkpmx\nkovxmxorlgwaomlswjxlpnbvltfv\nhykasjxqyjrmybejnmeumzha\ntuevlumpqbbhbww\nqgqsphvrmupxxc\ntrissbaf\nqfgrlinkzvzqdryckaizutd\nzzqtoaxkvwoscyx\noswytrlnhpjvvnwookx\nlpuzqgec\ngyzqfwxggtvpjhzmzmdw\nrlxjgmvdftvrmvbdwudra\nvsntnjpepnvdaxiporggmglhagv\nxlvcqkqgcrbgtgglj\nlyxwxbiszyhlsrgzeedzprbmcpduvq\nyrmqqvrkqskqukzqrwukpsifgtdc\nxpuohcsjhhuhvr\nvvlfrlxpvqejngwrbfbpmqeirxlw\nsvmasocxdvadmaxtrpakysmeaympy\nyuflqboqfdt\n", "3\nnjfngnrurunrgunrunvurn\njfvnjfdnvjdbfvsbdubruvbubvkdb\nksdnvidnviudbvibd\n", "1\nabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghij\n", "100\nm\nz\ns\nv\nd\nr\nv\ny\ny\ne\np\nt\nc\na\nn\nm\np\ng\ni\nj\nc\na\nb\nq\ne\nn\nv\no\nk\nx\nf\ni\nl\na\nq\nr\nu\nb\ns\nl\nc\nl\ne\nv\nj\nm\nx\nb\na\nq\nb\na\nf\nj\nv\nm\nq\nc\nt\nt\nn\nx\no\ny\nr\nu\nh\nm\nj\np\nj\nq\nz\ns\nj\no\ng\nc\nm\nn\no\nm\nr\no\ns\nt\nh\nr\np\nk\nb\nz\ng\no\nc\nc\nz\nz\ng\nr\n", "1\na\n", "1\ntcyctkktcctrcyvbyiuhihhhgyvyvyvyvjvytchjckt\n", "24\nyou\nare\nregistered\nfor\npractice\nyou\ncan\nsolve\nproblems\nunofficially\nresults\ncan\nbe\nfound\nin\nthe\ncontest\nstatus\nand\nin\nthe\nbottom\nof\nstandings\n"]

    # test_grammer = ["<S>->[01]{S} <n> [01]{S}"]
    # test_const = ["1<=S<=100"]

    # "output": ["word\nl10n\ni18n\np43s\n"]
    # "private_tests": {"input": ["26\na\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\ns\nt\nu\nv\nw\nx\ny\nz\n", "10\ngyartjdxxlcl\nfzsck\nuidwu\nxbymclornemdmtj\nilppyoapitawgje\ncibzc\ndrgbeu\nhezplmsdekhhbo\nfeuzlrimbqbytdu\nkgdco\n", "5\nabcdefgh\nabcdefghi\nabcdefghij\nabcdefghijk\nabcdefghijklm\n", "20\nlkpmx\nkovxmxorlgwaomlswjxlpnbvltfv\nhykasjxqyjrmybejnmeumzha\ntuevlumpqbbhbww\nqgqsphvrmupxxc\ntrissbaf\nqfgrlinkzvzqdryckaizutd\nzzqtoaxkvwoscyx\noswytrlnhpjvvnwookx\nlpuzqgec\ngyzqfwxggtvpjhzmzmdw\nrlxjgmvdftvrmvbdwudra\nvsntnjpepnvdaxiporggmglhagv\nxlvcqkqgcrbgtgglj\nlyxwxbiszyhlsrgzeedzprbmcpduvq\nyrmqqvrkqskqukzqrwukpsifgtdc\nxpuohcsjhhuhvr\nvvlfrlxpvqejngwrbfbpmqeirxlw\nsvmasocxdvadmaxtrpakysmeaympy\nyuflqboqfdt\n", "3\nnjfngnrurunrgunrunvurn\njfvnjfdnvjdbfvsbdubruvbubvkdb\nksdnvidnviudbvibd\n", "1\nabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghijabcdefghij\n", "100\nm\nz\ns\nv\nd\nr\nv\ny\ny\ne\np\nt\nc\na\nn\nm\np\ng\ni\nj\nc\na\nb\nq\ne\nn\nv\no\nk\nx\nf\ni\nl\na\nq\nr\nu\nb\ns\nl\nc\nl\ne\nv\nj\nm\nx\nb\na\nq\nb\na\nf\nj\nv\nm\nq\nc\nt\nt\nn\nx\no\ny\nr\nu\nh\nm\nj\np\nj\nq\nz\ns\nj\no\ng\nc\nm\nn\no\nm\nr\no\ns\nt\nh\nr\np\nk\nb\nz\ng\no\nc\nc\nz\nz\ng\nr\n", "1\na\n", "1\ntcyctkktcctrcyvbyiuhihhhgyvyvyvyvjvytchjckt\n", "24\nyou\nare\nregistered\nfor\npractice\nyou\ncan\nsolve\nproblems\nunofficially\nresults\ncan\nbe\nfound\nin\nthe\ncontest\nstatus\nand\nin\nthe\nbottom\nof\nstandings\n"], "output": ["a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no\np\nq\nr\ns\nt\nu\nv\nw\nx\ny\nz\n", "g10l\nfzsck\nuidwu\nx13j\ni13e\ncibzc\ndrgbeu\nh12o\nf13u\nkgdco\n", "abcdefgh\nabcdefghi\nabcdefghij\na9k\na11m\n", "lkpmx\nk26v\nh22a\nt13w\nq12c\ntrissbaf\nq21d\nz13x\no17x\nlpuzqgec\ng18w\nr19a\nv25v\nx15j\nl28q\ny26c\nx12r\nv26w\ns27y\ny9t\n", "n20n\nj27b\nk15d\n", "a98j\n", "m\nz\ns\nv\nd\nr\nv\ny\ny\ne\np\nt\nc\na\nn\nm\np\ng\ni\nj\nc\na\nb\nq\ne\nn\nv\no\nk\nx\nf\ni\nl\na\nq\nr\nu\nb\ns\nl\nc\nl\ne\nv\nj\nm\nx\nb\na\nq\nb\na\nf\nj\nv\nm\nq\nc\nt\nt\nn\nx\no\ny\nr\nu\nh\nm\nj\np\nj\nq\nz\ns\nj\no\ng\nc\nm\nn\no\nm\nr\no\ns\nt\nh\nr\np\nk\nb\nz\ng\no\nc\nc\nz\nz\ng\nr\n", "a\n", "t41t\n", "you\nare\nregistered\nfor\npractice\nyou\ncan\nsolve\nproblems\nu10y\nresults\ncan\nbe\nfound\nin\nthe\ncontest\nstatus\nand\nin\nthe\nbottom\nof\nstandings\n"]}, "description": "Sometimes some words like \"localization\" or \"internationalization\" are so long that writing them many times in one text is quite tiresome.\n\nLet's consider a word too long, if its length is strictly more than 10 characters. All too long words should be replaced with a special abbreviation.\n\nThis abbreviation is made like this: we write down the first and the last letter of a word and between them we write the number of letters between the first and the last letters. That number is in decimal system and doesn't contain any leading zeroes.\n\nThus, \"localization\" will be spelt as \"l10n\", and \"internationalization» will be spelt as \"i18n\".\n\nYou are suggested to automatize the process of changing the words with abbreviations. At that all too long words should be replaced by the abbreviation and the words that are not too long should not undergo any changes.\n\nInput\n\nThe first line contains an integer n (1 ≤ n ≤ 100). Each of the following n lines contains one word. All the words consist of lowercase Latin letters and possess the lengths of from 1 to 100 characters.\n\nOutput\n\nPrint n lines. The i-th line should contain the result of replacing of the i-th word from the input data.\n\nExamples\n\nInput\n\n4\nword\nlocalization\ninternationalization\npneumonoultramicroscopicsilicovolcanoconiosis\n\n\nOutput\n\nword\nl10n\ni18n\np43s"}

    '''
    discriminator = discriminator()
    for test_case in test_cases:
        res = discriminator(test_grammer, test_const, test_case)

        print(res)
    '''
    ...
