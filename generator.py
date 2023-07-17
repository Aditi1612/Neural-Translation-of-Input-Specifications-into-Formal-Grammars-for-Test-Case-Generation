import re
import random
import time

class TestCaseGenerator():

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

        self.generate_mode = 'generate' if generate_mode == None else generate_mode

    def __call__(self, grammer: list, constraints: list) -> str:
        self.__init__(self.generate_mode)
        self.make_derivate_dict(grammer)
        self.make_constraints_dict(constraints)
        # print(self.const_dict)
        # print(self.compare_dict)
        # print(self.derivation_dict)
        test_case = self.derivate()
        return test_case
        # return result

    def derivate(self):
        test_case = ''
        derivation_queue = [self.start_token]
        start_time = time.time()

        # 모든 variable이 test case를 생성할 때 까지
        while derivation_queue:

            # print(derivation_queue)

            curr_time = time.time()
            if curr_time - start_time > 4 and self.generate_mode == 'test':
                raise Exception(f"Error12: infinite loop")
            curr_variable = derivation_queue[0]
            del derivation_queue[0]

            # 공백, 개행문자 생성
            if curr_variable == self.new_line_token:
                test_case += '\n'
                continue

            if curr_variable == self.space_token:
                test_case += ' '
                continue
            if curr_variable == self.blink_token:
                continue

            # string 생성
            if self.RE_STRING.fullmatch(curr_variable):
                generated = self.get_string(curr_variable)
                test_case += generated
                continue


            # 기타 terminal 생성
            elif not self.RE_NONTERMINAL.fullmatch(curr_variable):

                # a_i 형태
                if re.match(r'.*_.*', curr_variable):

                    variable, counter = curr_variable.split('_')
                    variable += '_i'
                    counter = int(counter)
                    if counter < 0:
                        raise Exception(f"Error10: counter have negative value")

                    if variable in self.derivation_dict:
                        assert False
                        curr_list = []
                        next_variable = random.choice(self.derivation_dict[variable])
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

                        curr_list.extend(derivation_queue)
                        derivation_queue = curr_list
                        continue
                    # a_i 형태의 terminal을 처음 인식하거나 a_1을 생성할 때
                    if variable not in self.variable_dict or curr_variable in self.variable_dict[variable]:
                        self.variable_dict[variable] = {}

                    start, end = self.get_range(variable, counter)

                    generated = random.randint(start, end)

                    # 순열일 경우에 이전에 생성되지 않은 정수 생성
                    while_start_time = time.time()
                    while variable in self.permutation_variable and generated in self.variable_dict[variable].values():
                        curr_time = time.time()
                        if curr_time - while_start_time > 4:
                            raise Exception(f"Error12-2: infinite loop to make permutation")
                        # print(generated)
                        generated = random.randint(start, end)

                    # 생성된 것 저장
                    self.variable_dict[variable][curr_variable] = generated

                elif curr_variable in self.derivation_dict:
                    test_case += random.choice(self.derivation_dict[curr_variable])
                    continue
                else:
                    # N, [N]의 형태
                    if re.match(r'\[[^\[]*\]', curr_variable):
                        curr_variable = curr_variable[1:-1]

                    if curr_variable not in self.const_dict:

                        test_case += curr_variable
                        continue

                    start, end = self.get_range(curr_variable)
                    # print(curr_variable)
                    generated = random.randint(start, end)
                    self.variable_dict[curr_variable] = generated

                # 생성된 결과물 최종 반환될 test_case에 더하기
                test_case = f'{test_case}{generated}'
                continue

            # <T_i> 형태
            if re.match(r'<[^_]*_[^_]*>', curr_variable):
                nonterminal , counter = curr_variable.split('_')
                # <T_N>
                if ',' in counter:

                    ...
                if not self.RE_INTEGER.fullmatch(counter[:-1]):
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
                    raise Exception(f"Error10: counter have negative value")
            # derivate
            next_variable = random.choice(self.derivation_dict[curr_variable])
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

            # DFS로 진행하기 위해 제거된 variable에서 생성된 variable들을 queue의 앞에 배치함
            curr_list.extend(derivation_queue)
            derivation_queue = curr_list

        return test_case

    def get_addition(self, token):

        if re.match(r'[^+]+\+[0-9]+', token):
            token, addition = token.split('+')
            return token, int(addition)
        elif re.match(r'[^-]+-[0-9]+', token):
            token, addition = token.split('-')
            return token, int(addition) * -1
        else:
            return token, 0

    def get_range(self, variable, counter=None):
        '''
        * terminal variable이 생성할 정수의 범위를 반환하는 함수
        '''
        # print(variable)
        curr_token_const = self.const_dict[variable]
        include1 = curr_token_const['include1']
        include2 = curr_token_const['include2']
        addition = 0
        start, addition = self.get_addition(curr_token_const['start'])
        if start in self.variable_dict:
            if counter != None:
                start = self.variable_dict[start][start.split('_')[0] + f'_{counter}']
                # print(start)
            else:
                start = self.variable_dict[start]
        else: start = self.get_value(start)

        start = int(start) + addition
        # variable의 constraints가 a_i < a_i+1의 형태라면
        # a_i가 생성할 정수의 범위는 a_i-1 <= a_i <= end이다

        # constraint가 "start < variable"의 형태이면
        # "start+1 <= variable"과 같다
        start += 0 if include1 else 1

        end, addition = self.get_addition(curr_token_const['end'])

        end = self.get_value(end)
        end = end + addition
        # constraint가 "variable < end"의 형태이면
        # "variable <= end-1"과 같다
        end -= 0 if include2 else 1

        derivate_range = [start, end]

        if variable in self.compare_dict:
            range_index = 0 if self.compare_dict[variable]['symbol'] == '<' else 1
            # print('i', range_index)
            if self.compare_dict[variable]['type'] == 'same_variable':
                if len(self.variable_dict[variable]) != 0:
                    target = variable.split('_')[0] + f'_{counter-1}'
                    derivate_range[range_index] = self.variable_dict[variable][target]
                    derivate_range[range_index] += 0 if self.compare_dict[variable]['include'] else 1
                else:
                    derivate_range[range_index] += 0 if include1 else 1
            else:
                target = self.compare_dict[variable]['target']
                if re.match(r'.*_.*', target):
                    symbol = target.split('_')[0]
                    symbol += f'_{counter}'
                    derivate_range[range_index] = self.variable_dict[target][symbol]
                    derivate_range[range_index] += 0 if self.compare_dict[variable]['include'] else 1
                else:
                    derivate_range[range_index] = self.variable_dict[target]
                    derivate_range[range_index] += 0 if self.compare_dict[variable]['include'] else 1

        # print(derivate_range)
        return derivate_range[0], derivate_range[1]

    def get_string(self, variable):
        '''
        * string 만들어서 return하고
        * variable_dict에 list 넣기
        '''
        return_str = ''

        variable_range, string_len = variable.split(']{')
        variable_range = variable_range[1:]
        string_len = string_len[:-1]

        if string_len in self.variable_dict:
            string_len = int(self.variable_dict[string_len])
        elif ',' in string_len:
            start, end = string_len.split(',')
            string_len = random.randint(self.get_value(start), self.get_value(end))
        elif self.RE_INTEGER.fullmatch(string_len):
            string_len = int(string_len)
        else:
            start, end = self.get_range(string_len)
            variable = string_len
            string_len = random.randint(start, end)
            self.variable_dict[variable] = string_len

        if variable in self.derivation_dict:
            variable_list = self.derivation_dict[variable]
        else:
            variable_list = []
            while '-' in variable_range:
                idx = variable_range.find('-')
                char_range = variable_range[idx-1:idx+2]

                start, end = char_range.split('-')
                start = ord(start)
                end = ord(end)+1
                for ch in range(start, end):
                    variable_list.append(chr(ch))
                if idx == 1: variable_range = variable_range[3:]
                # elif
                else: variable_range = variable_range[:idx-1] + variable_range[idx+2:]
            for ch in variable_range:
                variable_list.append(ch)
            self.derivation_dict[variable] = variable_list

        for _ in range(string_len):
            return_str += random.choice(variable_list)

        return return_str

    def get_value(self, num):
        # print(2, num)
        if num in self.variable_dict:
            return self.variable_dict[num]

        elif 'min' in num:
            num = re.findall(r'\(.*,.*\)', num)[0]
            a, b = self.get_value(num.split(',')[0][1:]), self.get_value(num.split(',')[1][:-1])
            return min(a,b)

        elif 'max' in num:
            num = re.findall(r'\(.*,.*\)', num)[0]
            a, b = self.get_value(num.split(',')[0][1:]), self.get_value(num.split(',')[1][:-1])
            return max(a,b)

        elif '^' in num:
            if self.generate_mode == 'test':
                res = base, exp = num.split('^')
                if '*' in base:
                    bias , base = base.split('*')
                    return int(base) * int (bias)
                # print('1', ',', base)
                return int(base)

            if '*' in num:
                bias, num = num.split('*')
                base, exp = num.split('^')
                res = int(bias) * int(base) ** int(exp)
            else:
                base, exp = num.split('^')
                res = int(base) ** int(exp)
                if base[0] == '-' and res > 0:
                    return -res
            return res
        else:
            if self.generate_mode == 'test':
                num = int(num)
                # print(2, ',', num)
                while num > 50:
                    num //= 10
                # print(3, ',', num)
                return num
            return int(num)

    def is_terminal(self, token):
       return not self.RE_NONTERMINAL.fullmatch(token)

    def make_derivate_dict(self, grammer: list):
        for token in grammer:
            left_hand, right_hand = token.split(self.derivate_token)
            self.derivation_dict[left_hand] = []
            for token in right_hand.split('|'):
                self.derivation_dict[left_hand].append(token)

    def make_constraints_dict(self, constraints:list):
        for const in constraints:
            if re.match(r'[^<]*<=?[^<]*<=?[^<]*', const):

                variables = re.split(r'[<>]=?', const)

                start, end = variables[0], variables[-1]
                if self.generate_mode == 'test':
                    if '^' in end:
                        base = end.split('^')[0]

                        if '*' in base:
                            bias, base = base.split('*')
                            end = str(int(bias) * int(base))
                        else:
                            end = base
                    elif self.RE_INTEGER.fullmatch(end):
                        end = int(end)
                        while end >  50:
                            end //= 10
                            # print(1)
                        end = str(end)
                        ...
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

if __name__ == '__main__':

    test_grammer = ['<S> -> [N]', '[N] -> <enter> <T_N>', '<T_i> -> M <enter> <T_i-1>', 'M -> <L_M>', '<L_i> -> C_i <space> <L_i-1>', '<T_0> -> ε', '<L_0> -> ε'] # 'C_i -> [a-z]',
    test_const = ['1 <= N <= 50', '1 < M < 10', '1 <= C_i <= M', 'C_i != C_j']

    # test_grammer = ['<S> -> M <space> N']
    # test_const = ['1 <= M <= 16', 'M <= N <= 16', 'a <= b']

    generator = TestCaseGenerator()
    generator.get_string('[a-z]{S}')
    # res = generator(test_grammer, test_const)
    # print(res)

