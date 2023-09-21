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
        
        self.have_epsilon_transition = False

        self.derivation_queue = [self.start_token]

        self.generate_mode = generate_mode if generate_mode else 'generate'

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

        while self.derivation_queue:
            counter = None
            # print(self.variable_dict)
            curr_variable = self.derivation_queue[0]
            # print(self.flag)
            # print(self.derivation_queue)
            # print(self.test_case)
            if curr_variable == self.start_token:
                self.derivate(curr_variable)
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
                if self.space_token in curr_variable:
                    curr_variable = curr_variable.replace(self.space_token, ' ')
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
                else:
                    curr_variable = curr_variable.split('{')[0] + '{'
                    curr_variable = f'{curr_variable}{self.get_value(len_var)}' + '}'

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
                
                # a_i 형태
                if re.match(r'.*_.*', curr_variable):
                    # print('c', curr_variable)
                    variable, counter = curr_variable.split('_')
                    variable += '_i'
                    counter = int(counter)

                    if variable in self.derivation_dict:
                        self.derivate(variable, counter)
                        continue

                    else:
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
                    self.derivate(curr_variable)
                    '''
                    if curr_token not in self.derivation_dict[curr_variable]:
                        # 여기에 string이 안에 있을 때 확인 필요
                        derivate_list = self.derivation_dict[curr_variable]
                        if self.go_flag_point() : continue
                        if self.generate_mode == "test":
                            raise Exception(f"Error6: Invalid derivation\n\t{curr_token} not in {derivate_list}")

                        return False
                    '''
                    continue
                elif curr_variable in self.const_dict or re.fullmatch(r'\[[a-zA-Z]*\]', curr_variable):
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
                            
                            self.derivate(curr_variable)
                            '''
                            if curr_token not in self.derivation_dict[curr_variable]:
                                if self.go_flag_point() : continue
                                derivate_list = self.derivation_dict[curr_variable]
                                if self.generate_mode == "test":
                                    raise Exception(f"Error7: Invalid derivation\b\t{curr_variable} not in {derivate_list}")
                                return False
                            '''
                        continue

                    start, end = self.get_range(curr_variable)
                    if not start <= int(curr_token) <= end:
                        # print(8)
                        if self.go_flag_point() : continue

                        if self.generate_mode == "test":
                            raise Exception(f"Error9: variable is out of range\n\texpected: {start}<={curr_variable}<={end}\n\treal: {curr_token}")
                        return False
                    self.variable_dict[curr_variable] = int(curr_token)
                
                else:
                    # 여기 작업할 차례
                    curr_sep = self.get_sep_token()
                    if curr_sep == -1:
                        if self.go_flag_point() : continue
                        if self.generate_mode == "test":
                            raise Exception(f"Error3: Invalid value - can't find seperate token\n\t{self.derivation_queue}\n\t{self.test_case}")
                        return False

                    self.test_case = self.test_case.split(curr_sep)
                    curr_token = self.test_case[0]
                    self.test_case = curr_sep.join(self.test_case[1:])

                    if curr_token == curr_variable:
                        del self.derivation_queue[:2]
                         
                    # if self.test_case.find(curr_variable) == 0:
                    #     self.test_case = self.test_case.replace(curr_variable, '', 1)
                    #     del self.derivation_queue[0]
                    else:
                        if self.go_flag_point() : continue
                        if self.generate_mode == "test":
                            raise Exception(f"Error8: Invalid value: \n\texpectd: {self.test_case}\n\treal   : {curr_variable}")

                        return False
                continue

            # <T_i> 형태
            if re.fullmatch(r'<[a-zA-Z]*>', curr_variable):
                self.derivate(curr_variable)
                continue
            if re.match(r'<[^_]*_[^_]*>', curr_variable):
                nonterminal , counter = curr_variable.split('_')
                # <T_N>
                if ',' in counter:

                    ...
                if not self.RE_INTEGER.fullmatch(counter[:-1]):
                    # print(self.variable_dict)
                    # print(curr_variable)
                    # if '-' in counter or '+' in counter or '*' in counter or re.findall(r'[0-9]+[a-zA-Z]+', counter):
                    #     counter = self.get_value(counter[:-1])
                    # else:
                    #     counter = self.variable_dict[counter[:-1]]
                    counter = self.get_value(counter[:-1])
                    counter = f'{counter}>'
                    curr_variable = f'{nonterminal}_{counter}'
                    
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
            self.derivate(curr_variable, counter)
                
            if not self.test_case and not self.have_epsilon_transition:
                if self.go_flag_point():
                    continue
                raise("Error")

        self.test_case = self.test_case.strip()

        if self.test_case == '': return True

        if self.generate_mode == "test":
            raise Exception("Error11: not finish")

        return False

    def derivate(self, curr_variable, counter = None, deriv_idx = None):
        # print('c', curr_variable, counter)
        if len(self.derivation_dict[curr_variable]) > 1:
            self.flag['queue'].append(self.derivation_queue[:])
            self.flag['variable'].append(curr_variable)
            self.flag['deriv_idx'].append(deriv_idx if deriv_idx else 0)
            self.flag['test_case'].append(self.test_case[:])
            self.flag['counter'].append(counter)
        # print('1', len(self.derivation_dict[curr_variable]), curr_variable)
        
        next_variable = self.derivation_dict[curr_variable][deriv_idx if deriv_idx else 0]
        
        curr_list = []

        for variable in next_variable.split(' '):
            # <T_i-1>이나 a_i가 생성되었을 때 counter가 필요함
            if re.match(r'<.*_i-1>', variable):
                nonterminal = variable.split('_')[0]
                variable = f'{nonterminal}_{counter-1}>'
            elif re.fullmatch(r'[^_<]*_i', variable):
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
            
            # print('f', self.derivation_queue)
            # print('f', test_case)

            return True

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
                if self.blink_token in token: self.have_epsilon_transition = True

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