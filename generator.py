
class test_case_generator():
    import re
    import random
    
    def __init__(self) -> None:
        self.sep_token = '\t'
        self.new_line_token = '<n>'
        self.space_token = '<s>'
        self.start_token = '<S>'
        self.derivate_token = '->'
        self.RE_NONTERMINAL = self.re.compile(r'(<[^<]*>)')
        self.RE_STRING = self.re.compile(r'\[[^\[]*\]\{.*\}')
        self.const_dict = {}
        self.variable_dict = {}
        self.derivation_dict = {}
        self.permutation_variable = []
        self.compare_dict = {}

    def __call__(self, grammer: list, constraints: list) -> str:
        self.__init__()
        self.make_derivate_dict(grammer)
        self.make_constraints_dict(constraints)
        print(self.const_dict)
        print(self.compare_dict)
        test_case = self.derivate()
        return test_case
        # return result
          
    def derivate(self):
        test_case = ''
        derivation_queue = [self.start_token]
        
        # print(derivation_queue)
        
        # 모든 variable이 test case를 생성할 때 까지
        while derivation_queue:
            curr_variable = derivation_queue[0]
            
            # 공백, 개행문자 생성
            if curr_variable == self.new_line_token:
                test_case += '\n'
                del derivation_queue[0]
                
                continue
            
            if curr_variable == self.space_token:
                test_case += ' '
                del derivation_queue[0]
                
                continue
            
            # string 생성
            if self.RE_STRING.fullmatch(curr_variable):
                generated = self.get_string(curr_variable)
                test_case += generated
                
                del derivation_queue[0]
                
                continue
            
            
            # 기타 terminal 생성
            elif not self.RE_NONTERMINAL.fullmatch(curr_variable):
                
                # a_i 형태
                if self.re.match(r'.*_.*', curr_variable):
                    
                    variable, counter = curr_variable.split('_')
                    variable += '_i'
                    counter = int(counter)
                    
                    # a_i 형태의 terminal을 처음 인식하거나 a_1을 생성할 때
                    if variable not in self.variable_dict or curr_variable in self.variable_dict[variable]:
                        self.variable_dict[variable] = {}
                    
                    start, end = self.get_range(variable, counter)
                    
                    generated = self.random.randint(start, end)
                    
                    # 순열일 경우에 이전에 생성되지 않은 정수 생성
                    while variable in self.permutation_variable and generated in self.variable_dict[variable].values():
                        generated = self.random.randint(start, end)
                    
                    # 생성된 것 저장
                    self.variable_dict[variable][curr_variable] = generated
                else:
                    # N, [N]의 형태
                    if self.re.match(r'\[[^\[]*\]', curr_variable):
                        curr_variable = curr_variable[1:-1]
                    
                    if curr_variable not in self.derivation_dict and curr_variable not in self.const_dict:
                        
                        test_case += curr_variable
                        del derivation_queue[0]
                        continue
                        
                    start, end = self.get_range(curr_variable)
                    generated = self.random.randint(start, end)
                    self.variable_dict[curr_variable] = generated

                # 생성된 결과물 최종 반환될 test_case에 더하기
                test_case = f'{test_case}{generated}'
                
                # derivateion이 완료된 variable 제거
                del derivation_queue[0]
                continue
            
            # <T_i> 형태
            if self.re.match(r'<[^_]*_[^_]*>', curr_variable):
                nonterminal , counter = curr_variable.split('_')
                # <T_N>
                if not counter[:-1].isdigit():
                    counter = self.variable_dict[counter[:-1]]
                    counter = f'{counter}>'
                    curr_variable = nonterminal + '_' + counter
                # <T_1> 등의 상황
                if curr_variable in self.derivation_dict:
                    counter = int(counter[:-1])
                else:
                    curr_variable = nonterminal + '_i>'
                    counter = int(counter[:-1])
            
            # derivate
            next_variable = self.random.choice(self.derivation_dict[curr_variable])
            curr_list = []
            
            for variable in next_variable.split(' '):
                # <T_i-1>이나 a_i가 생성되었을 때 counter가 필요함
                if self.re.match(r'<.*_i-1>', variable):
                    nonterminal = variable.split('_')[0]
                    variable = f'{nonterminal}_{counter-1}>'
                elif self.re.match(r'<.*_i>', variable):
                    nonterminal = variable.split('_')[0]
                    variable = f'{nonterminal}_{counter}>'    
                elif self.re.fullmatch(r'[^_<]*_.*', variable):
                    nonterminal = variable.split('_')[0]
                    variable = f'{nonterminal}_{counter}'
                curr_list.append(variable)
            
            # derivation이 진행된 이후 완료된 token 제거
            del derivation_queue[0]
            
            # DFS로 진행하기 위해 제거된 variable에서 생성된 variable들을 queue의 앞에 배치함
            curr_list.extend(derivation_queue)
            derivation_queue = curr_list
        
        return test_case
          
    def derivate_old(self, curr_variable):
        '''
        * 이전 grammer에 대한 derivate 함수
        '''
        return_str = ''
        if curr_variable == self.space_token:
            return ' '
        if curr_variable == self.new_line_token:
            return '\n'
        
        if self.re.match(r'<[^_]*_[^_]*>', curr_variable):
            variable, curr_count = curr_variable.split('_')
            if not curr_count[:-1].isdigit():
                curr_count = self.variable_dict[curr_count[:-1]]
                curr_variable = variable + '_i>'
            elif curr_variable in self.derivation_dict:
                curr_count = int(curr_count[:-1])
            else:
                curr_variable = variable + '_i>'
                curr_count = int(curr_count[:-1])
        
        next_token = self.derivation_dict[curr_variable]
        next_token = self.random.choice(next_token).split(' ')
                
        for token in next_token:
                        
            #case1 token is terminal
            if self.is_terminal(token):
                if token == 'ε': continue
                
                if self.re.match(r'\[[a-zA-Z]*\]', token):
                    variable = token[1:-1]
                    start, end = self.get_range(variable)
                    generated = self.random.randint(start, end)
                    self.variable_dict[variable] = generated
                    return_str = f'{return_str}{generated}' + self.derivate(token)
                    continue
                
                # a_i
                if self.re.match(r'[^_]*_[^_]*', token):
                    counted_token = token.split('_')[0] + f'_{curr_count}'
                    
                    if token not in self.variable_dict or token.split('_')[0] + '_1' in self.variable_dict[token]:
                            self.variable_dict[token] = {}
                    
                    if token in self.derivation_dict:
                        derivate = self.random.choice(self.derivation_dict[token.split('_')[0] + '_i'])
                        
                        if token in self.const_dict and self.const_dict[token]['permutation']:
                            while token in self.variable_dict and derivate in self.variable_dict[token].values():
                                derivate = self.random.choice(self.derivation_dict[token.split('_')[0] + '_i'])
                        
                        self.variable_dict[token][counted_token] = derivate
                        return_str = f'{return_str}{derivate}'
                        continue
                    
                    else:
                        start, end = self.get_range(variable=token, counter=curr_count)
                        generated = self.random.randint(start, end)
                        
                        if token in self.const_dict and self.const_dict[token]['permutation']:
                            while token in self.variable_dict and generated in self.variable_dict[token].values():
                                generated = self.random.randint(start, end)
                        self.variable_dict[token][counted_token] = generated
                        return_str = f'{return_str}{generated}'
                        continue
                
                if token in self.const_dict:
                    start, end = self.get_range(token)
                    generated = self.random.randint(start, end)
                    self.variable_dict[token] = generated
                    # print(self.variable_dict)
                    
                    # case: M -> <T_i>
                    if token in self.derivation_dict:
                        # print(token)
                        # self.variable_dict[token] = {}
                        return_str += self.derivate(self.random.choice(self.derivation_dict[token]))
                        continue
                    return_str = f'{return_str}{generated}'
                    continue
                
                if token in self.derivation_dict:
                    return_str += self.random.choice(self.derivation_dict[token])
                    continue
                # print('r', return_str)
                continue
                
            # case2 token's shape is <T_i>
            if self.re.match(r'<[^_]*_[^_]*>', token):
                variable, count = token.split('_')
                
                # case2-1 token is <T_N>
                if not self.re.fullmatch(r'[^_+-]*[+-]1>', count):
                    return_str += self.derivate(token)
                    
                    continue
                if count == 'i+1>':
                    return_str += self.derivate(f'{variable}_{curr_count + 1}>')
                else:
                    return_str += self.derivate(f'{variable}_{curr_count - 1}>')
                continue
            
            return_str += self.derivate(token)
        return return_str
    
    def get_addition(self, token):
        
        if self.re.match(r'[^+]+\+[0-9]+', token):
            token, addition = token.split('+')
            return token, int(addition)
        elif self.re.match(r'[^-]+-[0-9]+', token):
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
        elif '^' in start:
            num = 1
            if '*' in start:
                num, start = start.split('*')
            base, exp = start.split('^')
            res = int(num) * int(base) ** int(exp)
            start = res
        
        start = int(start) + addition
        # variable의 constraints가 a_i < a_i+1의 형태라면
        # a_i가 생성할 정수의 범위는 a_i-1 <= a_i <= end이다
        
        # constraint가 "start < variable"의 형태이면
        # "start+1 <= variable"과 같다
        start += 0 if include1 else 1
            
        end, addition = self.get_addition(curr_token_const['end'])

        if end in self.variable_dict:
            end = self.variable_dict[end]
        
        elif '^' in end:
            if '*' in end:
                num, end = end.split('*')
                base, exp = end.split('^')
                res = int(num) * int(base) ** int(exp)
            else:
                base, exp = end.split('^')
                res = int(base) ** int(exp)
            end = res
        
        end = int(end) + addition
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
                if self.re.match(r'.*_.*', target):
                    symbol = target.split('_')[0]
                    symbol += f'_{counter}'
                    derivate_range[range_index] = self.variable_dict[target][symbol]
                    derivate_range[range_index] += 0 if self.compare_dict[variable]['include'] else 1
                else:
                    derivate_range[range_index] = self.variable_dict[target]
                    derivate_range[range_index] += 0 if self.compare_dict[variable]['include'] else 1

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
        elif string_len.isnumeric():
            string_len = int(string_len)
        else:
            start, end = self.get_range(string_len)
            string_len = self.random.randint(start, end)
        
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
            return_str += self.random.choice(variable_list)

        return return_str
 
    def is_terminal(self, token):
       return not self.RE_NONTERMINAL.fullmatch(token)

    def make_derivate_dict(self, grammer: list):
        for token in grammer:
            left_hand, right_hand = token.split(self.derivate_token)
            self.derivation_dict[left_hand] = []
            for token in right_hand.split(' | '):
                self.derivation_dict[left_hand].append(token)

    def make_derivate_dict_old(self, grammer: list):
        for token in grammer:
            dict_key, dict_list = token.split(self.derivate_token)
            self.derivation_dict[dict_key] = []
            for target in dict_list.split(' | '):
                target = target.strip()
                if self.re.match(r'\[[^-]-[^-]\]', target):
                    start, end = target.split('-')
                    start = start[1:]
                    end = end[:-1]
                    start = ord(start)
                    end = ord(end)
                    self.derivation_dict[dict_key].extend([chr(x) for x in range(start, end+1)])
                else:
                    self.derivation_dict[dict_key].append(target)
    
    def make_constraints_dict(self, constraints:list):
        for const in constraints:
            if self.re.match(r'[^<]*<=?[^<]*<=?[^<]*', const):
                
                variables = self.re.split(r'[<>]=?', const)
                
                start, end = variables[0], variables[-1]
                
                del variables[0]
                del variables[-1]
                
                compare_symbols = self.re.findall(r'[<>]=?', const)
                
                for variable in variables:
                    # start, variable, end = self.re.split(r'<=?', const)
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
                
            elif self.re.fullmatch(r'[^<>]*[<>]=?[^<>]*', const):
                
                variable1, variable2 = self.re.split(r'[<>]=?', const)
                variable1, variable2 = variable1.strip(), variable2.strip()
                compare_symbol = self.re.findall(r'[<>]=?', const)[0]
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
                        
            elif self.re.fullmatch(r'[^=]*!=[^=]*', const):
                variable1, variable2 = const.split('!=')
                self.permutation_variable.append(variable1)
        
        '''
            # case 1: <= N <=
            if self.re.fullmatch(r'[^<]* <= [a-zA-Z]* <= [^<]*', const):
                start, variable, end = self.re.split(r'<=|<', const, 2)
                start = int(start)
                end = int(end)
                const_dict[variable] = {'start': start, 'end': end, 'include1': True, 'include2': True}
                
            # case 2: <= N < 
            elif self.re.fullmatch(r'[^<]* <= [a-zA-Z]* < [^<]*', const):
                start, variable, end = self.re.split(r'<=|<', const, 2)
                start = int(start)
                end = int(end)
                const_dict[variable] = {'start': start, 'end': end, 'include1': True, 'include2': False}
                
            # case 3: <  N <=
            elif self.re.fullmatch(r'[^<]* < [a-zA-Z]* <= [^<]*', const):
                start, variable, end = self.re.split(r'<=|<', const, 2)
                start = int(start)
                end = int(end)
                const_dict[variable] = {'start': start, 'end': end, 'include1': False, 'include2': True}
                
            # case 4: <  N < 
            elif self.re.fullmatch(r'[^<]* < [a-zA-Z]* < [^<]*', const):
                start, variable, end = self.re.split(r'<=|<', const, 2)
                start = int(start)
                end = int(end)
                const_dict[variable] = {'start': start, 'end': end, 'include1': False, 'include2': False}
                
            else:
                start, variable, end = self.re.split(r'<=|<', const, 2)
                if len(self.re.findall(r'<=|<', end)) == 1:
                    end = self.re.split(r'<=|<', end)[0]
        '''
        
if __name__ == '__main__':
    
    test_grammer = ['<S> -> [N]', '[N] -> <enter> <T_N>', '<T_i> -> M <enter> <T_i-1>', 'M -> <L_M>', '<L_i> -> C_i <space> <L_i-1>', '<T_0> -> ε', '<L_0> -> ε'] # 'C_i -> [a-z]', 
    test_const = ['1 <= N <= 50', '1 < M < 10', '1 <= C_i <= M', 'C_i != C_j']
    
    # test_grammer = ['<S> -> M <space> N']
    # test_const = ['1 <= M <= 16', 'M <= N <= 16', 'a <= b']

    generator = test_case_generator()
    generator.get_string('[a-z]{S}')
    # res = generator(test_grammer, test_const)
    # print(res)
    