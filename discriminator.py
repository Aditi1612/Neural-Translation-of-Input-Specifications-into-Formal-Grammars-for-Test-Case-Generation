from typing import Any


class discriminator():
    
    import re
    
    def __init__(self) -> None:
        
        self.new_line_token = '<enter>'
        self.space_token = '<space>'
        self.start_token = '<S>'
        self.derivate_token = '->'
        self.RE_NONTERMINAL = self.re.compile(r'(<[^<> ]*>)')
        self.const_dict = {}
        self.variable_dict = {}
        self.derivation_dict = {}
        self.shape_dict = {}
        
        import sys
        sys.setrecursionlimit(2000000000)
        
        pass
    
    

    def __call__(self, grammer: list, constraints: list, test_case: str) -> bool:
        self.__init__()
        self.make_derivate_dict(grammer)
        self.make_constraints_dict(constraints)
        self.predict()
        
        line_num = 0
        # first_line = ''
        
        first_line = self.derivation_dict[self.start_token][0]
        
        if self.start_token in self.shape_dict:
            # test_case = test_case.split()
            # shape = self.re.split(f' {self.space_token} | {self.new_line_token} ', first_line)
            
            # for variable, value in zip(shape, test_case):
            #     if self.re.fullmatch(r'\[.*\]', variable):
            #         variable = variable[1:-1]
            #     self.variable_dict[variable] = value
            #     line_num = int(value)
            
            # # print(self.variable_dict)
            self.parsing(test_case)
            return
        
        
        while self.new_line_token not in first_line :
            tmp = ''
            
            for token in first_line.split(' '):
                if self.is_terminal(token):
                    tmp += token + ' '
                elif self.re.fullmatch(r'\[[^\[]*\]', token):
                    tmp += token + ' ' + self.derivation_dict[token][0] + ' '
                else:
                    tmp += token + ' '
            
            first_line = tmp
        
        first_line, next_line =  first_line.split(f' {self.new_line_token} ')
        first_line = first_line.strip()
        next_line = next_line.strip()
        print('n', next_line)
        
        test_case_first_line = test_case.split('\n')[0]
        first_line = first_line.split(f' {self.space_token} ')

        for variable, value in zip(first_line, test_case_first_line.split()):
            print('v', variable)
            if self.re.fullmatch(r'\[.*\]', variable):
                variable = variable[1:-1]
            self.variable_dict[variable] = value
        
        print(self.variable_dict)
        
        self.parsing('\n'.join(test_case.split('\n')[1:]), '[N]', next_line)
        
        
    pass
    
    def parsing(self, test_case, line_num_token=None, start_token=None) -> bool:
        if line_num_token == None:
            curr = self.derivation_dict[self.start_token][0]
            curr = curr.replace(f' {self.new_line_token} ', '\n').replace(f' {self.space_token} ', ' ')
            curr = curr.split()
            curr_line = test_case.split()
            
            for variable, value in zip(curr, curr_line):
                self.variable_dict[variable] = value
            
            print(self.variable_dict)
            
            
            return True
        
        line_num = int(self.variable_dict[line_num_token[1:-1]])
        if self.re.fullmatch(r'<[^[_]*_[^[_]*>', start_token):
            start_token = start_token.split('_')[0] + '_i>'
        
        
        for _ in range(line_num):
            curr = self.derivation_dict[start_token][0]
            curr_line = test_case.split('\n')[0].strip()
            # test_case = '\n'.join(test_case.split('\n')[1:])
            for token in curr.split(' '):
                if token == self.new_line_token:
                    curr_line = test_case.split('\n')[0].strip()
                    test_case = '\n'.join(test_case.split('\n')[1:])
                    continue
                if token in self.shape_dict:
                    shape = self.shape_dict[token]
                    if shape == 'string':
                        self.variable_dict[token] = len(curr_line)
                        ...
                    elif shape == 'space':
                        self.variable_dict[token] = len(curr_line.split(' '))
                        ...
                    else:
                        ...
                    print('-------------')
                    print(token)
                    print(self.variable_dict)
                    print(curr_line)
                    print('-------------')

        return False
        

    def predict(self):
        
        # case 1
        if len(self.derivation_dict.keys()) == 1:
            self.shape_dict[self.start_token] = 'single_line'
            return

        for target in self.derivation_dict:
            next = self.derivation_dict[target][0]
            
            if self.is_temp_variable(target):
                next = next.split('_')[0] + '_i>'
                next = self.derivation_dict[next][0]
                self.shape_dict[target] = self.get_shape(next)
                
            elif self.re.fullmatch(r'\[[^\[]*\]', target):
                next = next.split(' ')[-1]
                next = next.split('_')[0] + '_i>'
                next = self.derivation_dict[next][0]
                self.shape_dict[target] = self.get_shape(next)
        
        # return ''
        
    
    def get_shape(self, token:str) -> str:
        if self.space_token in token:
            return 'space'
        elif self.new_line_token in token:
            return 'newline'
        else: return 'string'
        
        
    
    
    def is_temp_variable(self, token):
        if self.re.fullmatch(r'<[^<]*>', token): return False
        if self.re.fullmatch(r'\[[^\[]*\]', token): return False
        if token in self.derivation_dict:
            next_tokens = self.derivation_dict[token]
            if len(next_tokens) == 1 and self.re.fullmatch(r'<[^<]*>', next_tokens[0]): return True
        return False
    
    def is_terminal(self, token):
        if self.re.fullmatch(r'<[^<]*>', token): return False
        if self.re.fullmatch(r'\[[^\[]*\]', token): return False
        if token in self.derivation_dict:
            next_tokens = self.derivation_dict[token]
            for next in next_tokens:
                if self.re.match(r'<[^<]*>', next): return False
        return True
    
    
    def make_derivate_dict(self, grammer: list):
        # grammer_lst = grammer.split(self.sep_token)
        for token in grammer:
            dict_key, dict_list = token.split(self.derivate_token)
            dict_key = dict_key.strip()
            self.derivation_dict[dict_key] = []
            for target in dict_list.split('|'):
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
        # print(self.derivation_dict)
    
    def make_constraints_dict(self, constraints:list):
        for const in constraints:
            if self.re.fullmatch(r'[^<]* <=? [^<]* <=? [^<]*', const):
                start, variable, end = self.re.split(r'<=?', const)
                include_list = self.re.findall(r'<=?', const)
                self.const_dict[variable.strip()] = {
                                        'start': start.strip(), 'end': end.strip(),
                                        'include1': include_list[0] == '<=', 
                                        'include2': include_list[1] == '<=',
                                        'permutation': False, 
                                        'type': 'range'
                                        }
                
            elif self.re.fullmatch(r'[^<]* <=? [^<]*', const):
                    variable1, variable2 = self.re.split(r'<=?', const)
                    variable1, variable2 = variable1.strip(), variable2.strip()
                    self.const_dict[variable1] = {
                                        'target': variable2,
                                        'include': self.re.findall(r'<=?', const)[0] == '<=',
                                        'permutation': False, 
                                        'type': 'compare'
                                        }
            elif self.re.fullmatch(r'[^=]* != [^=]*', const):
                variable1, variable2 = const.split(' != ')
                # self.const_dict[variable1] = {'type': 'permutation'}
                self.const_dict[variable1]['permutation'] = True
                ...
        # print(self.const_dict)
 
    
if __name__ == '__main__':
    test_grammer = ['<S> -> [N]', '[N] -> <enter> <T_N>', '<T_i> -> M <enter> <T_i-1>', 'M -> <L_M>', '<L_i> -> C_i <space> <L_i-1>', '<T_0> -> ε', '<L_0> -> ε']
    test_const = ['1 <= N <= 50', '1 < M < 10', '1 <= C_i <= M', 'C_i != C_j']
    test_case = '6\n23 16 11 21 22 2 13 9 10 19 24 18 14 12 3 4 7 8 15 5 25 6 17 20 1 \n6 2 1 5 4 3 \n6 5 11 3 29 2 26 17 14 13 9 12 18 16 15 25 21 30 4 27 7 23 22 20 10 24 8 1 28 19 \n20 15 12 9 10 8 21 19 13 14 5 7 6 1 16 2 11 17 4 3 18 \n8 33 28 32 3 35 34 29 13 25 24 12 16 5 4 30 6 7 11 19 22 9 20 17 26 31 18 21 10 1 23 15 14 27 2 \n28 31 17 11 22 30 12 23 14 18 26 15 1 19 21 16 8 5 29 20 10 3 7 13 2 6 9 24 4 27 25 \n'
    # test_grammer = ['<S> -> M <space> N']
    # test_const = ['1 <= M <= 16', 'M <= N <= 16', 'a <= b']

    discriminator = discriminator()
    res = discriminator(test_grammer, test_const, test_case)
    
    print(res)
    
    ...