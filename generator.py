
class test_case_generator():
    import re
    import random
    
    def __init__(self) -> None:
        import sys
        self.sep_token = '\t'
        self.new_line_token = '<enter>'
        self.space_token = '<space>'
        self.start_token = '<S>'
        self.derivate_token = '->'
        self.RE_NONTERMINAL = self.re.compile(r'(<[^<> ]*>)')
        self.const_dict = {}
        self.variable_dict = {}
        self.derivation_dict = {}
        
        sys.setrecursionlimit(2000000000)


        
    def generate(self, grammer: str, constraints: list):
        self.make_derivate_dict(grammer)
        self.make_constraints_dict(constraints)
        result = self.derivate(self.start_token)
    
        # print(result)
        return result
          
    def derivate(self, curr_variable):
        return_str = ''
        # print(curr_variable)
        if curr_variable == self.space_token:
            return ' '
        if curr_variable == self.new_line_token:
            return '\n'
        
        # print(curr_variable)
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
                    derivate = self.random.choice(self.derivation_dict[token.split('_')[0] + '_i'])
                    self.variable_dict[token] = derivate
                    return_str = f'{return_str}{derivate}'
                    continue
                
                if token in self.const_dict and self.const_dict[token]['type'] == 'range':
                    start, end = self.get_range(token)
                    generated = self.random.randint(start, end)
                    self.variable_dict[token] = generated
                    if token in self.derivation_dict:
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
                # case2-1 token is T_N
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
    
    def get_range(self, variable):
        curr_token_const = self.const_dict[variable]
        start = curr_token_const['start']
        end = curr_token_const['end']
        include1 = curr_token_const['include1']
        include2 = curr_token_const['include2']
        
        return (start + 0 if include1 else 1, end + 1 if include2 else 0)
    
    def is_terminal(self, token):
       return not self.RE_NONTERMINAL.fullmatch(token)


    def make_derivate_dict(self, grammer: str):
        import string
        grammer_lst = grammer.split(self.sep_token)
        for token in grammer_lst:
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
    
    
    def make_constraints_dict(self, constraints:list):
        for const in constraints:
            if self.re.fullmatch(r'[^<]* <=? [^<]* <=? [^<]*', const):
                start, variable, end = self.re.split(r'<=|<', const)
                include_list = self.re.findall(r'<=|<', const)
                self.const_dict[variable.strip()] = {
                                        'start': int(start), 'end': int(end),
                                        'include1': include_list[0] == '<=', 
                                        'include2': include_list[1] == '<=',
                                        'type': 'range'
                                        }
                
            else:
                if self.re.fullmatch(r'[^<]* <=? [^<]*', const):
                    variable1, variable2 = self.re.split(r'<=|<', const)
                    self.const_dict[variable1] = {
                                        'target': variable2,
                                        'include': self.re.findall(r'<=?', const)[0] == '<=',
                                        'type': 'compare'
                                        }
                
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
    
    test_grammer = '\t'.join(['<S> -> [N]', '[N] -> <enter> <T_N>', '<T_i> -> C_i <T_i-1>', 'C_i -> [a-z] | [A-D] | [1-5]', '<T_1> -> ε'])
    test_const = ['1 <= N <= 50', '1 < M < 10']

    generator = test_case_generator()
    res = generated_test_case = generator.generate(test_grammer, test_const)
    print(res)
    