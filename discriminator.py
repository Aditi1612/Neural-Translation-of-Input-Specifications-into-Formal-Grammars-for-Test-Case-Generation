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
        
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print(1)
        pass
    
    
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
    dis = discriminator()
    dis()
    
    ...