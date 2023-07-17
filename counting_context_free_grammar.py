from enum import Enum
import logging
import random
import re
import math

from xeger import Xeger

from constraint import Comparison, get_constraints_and_comparisons

# MAX_UPPER_BOUND = math.inf
MAX_UPPER_BOUND = 50

MAX_ITER = 1000

SEP_TOKEN = '\t'
NEW_LINE_TOKEN = '<n>'
SPACE_TOKEN = '<s>'
START_TOKEN = '<S>'
DERIVATE_TOKEN = '->'
BLANK_TOKEN = 'ε'


class TokenType(Enum):
    TERMINAL = 0
    NONTERMINAL = 1  # <A>, <A_i>, <A_3>
    VARIABLE = 2  # [N], X, X_i, X_2, X_3, X_N


class VariableType(Enum):
    DEFAULT = 0  # X
    UNINDEXED = 1  # X_i
    VAR_INDEXED = 2  # X_N
    INDEXED = 3  # X_3
    COUNTER = 4  # [N]


class NonterminalType(Enum):
    DEFAULT = 0  # <A>
    UNINDEXED = 1  # <A_i>
    VAR_INDEXED = 2  # <A_N>
    INDEXED = 3  # <A_3>


class CountingContextFreeGrammar():

    def __init__(self, rule_strings, constraint_strings):

        self.productions = {}
        for rule_string in rule_strings:
            lhs, rhss = rule_string.split(DERIVATE_TOKEN)
            self.productions[lhs] = []
            for rhs in rhss.split('|'):
                self.productions[lhs].append(rhs.split(' '))

        self.placeholders = set()

        self.constraints, self.comparisons = (
            get_constraints_and_comparisons(constraint_strings))

        # for variable in self.constraints:
            # _, index = _split_variable(variable)
            # if index is not None:
                # self.placeholders.add(index)

        for nonterminal in self.productions:
            _, index = _split_nonterminal(nonterminal)
            if index is not None and not index.isdigit():
                self.placeholders.add(index)

        for placeholder in self.placeholders:
            if placeholder in self.constraints:
                del self.constraints[placeholder]
            if placeholder in self.comparisons:
                del self.comparisons[placeholder]

    def generate(self):
        string = ''

        derivation_queue = [START_TOKEN]
        assignment = {}
        while derivation_queue:

            logging.debug(derivation_queue)
            logging.debug(assignment)
            logging.debug(f'\n{string}')

            token = derivation_queue.pop()
            token_type = self._get_token_type(token)

            production = []

            if token_type == TokenType.TERMINAL:
                string += self._sample_terminal(token, assignment)

            elif token_type == TokenType.NONTERMINAL:
                production = self._derivate_nonterminal(token, assignment)

            elif token_type == TokenType.VARIABLE:
                string += str(self._assign(token, assignment))
                if token in self.productions:
                    production = random.choice(self.productions[token])

            derivation_queue += production[::-1]

        return string

    def _substitute(self, token, old, new):
        token_type = self._get_token_type(token)

        if token_type == TokenType.NONTERMINAL:
            nonterminal_type = self._get_nonterminal_type(token)
            if nonterminal_type in (
                    {NonterminalType.UNINDEXED, NonterminalType.VAR_INDEXED}):
                return _substitute_nonterminal(token, old, new)

        elif token_type == TokenType.VARIABLE:
            variable_type = self._get_variable_type(token)
            if variable_type in (
                    {VariableType.UNINDEXED, VariableType.VAR_INDEXED}):
                return _substitute_variable(token, old, new)

        return token

    def _substitute_production(self, production, placeholder, index):
        return [
            self._substitute(token, placeholder, int(index))
            for token in production
        ]

    def _derivate_nonterminal(self, nonterminal, assignment):
        nonterminal_type = self._get_nonterminal_type(nonterminal)

        if nonterminal_type == NonterminalType.DEFAULT:
            return random.choice(self.productions[nonterminal])
        elif nonterminal_type == NonterminalType.UNINDEXED:
            raise ValueError(
                f"Invalid derivation of nonterminal {nonterminal}")

        elif nonterminal_type == NonterminalType.VAR_INDEXED:
            _, variable = _split_nonterminal(nonterminal)
            return [self._substitute(
                nonterminal, variable, assignment[variable])]

        elif nonterminal_type == NonterminalType.INDEXED:
            if nonterminal in self.productions:  # E.g., <T_0> -> <L_N>
                return random.choice(self.productions[nonterminal])

            frag, index = _split_nonterminal(nonterminal)

            placeholder = None
            production = None
            for placeholder in self.placeholders:
                unindexed = f"<{frag}_{placeholder}>"
                if unindexed in self.productions:
                    production = random.choice(self.productions[unindexed])
                    break

            if production is None:
                raise ValueError(f"Cannot find production of {nonterminal}")

            return self._substitute_production(production, placeholder, index)

    def _assign(self, variable, assignment):
        variable_type = self._get_variable_type(variable)

        if variable_type == VariableType.DEFAULT:
            return self._assign_constrained(variable, assignment)

        elif variable_type == VariableType.UNINDEXED:
            raise RuntimeError(f"Invalid variable: {variable}")

        elif variable_type == VariableType.INDEXED:
            frag, index = _split_variable(variable)
            for placeholder in self.placeholders:
                # XXX: Some constraints result in double assignment
                unindexed_variable = f"{frag}_{placeholder}"
                return self._assign_constrained(
                    unindexed_variable, assignment, int(index))
            raise ValueError(f"Unbounded variable: {variable}")

        elif variable_type == VariableType.COUNTER:
            return self._assign_constrained(variable[1:-1], assignment)

    def _get_or_assign(self, variable, assignment):
        if variable in assignment:
            return assignment[variable]
        else:
            return self._assign(variable, assignment)

    def _get_token_type(self, token):

        if token in {NEW_LINE_TOKEN, SPACE_TOKEN, BLANK_TOKEN}:
            return TokenType.TERMINAL

        if token in self.productions:
            return TokenType.NONTERMINAL

        if token[0] == '<' and token[-1] == '>':
            return TokenType.NONTERMINAL
        elif token[0] == '[' and token[-1] == ']':
            if '_' not in token[1:-1] and token[1:-1] in self.constraints:
                return TokenType.VARIABLE
        elif token in self.constraints:
            return TokenType.VARIABLE
        elif '_' in token:
            frag, _ = _split_variable(token)
            for placeholder in self.placeholders:
                if f"{frag}_{placeholder}" in self.constraints:
                    return TokenType.VARIABLE
        return TokenType.TERMINAL

    def _assign_constrained(self, variable, assignment, index=None):
        # assign variable whose form is in constraints (e.g., X_i, X)
        placeholder = None
        if index is not None:
            _, placeholder = _split_variable(variable)

        constraint = self.constraints.get(variable)
        comparison = self.comparisons.get(variable, Comparison())

        upper_bound = constraint.upper_bound
        lower_bound = constraint.lower_bound

        # XXX: instead of these, update constraints
        for lower_variable, inclusive in comparison.lower_bounds:
            new_lower_bound = assignment.get(
                self._substitute(lower_variable, placeholder, index),
                self.constraints[lower_variable].lower_bound)
            if not inclusive:
                new_lower_bound += 1
            lower_bound = max(new_lower_bound, lower_bound)

        for upper_variable, inclusive in comparison.upper_bounds:
            new_upper_bound = assignment.get(
                self._substitute(upper_variable, placeholder, index),
                self.constraints[upper_variable].upper_bound)
            if not inclusive:
                new_upper_bound -= 1
            upper_bound = min(new_upper_bound, upper_bound)

        # XXX: max iteration
        comparison_inequal = {assignment.get(e) for e in comparison.inequal}
        inequal = constraint.inequal | comparison_inequal
        indexed_variable = self._substitute(variable, placeholder, index)
        for _ in range(MAX_ITER):
            assignment[indexed_variable] = random.randint(
                lower_bound, min(upper_bound, MAX_UPPER_BOUND))
            if assignment[indexed_variable] not in inequal:
                break

        return assignment[indexed_variable]

    def _get_nonterminal_type(self, nonterminal) -> NonterminalType:
        if '_' in nonterminal:
            _, index = _split_nonterminal(nonterminal)
            if index.isdigit():
                return NonterminalType.INDEXED
            elif index in self.constraints:
                return NonterminalType.VAR_INDEXED
            else:
                return NonterminalType.UNINDEXED
        else:
            return NonterminalType.DEFAULT

    def _get_variable_type(self, variable) -> VariableType:
        if variable[0] == '[' and variable[-1] == ']':
            return VariableType.COUNTER
        elif '_' in variable:
            _, index = _split_variable(variable)
            if index.isdigit():
                return VariableType.INDEXED
            elif index in self.constraints:
                return VariableType.VAR_INDEXED
            else:
                return VariableType.UNINDEXED
        else:
            return VariableType.DEFAULT

    def _sample_terminal(self, terminal, assignment):
        if terminal == NEW_LINE_TOKEN:
            return '\n'
        elif terminal == SPACE_TOKEN:
            return ' '
        elif terminal == BLANK_TOKEN:
            return ''
        else:
            try:
                replaced = _RE_COUNTER_VARIABLE_OP.sub(
                    lambda e: f'{self._get_or_assign(e.group(0), assignment)}',
                    terminal)
                generated = _x.xeger(replaced)
                return generated
            except re.error:
                return terminal

    def __str__(self):
        return "\n".join([
            "Productions:",
            str(self.productions),
            "Constraints:",
            str({k: str(v) for k, v in self.constraints.items()}),
            "Comparisons:",
            str({k: str(v) for k, v in self.comparisons.items()}),
            "Placeholders:",
            str(self.placeholders)
        ])


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    test_grammar = [
        '<S>->N <s> [M]',
        '[M]-><s> S <n> <T_M>',
        '<T_i>->U_i <s> V_i <s> A_i <s> B_i <n> <T_i-1>',
        '<T_0>-><L_N>',
        '<L_i>->C_i <s> D_i <n> <L_i-1>',
        '<L_0>->ε',
    ]
    test_const = [
        "2<=N<=50",
        "N<M<=100",
        "0<=S<=1000000000",
        "1<=A_i<=50",
        "1<=B_i<=1000000000",
        "1<=C_i<=1000000000",
        "1<=D_i<=1000000000",
        "1<=U_i<N",
        "U_i<V_i<=N",
    ]

    ccfg = CountingContextFreeGrammar(test_grammar, test_const)
    print(ccfg)
    print(ccfg.generate())


_RE_NONTERMINAL = re.compile(r'<\w+?>')
# Only support {N}, where N is variable
_RE_COUNTER_VARIABLE_OP = re.compile(r'(?<=\{)[A-Za-z_]+?(?=\})')

_x = Xeger(limit=0)


def _substitute_nonterminal(nonterminal, placeholder, index: int):
    if placeholder is None:
        return nonterminal

    # XXX: Hard-coded
    frag, token_placeholder = _split_nonterminal(nonterminal)
    if token_placeholder == placeholder:
        return f"<{frag}_{index}>"
    if token_placeholder == f"{placeholder}-1":
        return f"<{frag}_{index-1}>"
    return nonterminal


def _substitute_variable(variable, placeholder, index: int):
    # XXX: Hard-coded
    frag, token_placeholder = _split_variable(variable)
    if token_placeholder == placeholder:
        return f"{frag}_{index}"
    if token_placeholder == f"{placeholder}-1":
        return f"{frag}_{index-1}"
    return variable


def _split_variable(variable):
    tmp = tuple(variable.rsplit('_', 1))
    if len(tmp) != 2:
        return variable, None
    return tuple(tmp)


def _split_nonterminal(nonterminal):
    tmp = tuple(nonterminal[1:-1].rsplit('_', 1))
    if len(tmp) != 2:
        return nonterminal, None
    return tuple(tmp)


def _is_nonterminal_token(token):
    return _RE_NONTERMINAL.fullmatch(token)
