from enum import Enum
from typing import (Optional, Callable, )
import logging
import random
import re

from constraint import get_constraints_and_comparisons
from constraint import parse_comparand
from constraint import ExtInt

Assignment = dict[str, int]

MAX_ITER = 100
TESTMODE_VARIABLE_UPPER_BOUND = 50
TESTMODE_MAXIMUM_TERMINAL_LEN = 50

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

    def __init__(
        self,
        production_strings: list[str],
        constraint_strings: list[str],
        testmode: bool = False,
    ):
        """Counting context-free grammar.

        Args:
            production_strings: A list of strings describing productions.,
            constraint_strings: A list of strings describing constraints.
        """

        self.testmode = testmode

        # Parse productions
        self.productions = {}
        for rule_string in production_strings:
            lhs, rhss = rule_string.split(DERIVATE_TOKEN)
            self.productions[lhs] = []
            for rhs in rhss.split('|'):
                self.productions[lhs].append(rhs.split(' '))

        # Parse constraints and comparisons
        self.constraints, self.comparisons = (
            get_constraints_and_comparisons(constraint_strings))

        # Initialize placeholders from constraints and comparisons
        self.placeholders = set()
        for token in self.productions:
            if token[0] == '<' and token[-1] == '>':
                _, subscript = _split_nonterminal(token)
            else:
                _, subscript = _split_variable(token)
            if subscript is not None and not subscript.isdigit():
                self.placeholders.add(subscript)

        # Remove constraints and comparisons of placeholders
        for placeholder in self.placeholders:
            if placeholder in self.constraints:
                del self.constraints[placeholder]
            if placeholder in self.comparisons:
                del self.comparisons[placeholder]

    def generate(self) -> str:
        """Randomly generate a string of the counting context-free grammar.

        Returns:
            A generated string in the counting context-free grammar.
        """

        string = ''

        derivation_queue = [START_TOKEN]
        assignment = {}
        while derivation_queue:

            logging.debug(derivation_queue)
            logging.debug(assignment)
            logging.debug(f'\n{string}')

            token = derivation_queue.pop()
            production = None
            token_type = self._get_token_type(token)

            logging.debug(f'Current token: {token}')
            logging.debug(f'Token type: {token_type}')

            if token_type == TokenType.TERMINAL:
                string += self._sample_terminal(token, assignment)

            elif token_type == TokenType.NONTERMINAL:
                production = self._derivate_nonterminal(token, assignment)

            elif token_type == TokenType.VARIABLE:
                variable_type = self._get_variable_type(token)

                constraint_form_variables, index = (
                    self._to_constraint_form(token))

                _production, value = self._derivate_variable(token, assignment)

                assignment_form_variable = token
                if variable_type is VariableType.COUNTER:
                    assignment_form_variable = token[1:-1]

                if variable_type is VariableType.COUNTER:
                    production = _production
                elif _production is not None:
                    if value is not None:
                        raise RuntimeError(
                            f"A variable {token} has both"
                            + "constraint and production.")
                    string += " ".join(_production)

                if value is not None:
                    assignment[assignment_form_variable] = value
                    string += str(value)

            if production is not None:
                derivation_queue += production[::-1]

        return string

    def _substitute(
        self, token: str, subscript: Optional[str], index: int
    ) -> str:
        """Substitute ``subscript`` of ``token`` with ``index``.

        Substitute ``subscript`` of ``token`` with ``index``.
        If the subscript of ``token`` is not ``subscript``, return ``token``

        Args:
            token: a nonterminal or variable token
            subscript: a subscript to be replaced
            index: an index to replace with

        Returns:
            A ``index``-subscripted ``token`` if ``token`` has
            ``subscript``. Otherwise, return ``token``.
        """
        if subscript is None:
            return token

        token_type = self._get_token_type(token)

        if token_type == TokenType.NONTERMINAL:
            nonterminal_type = self._get_nonterminal_type(token)
            if nonterminal_type in (
                    {NonterminalType.UNINDEXED, NonterminalType.VAR_INDEXED}):
                return _substitute_nonterminal(token, subscript, index)

        elif token_type == TokenType.VARIABLE:
            variable_type = self._get_variable_type(token)
            if variable_type in (
                    {VariableType.UNINDEXED, VariableType.VAR_INDEXED}):
                return _substitute_variable(token, subscript, index)

        return token

    def _substitute_production(
        self, production: list[str], placeholder: str, index: int
    ) -> str:

        assert placeholder in self.placeholders

        return [
            self._substitute(token, placeholder, int(index))
            for token in production
        ]

    def _derivate_nonterminal(
        self, nonterminal: str, assignment: Assignment
    ) -> list[str]:
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

    def _derivate_variable(
        self, variable: str, assignment: Assignment
    ) -> tuple[Optional[list[str]], Optional[int]]:
        """Randomly select production of variable and generate value

        Args:
            variable: A variable
            assignment: An assignment

        Returns:
            A tuple consists of production and value.

        Raises:
            RuntimeError: Constraint or production is ambiguous.
        """
        variable_type = self._get_variable_type(variable)

        constraint_form_variables, index = (
            self._to_constraint_form(variable))

        production_form_variables = constraint_form_variables
        if variable_type == VariableType.COUNTER:
            production_form_variables = [variable]

        # Sample value of the variable
        values = []
        for constraint_form_variable in constraint_form_variables:
            value = self._sample_variable(
                constraint_form_variable, assignment, index)
            values.append(value)
        values = list(filter(lambda e: e is not None, values))
        if len(values) > 1:
            raise RuntimeError(
                f"Constraint is ambiguous for {variable}.")

        value = None
        if len(values) == 1:
            value = values[0]

        # Find production of the variable
        productions = []
        for production_form_variable in production_form_variables:
            if production_form_variable in self.productions:
                production = random.choice(
                    self.productions[production_form_variable])
                productions.append(production)
        productions = list(
            filter(lambda e: e is not None, productions))
        if len(productions) > 1:
            raise RuntimeError(
                f"Production is ambiguous for {variable}.")

        production = None
        if len(productions) == 1:
            production = productions[0]

        return production, value

    def _to_constraint_form(self, variable: str) -> tuple[list[str], int]:
        """ Format variable to find the variable in constraints.

        Args:
            variable: A variable

        Returns:
            A tuple consists of the list of formatted variable and the index
        Raises:
            ValueError: The given variable is in unindexed or variable-indexed
            form.
        """
        variable_type = self._get_variable_type(variable)

        index = None
        constraint_form_variables = []
        if variable_type == VariableType.DEFAULT:
            constraint_form_variables.append(variable)
        elif variable_type == VariableType.INDEXED:
            frag, _index = _split_variable(variable)
            if _index is not None:
                index = int(_index)
            for placeholder in self.placeholders:
                constraint_form_variables.append(f"{frag}_{placeholder}")
        elif variable_type == VariableType.COUNTER:
            constraint_form_variables = [variable[1:-1]]
        else:
            raise ValueError(f"Invalid variable: {variable}")

        return constraint_form_variables, index

    def _get_variable_bound(
        self,
        variable: str,
        assignment: Assignment,
        index: Optional[int],
        reverse: bool = False
    ) -> int:

        comparison = self.comparisons[variable]

        constraint_bound = None
        bound_variables = None
        tightest = None
        get_target_bound: Callable[[str], ExtInt]
        tighter_than: Callable[[ExtInt, ExtInt], bool]
        tighten: Callable[[ExtInt], ExtInt]
        if not reverse:
            constraint_bound = self.constraints[variable].lower_bound
            bound_variables = comparison.lower_bounds
            tightest = max
            def get_target_bound(e): return self.constraints[e].upper_bound
            def tighter_than(a, b): return a >= b
            def tighten(e): return e + 1
        else:
            constraint_bound = self.constraints[variable].upper_bound
            bound_variables = comparison.upper_bounds
            tightest = min
            def get_target_bound(e): return self.constraints[e].lower_bound
            def tighter_than(a, b): return a <= b
            def tighten(e): return e - 1

        unassigned_bound_variables = []
        bounds = [constraint_bound]

        placeholder = None
        if index is not None:
            _, placeholder = _split_variable(variable)

        for bound_variable, inclusive in bound_variables:
            indexed_bound_variable = (
                self._substitute(bound_variable, placeholder, index))

            if indexed_bound_variable in assignment:
                bound = assignment[indexed_bound_variable]
                if not inclusive:
                    bound = tighten(bound)
                bounds.append(bound)
            else:
                unassigned_bound_variables.append(bound_variable)

        bound = tightest(bounds)

        # If there exists unassigned variables whose bounds are
        # intersect with the current variable, we have to make a room
        # for them.
        # TODO
        for bound_variable in unassigned_bound_variables:
            target_bound = get_target_bound(bound_variable)
            if tighter_than(target_bound, bound):
                pass
                # bound = tighten(bound)
        return bound

    def _sample_variable(
        self,
        variable: str,
        assignment: Assignment,
        index: Optional[int] = None
    ) -> Optional[int]:
        """Sample a value of variables whose form is in constraints

        Args:
            variable: A variable occurs in constraints.,
            index: Index of variable, which is omitted in ``variable``.

        Returns:
            Return an integer or ``None`` if the variable is not in
            constraints.
        """
        if variable not in self.constraints:
            return None

        constraint = self.constraints[variable]

        lower_bound = self._get_variable_bound(variable, assignment, index)
        upper_bound = self._get_variable_bound(
            variable, assignment, index, reverse=True)

        # print(variable, lower_bound, upper_bound)

        if self.testmode:
            upper_bound = min(upper_bound, TESTMODE_VARIABLE_UPPER_BOUND)
            upper_bound = max(lower_bound, upper_bound)

        comparison = self.comparisons[variable]
        comparison_inequal = {assignment.get(e) for e in comparison.inequal}
        inequal = constraint.inequal | comparison_inequal

        # XXX: It depends on max iteration
        for _ in range(MAX_ITER):
            value = random.randint(lower_bound, upper_bound)
            if value not in inequal:
                return value

        raise RuntimeError(
            f"Fail to sample variable: {variable}\n Assignment: {assignment}")

    def _get_token_type(self, token: str) -> TokenType:
        if token in {NEW_LINE_TOKEN, SPACE_TOKEN, BLANK_TOKEN}:
            return TokenType.TERMINAL
        elif _RE_REGEX_TERMINAL.fullmatch(token):
            return TokenType.TERMINAL
        elif token[0] == '<' and token[-1] == '>':
            return TokenType.NONTERMINAL
        else:
            return TokenType.VARIABLE

    def _get_nonterminal_type(self, nonterminal: str) -> NonterminalType:
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

    def _get_variable_type(self, variable: str) -> VariableType:
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

    def _parse_counter_operator(
        self,
        counter_operator: str,
        assignment: Assignment
    ) -> int:
        if _RE_NUMBER_CBE.fullmatch(counter_operator):
            value = parse_comparand(counter_operator)
        elif counter_operator in assignment:
            value = assignment[counter_operator]
        elif counter_operator in self.constraints:
            value = self._sample_variable(counter_operator, assignment)
            assignment[counter_operator] = value
        elif counter_operator.isdigit():
            value = int(counter_operator)
        else:
            raise ValueError(
                f'Counter operater parse failed: {counter_operator}')
        return value

    def _sample_terminal(
        self, terminal: str, assignment: Assignment
    ) -> str:
        if terminal == NEW_LINE_TOKEN:
            return '\n'
        elif terminal == SPACE_TOKEN:
            return ' '
        elif terminal == BLANK_TOKEN:
            return ''

        match = _RE_REGEX_TERMINAL.fullmatch(terminal)
        if not match:
            return terminal

        # Parse regex operands
        counter_operands = list(_parse_counter_oparands(match.group(1)))
        counter_operators = match.group(2).split(',')

        if len(counter_operators) > 2:
            raise ValueError(f'Too many counter operators: {terminal}')

        values = []
        for counter_operator in counter_operators:
            value = self._parse_counter_operator(counter_operator, assignment)
            values.append(value)

        start = 0
        end = 0
        if len(values) == 1:
            start = values[0]
            end = values[0]
        elif len(values) == 2:
            start, end = tuple(values)

        if self.testmode:
            end = min(start, TESTMODE_MAXIMUM_TERMINAL_LEN)
            end = max(start, end)

        terminal_len = random.choice(range(start, end+1))
        terminal_string = ""
        for _ in range(terminal_len):
            terminal_string += random.choice(counter_operands)
        return terminal_string

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


_RE_REGEX_TERMINAL = re.compile(r'(.+?)\{([\w\-\*\^,]*)\}')
_RE_NUMBER_CBE = re.compile(r'(?:(-?\d+)\*)?(-?\d+)(?:\^(-?\d+))?')


def _substitute_nonterminal(
    nonterminal: str, placeholder: Optional[str], index: int
) -> str:
    if placeholder is None:
        return nonterminal

    # XXX: Hard-coded
    frag, token_placeholder = _split_nonterminal(nonterminal)
    if token_placeholder == placeholder:
        return f"<{frag}_{index}>"
    elif token_placeholder == f"{placeholder}-1":
        return f"<{frag}_{index-1}>"
    return nonterminal


def _substitute_variable(
    variable: str, placeholder: Optional[str], index: int
) -> str:
    if placeholder is None:
        return variable

    # XXX: Hard-coded
    frag, token_placeholder = _split_variable(variable)
    if token_placeholder == placeholder:
        return f"{frag}_{index}"
    elif token_placeholder == f"{placeholder}-1":
        return f"{frag}_{index-1}"
    return variable


def _split_token(token: str) -> tuple[str, Optional[str]]:
    if token[0] == '<' and token[-1] == '>':
        return _split_nonterminal(token)
    else:
        return _split_variable(token)


def _split_variable(variable: str) -> tuple[str, Optional[str]]:
    tmp = tuple(variable.rsplit('_', 1))
    if len(tmp) != 2:
        return variable, None
    return tuple(tmp)


def _split_nonterminal(nonterminal: str) -> tuple[str, Optional[str]]:
    tmp = tuple(nonterminal[1:-1].rsplit('_', 1))
    if len(tmp) != 2:
        return nonterminal, None
    return tuple(tmp)


def _get_alphabet_from_charclass(regexes: list) -> set[str]:
    alphabet = set()
    for opcode, value in regexes:
        if str(opcode) == 'LITERAL':
            alphabet.add(chr(value))
        elif str(opcode) == 'RANGE':
            for n in range(value[0], value[1]+1):
                alphabet.add(chr(n))
        else:
            raise ValueError(f'Unsupported opcode: {opcode}')
    return alphabet


def _parse_counter_oparands(regex_string: str) -> set[str]:
    counter_operands = set()
    parsed = re.sre_parse.parse(regex_string)

    if len(parsed) != 1:
        raise ValueError(f'Too many nodes: {regex_string}')

    opcode, value = parsed[0]
    if str(opcode) == 'LITERAL':
        counter_operands.add(chr(value))
    elif str(opcode) == 'IN':
        counter_operands |= _get_alphabet_from_charclass(value)
    else:
        raise ValueError(f'Unsupported opcode: {opcode}')

    return counter_operands


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    test_grammar = [
        '<S>->[LR]{2,10^5}',
    ]
    test_const = []

    ccfg = CountingContextFreeGrammar(test_grammar, test_const)
    print(ccfg)
    print(ccfg.generate())
