from enum import Enum
from typing import (Optional, Callable, Union, NewType, TypeVar, cast, )
import logging
import random
import re

from constraint import get_constraints_and_comparisons
from constraint import parse_comparand
from constraint import ExtInt
from constraint import Variable
# from generator import test_case_generator as TestCaseGenerator

Nonterminal = NewType('Nonterminal', str)
Terminal = NewType('Terminal', str)
Placeholder = NewType('Placeholder', str)

Token = Union[Nonterminal, Variable, Terminal]
Assignment = dict[Variable, int]
Production = list[Token]
Subscript = Union[None, int, Variable, Placeholder]

TToken = TypeVar('TToken', bound=Token)

MAX_ITER = 100
TESTMODE_VARIABLE_UPPER_BOUND = 50
TESTMODE_MAXIMUM_TERMINAL_LEN = 50

START_TOKEN = Variable('<S>')

NEW_LINE_TOKEN = Terminal('<n>')
SPACE_TOKEN = Terminal('<s>')
BLANK_TOKEN = Terminal('ε')

DERIVATE_TOKEN = '->'
# SEP_TOKEN = '\t'


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
        self.productions: dict[Token, list[Production]] = {}
        for rule_string in production_strings:
            lhs, rhss = rule_string.split(DERIVATE_TOKEN)
            variable = Variable(lhs)
            self.productions[variable] = []
            for rhs in rhss.split('|'):
                tokenize = CountingContextFreeGrammar._tokenize
                production = [tokenize(e) for e in rhs.split()]
                self.productions[Variable(lhs)].append(production)

        # Parse constraints and comparisons
        self.constraints, self.comparisons = (
            get_constraints_and_comparisons(constraint_strings))

        # Initialize placeholders from constraints and comparisons
        self.placeholders: set[Placeholder] = set()
        for token in self.productions:
            subscript: Subscript
            _, subscript = _split_token(token)
            if type(subscript) is str:
                self.placeholders.add(cast(Placeholder, subscript))

        # Remove constraints and comparisons of placeholders
        for placeholder in self.placeholders:
            if placeholder in self.constraints:
                del self.constraints[Variable(placeholder)]
            if placeholder in self.comparisons:
                del self.comparisons[Variable(placeholder)]

    def generate(self) -> str:
        """Randomly generate a string of the counting context-free grammar.

        Returns:
            A generated string in the counting context-free grammar.
        """

        string = ''

        derivation_queue: list[Token] = [START_TOKEN]
        assignment: Assignment = {}
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
                terminal = cast(Terminal, token)
                string += self._sample_terminal(terminal, assignment)

            elif token_type == TokenType.NONTERMINAL:
                nonterminal = cast(Nonterminal, token)
                production = (
                    self._derivate_nonterminal(nonterminal, assignment))

            elif token_type == TokenType.VARIABLE:
                variable = cast(Variable, token)
                variable_type = self._get_variable_type(variable)

                constraint_form_variables, index = (
                    self._to_constraint_form(variable))

                _production, value = self._derivate_variable(
                    variable, assignment)

                assignment_form_variable = variable

                if variable_type is VariableType.COUNTER:
                    assignment_form_variable = Variable(variable[1:-1])
                    production = _production

                elif _production is not None:
                    if value is not None:
                        raise RuntimeError(
                            f"A variable {variable} has both"
                            + " constraint and production.")
                    string += " ".join(_production)

                if value is not None:
                    assignment[assignment_form_variable] = value
                    string += str(value)

            if production is not None:
                derivation_queue += production[::-1]

        return string

    def _substitute(
        self,
        token: TToken,
        opt_subscript: Optional[Subscript],
        index: Optional[int]
    ) -> TToken:
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
        if opt_subscript is None or index is None:
            return token

        placeholder = cast(Placeholder, opt_subscript)
        token_type = self._get_token_type(token)

        if token_type == TokenType.NONTERMINAL:
            nonterminal = Nonterminal(token)
            nonterminal_type = self._get_nonterminal_type(nonterminal)
            if nonterminal_type not in (
                    {NonterminalType.UNINDEXED, NonterminalType.VAR_INDEXED}):
                return token

        elif token_type == TokenType.VARIABLE:
            variable = cast(Variable, token)
            variable_type = self._get_variable_type(variable)
            if variable_type not in (
                    {VariableType.UNINDEXED, VariableType.VAR_INDEXED}):
                return token

        return _substitute_token(token, placeholder, index)

    def _substitute_production(
        self,
        production: Production,
        placeholder: Placeholder,
        index: int
    ) -> Production:

        assert placeholder in self.placeholders

        return [
            self._substitute(token, placeholder, int(index))
            for token in production
        ]

    def _derivate_nonterminal(
        self, nonterminal: Nonterminal, assignment: Assignment
    ) -> Production:
        nonterminal_type = self._get_nonterminal_type(nonterminal)

        if nonterminal_type == NonterminalType.DEFAULT:
            return random.choice(self.productions[nonterminal])

        elif nonterminal_type == NonterminalType.UNINDEXED:
            raise ValueError(
                f"Invalid derivation of nonterminal {nonterminal}")

        elif nonterminal_type == NonterminalType.VAR_INDEXED:
            _, subscript = _split_token(nonterminal)
            assert type(subscript) is str
            variable = cast(Variable, subscript)

            return [self._substitute(
                nonterminal, variable, assignment[variable])]

        elif nonterminal_type == NonterminalType.INDEXED:
            if nonterminal in self.productions:  # E.g., <T_0> -> ...
                return random.choice(list(self.productions[nonterminal]))

            frag, opt_index = _split_token(nonterminal)
            assert type(opt_index) is int

            index = cast(int, opt_index)
            opt_placeholder = None
            production = None
            for _placeholder in self.placeholders:
                unindexed = Nonterminal(f"<{frag}_{_placeholder}>")
                if unindexed in self.productions:
                    production = random.choice(self.productions[unindexed])
                    opt_placeholder = _placeholder
                    break

            if production is None or opt_placeholder is None:
                raise ValueError(f"Cannot find production of {nonterminal}")

            placeholder = cast(Placeholder, opt_placeholder)
            return self._substitute_production(production, placeholder, index)

    def _derivate_variable(
        self, variable: Variable, assignment: Assignment
    ) -> tuple[Optional[Production], Optional[int]]:
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
                _production = random.choice(
                    self.productions[production_form_variable])
                productions.append(_production)
        productions = list(
            filter(lambda e: e is not None, productions))
        if len(productions) > 1:
            raise RuntimeError(
                f"Production is ambiguous for {variable}.")

        production = None
        if len(productions) == 1:
            production = productions[0]

        return production, value

    def _to_constraint_form(
        self,
        variable: Variable
    ) -> tuple[list[Variable], Optional[int]]:
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
            frag, _index = _split_token(variable)
            assert type(_index) is int
            index = _index

            for placeholder in self.placeholders:
                variable = Variable(f"{frag}_{placeholder}")
                constraint_form_variables.append(variable)
        elif variable_type == VariableType.COUNTER:
            variable = Variable(variable[1:-1])
            constraint_form_variables = [variable]
        else:
            raise ValueError(f"Invalid variable: {variable}")

        return constraint_form_variables, index

    def _get_variable_bound(
        self,
        variable: Variable,
        assignment: Assignment,
        index: Optional[int],
        reverse: bool = False
    ) -> tuple[int, list[Variable]]:

        comparison = self.comparisons[variable]

        constraint_bound: ExtInt
        bound_variables: set[tuple[Variable, bool]]
        tightest: Callable[[list[ExtInt]], ExtInt]
        get_target_bound: Callable[[Variable], ExtInt]
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

        subscript = None
        if index is not None:
            _, subscript = _split_token(variable)

        assert type(subscript) is not int
        opt_placeholder = cast(Optional[Placeholder], subscript)

        for bound_variable, inclusive in bound_variables:
            indexed_bound_variable = (
                self._substitute(bound_variable, opt_placeholder, index))

            if indexed_bound_variable in assignment:
                bound = assignment[indexed_bound_variable]
                if not inclusive:
                    bound = tighten(bound)
                bounds.append(bound)
            else:
                unassigned_bound_variables.append(bound_variable)

        _bound = tightest(bounds)
        assert type(_bound) is int
        bound = cast(int, _bound)

        # If there exists unassigned variables whose bounds are
        # intersect with the current variable, we have to make a room
        # for them.
        # TODO
        for bound_variable in unassigned_bound_variables:
            target_bound = get_target_bound(bound_variable)
            if tighter_than(target_bound, bound):
                pass
                # bound = tighten(bound)
        return bound, unassigned_bound_variables

    def _sample_variable(
        self,
        variable: Variable,
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

        lower_bound, _ = self._get_variable_bound(variable, assignment, index)
        upper_bound, _ = self._get_variable_bound(
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

    @staticmethod
    def _tokenize(string: str) -> Token:
        if string in {NEW_LINE_TOKEN, SPACE_TOKEN, BLANK_TOKEN}:
            return Terminal(string)
        elif _RE_REGEX_TERMINAL.fullmatch(string):
            return Terminal(string)
        elif string[0] == '<' and string[-1] == '>':
            return Nonterminal(string)
        else:
            return Variable(string)

    def _get_token_type(self, token: str) -> TokenType:
        # raise DeprecationWarning("Now use typing instead")
        if token in {NEW_LINE_TOKEN, SPACE_TOKEN, BLANK_TOKEN}:
            return TokenType.TERMINAL
        elif _RE_REGEX_TERMINAL.fullmatch(token):
            return TokenType.TERMINAL
        elif token[0] == '<' and token[-1] == '>':
            return TokenType.NONTERMINAL
        else:
            return TokenType.VARIABLE

    def _get_nonterminal_type(
        self,
        nonterminal: Nonterminal
    ) -> NonterminalType:
        _, index = _split_token(nonterminal)
        if index is None:
            return NonterminalType.DEFAULT
        elif type(index) is int:
            return NonterminalType.INDEXED
        elif index in self.constraints:
            return NonterminalType.VAR_INDEXED
        else:
            return NonterminalType.UNINDEXED

    def _get_variable_type(self, variable: Variable) -> VariableType:
        if variable[0] == '[' and variable[-1] == ']':
            return VariableType.COUNTER

        _, index = _split_token(variable)
        if index is None:
            return VariableType.DEFAULT
        elif type(index) is int:
            return VariableType.INDEXED
        elif index in self.constraints:
            return VariableType.VAR_INDEXED
        else:
            return VariableType.UNINDEXED

    def _parse_counter_operator(
        self,
        counter_operator: str,
        assignment: Assignment
    ) -> int:
        if _RE_NUMBER_CBE.fullmatch(counter_operator):
            parsed = parse_comparand(counter_operator)
            assert type(parsed) is int
            return parsed
        elif counter_operator in assignment:
            return assignment[Variable(counter_operator)]
        elif counter_operator in self.constraints:
            value = self._sample_variable(
                Variable(counter_operator), assignment)
            assert value is not None
            return value
            assignment[counter_operator] = value
        elif counter_operator.isdigit():
            return int(counter_operator)
        else:
            raise ValueError(
                f'Counter operater parse failed: {counter_operator}')

    def _sample_terminal(
        self, terminal: Terminal, assignment: Assignment
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


def _substitute_token(
    token: TToken,
    placeholder: Optional[Placeholder],
    index: int
) -> TToken:
    if placeholder is None:
        return token

    # XXX: Hard-coded
    frag, token_placeholder = _split_token(token)

    substituted: str
    if token_placeholder == placeholder:
        substituted = f"{frag}_{index}"
    elif token_placeholder == f"{placeholder}-1":
        substituted = f"{frag}_{index-1}"
    else:
        return token

    if token[0] == '<' and token[-1] == '>':
        substituted = f'<{substituted}>'
    return cast(TToken, substituted)


def _split_token(token: TToken) -> tuple[str, Subscript]:

    tmp_string = str(token)
    if token[0] == '<' and token[-1] == '>':
        tmp_string = token[1:-1]

    tmp = tuple(tmp_string.rsplit('_', 1))
    if len(tmp) != 2:
        return token, None
    elif tmp[1].isdigit():
        return tmp[0], int(tmp[1])
    else:
        return (tmp[0], Variable(tmp[1]))


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
    parsed = re.sre_parse.parse(regex_string)  # type: ignore[attr-defined]

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
    # logging.basicConfig(level=logging.DEBUG)

    production_strings = [
        '<S>->A <s> B <s> C <s> D',
    ]
    constraint_strings = [
        "0 <= A, B, C, D < 5 * 10 ^ 5",
        "A <= B",
        "B <= C",
        "C <= D"
    ]

    ccfg = CountingContextFreeGrammar(production_strings, constraint_strings)
    # generator = TestCaseGenerator()

    def true_distribution():
        variables = [random.randrange(0, 5 * (10 ** 5)) for _ in range(4)]
        return " ".join(map(str, sorted(variables)))

    def print_avg(generate):
        variable_lists = [[] for _ in range(4)]
        for _ in range(10000):
            variables = tuple(map(int, generate().split()))
            for i in range(4):
                variable_lists[i].append(variables[i])
        for i in range(4):
            variable_list = variable_lists[i]
            print(sum(variable_list)/len(variable_list))
        print()

    # def generator_distribution():
        # return generator(production_strings, constraint_strings)

    print()
    # print(ccfg)
    print_avg(ccfg.generate)
    # print_avg(generator_distribution)
    print_avg(true_distribution)
