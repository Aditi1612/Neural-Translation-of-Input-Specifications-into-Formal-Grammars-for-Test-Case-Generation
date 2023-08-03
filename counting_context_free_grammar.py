from enum import Enum
from types import ModuleType
from typing import (Optional, Callable, Union, cast, )
import typing
import logging
import math
import random
import re
import sys

import jsonlines

import constraint
from constraint import normalize_variable
from constraint import (ExtInt, Variable, Placeholder, Normalization, )

from invalid_grammar_error import InvalidConstraintError
from invalid_grammar_error import InvalidProductionError

sre_parse: ModuleType
try:
    import sre_parse
except ImportError:
    sre_parse = re.sre_parse  # type: ignore[attr-defined]

Nonterminal = typing.NewType('Nonterminal', str)
Terminal = typing.NewType('Terminal', str)

Token = Union[Nonterminal, Variable, Terminal]
Assignment = dict[Variable, int]
Production = list[Token]

TToken = typing.TypeVar('TToken', bound=Token)

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


class SubscriptType(Enum):
    PLACEHOLDER = 0  # X_i, <A_i>
    VARIABLE = 1  # X_N, X_N-1, <A_N>
    CONSTANT = 2  # X_3, <A_3>
    PLACEHOLDER_DECREASING = 3  # X_i-1
    VARIABLE_DECREASING = 4  # X_i-1
    PLACEHOLDER_INCREASING = 5  # X_i+1
    VARIABLE_INCREASING = 6  # X_i+1


class CountingContextFreeGrammar:

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
            try:
                lhs, rhss = rule_string.split(DERIVATE_TOKEN)
            except ValueError:
                raise InvalidProductionError("Improper production form")

            variable = Variable(lhs)
            self.productions[variable] = []
            for rhs in rhss.split('|'):
                tokenize = CountingContextFreeGrammar._tokenize
                production = [tokenize(e) for e in rhs.split()]
                self.productions[Variable(lhs)].append(production)

        # Parse constraints and comparisons
        parsed = constraint.parse(constraint_strings)
        self.constraints, self.comparisons, self.placeholders = parsed

        # Add placeholders from constraints and comparisons
        for token in self.productions:
            _, subscript = _split_token(token)
            if subscript is None:
                continue
            if _is_placeholder(subscript):
                self.placeholders.add(cast(Placeholder, subscript))

        # Remove constraints and comparisons of placeholders
        for placeholder in self.placeholders:
            self.constraints.pop(Variable(placeholder), None)
            self.comparisons.pop(Variable(placeholder), None)

    def generate(self) -> str:
        """Randomly generate a string of the counting context-free grammar.

        Returns:
            A generated string in the counting context-free grammar.
        """

        string = ''

        derivation_queue: list[Token] = [START_TOKEN]
        assignment: Assignment = {}
        while derivation_queue:

            logging.debug(f'derivation_queue: {derivation_queue}')
            logging.debug(f'assignment:\n{assignment}')
            logging.debug(f'string:\n{string}')

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

                _production, value = (
                    self._derivate_variable(variable, assignment))

                assignment_form_variable = variable
                constraint_form_variables, index = (
                    self._to_constraint_form(variable))

                if _production is None and value is None:
                    raise InvalidConstraintError(
                        f"{variable} has no constraint")

                if _is_counter(variable):
                    assignment_form_variable = Variable(variable[1:-1])
                    production = _production
                elif _production is not None:
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
        substitutable: Union[Variable, Placeholder],
        index: int
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

        fragment, subscript = _split_token(token)
        if subscript is None:
            return token

        # XXX: Hard-coded
        substituted: str
        if subscript == substitutable:
            substituted = f"{fragment}_{index}"
        elif subscript == f"{substitutable}-1":
            substituted = f"{fragment}_{index-1}"
        elif subscript == f"{substitutable}+1":
            substituted = f"{fragment}_{index+1}"
        else:
            return token

        if token[0] == '<' and token[-1] == '>':
            substituted = f'<{substituted}>'

        return cast(TToken, substituted)

    def _substitute_production(
        self,
        production: Production,
        placeholder: Placeholder,
        index: int
    ) -> Production:

        return [
            self._substitute(token, placeholder, index)
            for token in production
        ]

    def _derivate_nonterminal(
        self, nonterminal: Nonterminal, assignment: Assignment
    ) -> Production:

        fragment, subscript = _split_token(nonterminal)

        if subscript is None:
            return random.choice(self.productions[nonterminal])

        # XXX: Ad Hoc Grammar Errors
        if subscript == "-1":
            raise InvalidProductionError(
                f"Invalid indexed nonterminal {nonterminal}")
        if subscript[-1] == ">":
            raise InvalidProductionError(
                f"Invalid indexed nonterminal {nonterminal}")

        subscript = cast(str, subscript)
        subscript_type = self._get_subscript_type(subscript)

        if subscript_type == SubscriptType.VARIABLE:
            variable = cast(Variable, subscript)
            return [self._substitute(
                nonterminal, variable, assignment[variable])]
        elif subscript_type == SubscriptType.VARIABLE_DECREASING:
            variable = cast(Variable, subscript[:-2])
            return [self._substitute(
                nonterminal, variable, assignment[variable])]
        elif subscript_type == SubscriptType.CONSTANT:
            index = int(subscript)

            if nonterminal in self.productions:  # E.g., <T_0> -> ...
                production = random.choice(list(self.productions[nonterminal]))

                # XXX: Implicit indexing
                if len(self.placeholders) == 1:
                    return self._substitute_production(
                        production, list(self.placeholders)[0], index)
                return production

            placeholder: Optional[Placeholder] = None
            opt_production: Optional[Production] = None
            for _placeholder in self.placeholders:
                unindexed = Nonterminal(f"<{fragment}_{_placeholder}>")
                if unindexed in self.productions:
                    opt_production = random.choice(self.productions[unindexed])
                    placeholder = _placeholder
                    break

            if opt_production is None or placeholder is None:
                raise ValueError(f"Cannot find production of {nonterminal}")
            production = cast(Production, opt_production)

            return self._substitute_production(production, placeholder, index)

        raise ValueError(f"Invalid nonterminal: {nonterminal}")

    def _derivate_variable(
        self, variable: Variable, assignment: Assignment
    ) -> tuple[Optional[Production], Optional[int]]:
        """Randomly select a production of variable and generate a value of the
        variable

        Args:
            variable: A variable
            assignment: An assignment

        Returns:
            A tuple consists of optional ``production`` and optional ``value``.
            If there is no production rule, then ``production`` is None.
            If there is no constraint, then ``value`` is None.

        Raises:
            RuntimeError: Constraint or production is ambiguous.
        """
        # XXX: For a variable a_3 with a constraint a_i < a_i+1,
        # we can not check "a_2 < a_3"
        constraint_form_variables, index = (
            self._to_constraint_form(variable))

        production_form_variables = constraint_form_variables
        if _is_counter(variable):
            production_form_variables = [variable]

        # Sample value of the variable
        values = [
            self._sample_variable(variable, assignment, index)
            for variable in constraint_form_variables
            if variable in self.constraints
        ]

        if len(values) > 1:
            raise InvalidConstraintError(
                f"Constraint is ambiguous for {variable}.")

        value = None if len(values) == 0 else values[0]

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
        """ Format a variable to find the variable in constraints.

        Args:
            variable: A variable

        Returns:
            A tuple consists of the list of formatted variable and ``index``.
            If there is no index, ``index`` is None.
        Raises:
            ValueError: The given variable is in unindexed or variable-indexed
            form.
        """

        fragment, subscript = _split_token(variable)
        if subscript is None:
            if _is_counter(variable):
                new_variable = Variable(variable[1:-1])
                return [new_variable], None
            else:
                return [variable], None

        constraint_form_variables = []
        index = None

        subscript = cast(str, subscript)
        subscript_type = self._get_subscript_type(subscript)

        if subscript_type == SubscriptType.CONSTANT:
            index = int(subscript)
            for placeholder in self.placeholders:
                variable = Variable(f"{fragment}_{placeholder}")
                constraint_form_variables.append(variable)
        else:
            raise ValueError(f"Invalid variable: {variable}")

        return constraint_form_variables, index

    def _get_variable_bound(
        self,
        variable: Variable,
        assignment: Assignment,
        indexing: Optional[tuple[Placeholder, int]],
        reverse: bool = False
    ) -> tuple[int, int]:
        logging.debug(
            f"Get {'lower' if reverse else 'upper'} bound of {variable}")

        comparison = self.comparisons[variable]

        constraint_bound: ExtInt
        bound_variables: set[tuple[Variable, bool]]
        tightest: Callable[[list[ExtInt]], ExtInt]
        get_target_bound: Callable[[Variable], ExtInt]
        tighter_than: Callable[[ExtInt, ExtInt], bool]
        tighten: Callable[[ExtInt], ExtInt]
        if not reverse:
            constraint_bound = self.constraints[variable].lower_bound
            bound_variables = comparison.lower_variables
            tightest = max
            def get_target_bound(e): return self.constraints[e].upper_bound
            def tighter_than(a, b): return a >= b
            def tighten(e, n=1): return e + n
        else:
            constraint_bound = self.constraints[variable].upper_bound
            bound_variables = comparison.upper_variables
            tightest = min
            def get_target_bound(e): return self.constraints[e].lower_bound
            def tighter_than(a, b): return a <= b
            def tighten(e, n=1): return e - n

        unassigned_bound_variables = []
        bounds = [constraint_bound]

        index_variable: Callable[[Variable, int], Variable]
        if indexing is not None:
            placeholder, index = cast(tuple[Placeholder, int], indexing)

            def index_variable(variable: Variable, delta: int = 0):
                return self._substitute(variable, placeholder, index)
        else:
            def index_variable(variable: Variable, delta: int = 0):
                return variable

        for bound_variable, inclusive in bound_variables:
            bound = assignment.get(index_variable(bound_variable), None)
            if bound is not None:
                bounds.append(bound if inclusive else tighten(bound))
            else:
                unassigned_bound_variables.append((bound_variable, inclusive))

        bound = cast(int, tightest(bounds))
        if bound in [math.inf, -math.inf]:
            raise RuntimeError(f"Unbounded variable: {variable}")

        # If there exists unassigned variables whose bounds are intersect with
        # the current variable, we have to consider them.

        # E.g., For a variable (a_i, N) (= a_N), inner variables
        # [(a_i, i, +1), (b, None, 0), ...]
        inner_variables: list[tuple[Normalization, bool]] = []
        for bound_variable, inclusive in unassigned_bound_variables:
            normalization = normalize_variable(bound_variable)
            normalized_variable, bound_placeholder, delta = normalization
            if tighter_than(get_target_bound(normalized_variable), bound):
                inner_variables.append((normalization, inclusive))

        logging.debug(
            f"unassigned bound variables: {unassigned_bound_variables}")
        logging.debug(f"inner variables: {inner_variables}")

        # Update the bounds and the number of constant-indexed inner_variables
        # according to the placeholder-indexed inner variables
        number_of_inner_variables = 0
        for normalization, inclusive in inner_variables:
            inner_variable, inner_placeholder, delta = normalization

            # E.g., (b, None, 0) yields one inner variable between the variable
            # and the bound. And current bound considering it.
            if inner_placeholder is None:
                number_of_inner_variables += 1

            # E.g., (a_i, i, xx)
            elif inner_variable == variable:

                # If we are considering non decreasing placeholder (e.g.,
                # a_i+1), we believe that a_N+1 must be assigned or not exists
                if delta >= 0:
                    continue

                # E.g., if a_i-1's are inner variables, a_N has N inner
                # variables
                number_of_indexed_inner_variables = index // (-delta)
                number_of_inner_variables += number_of_indexed_inner_variables
                if not inclusive:
                    bound = tighten(bound, number_of_indexed_inner_variables)
                    bound = cast(int, bound)
            else:
                raise NotImplementedError(
                    f"Constraint between {variable} and {inner_variable}")

        return bound, number_of_inner_variables

    def _sample_variable(
        self,
        variable: Variable,
        assignment: Assignment,
        index: Optional[int] = None
    ) -> int:
        """Sample a value of a constraint-form variable

        Args:
            variable: A variable occurs in constraints.,
            index: Index of variable, which is omitted in ``variable``.

        Returns:
            Return an integer or ``None`` if the variable is not in
            constraints.
        """

        _, placeholder = _split_token(variable)
        assert (placeholder is None) == (index is None)
        indexing = None
        if placeholder is not None and index is not None:
            indexing = Placeholder(placeholder), index

        constraint = self.constraints[variable]
        lower_bound, lower_inner_variables = self._get_variable_bound(
            variable, assignment, indexing)
        upper_bound, upper_inner_variables = self._get_variable_bound(
            variable, assignment, indexing, reverse=True)

        logging.debug(
            f"({variable}, {index}) in [{lower_bound}, {upper_bound}]")

        if self.testmode:
            upper_bound = min(upper_bound, TESTMODE_VARIABLE_UPPER_BOUND)
            upper_bound = max(lower_bound, upper_bound)

        comparison = self.comparisons[variable]
        inequal_variables: set[Variable]
        if indexing is not None:
            placeholder, index = indexing
            inequal_variables = {
                self._substitute(token, placeholder, index)
                for token in comparison.inequal_variables
            }
        else:
            inequal_variables = comparison.inequal_variables
        comparison_inequal = {
            assignment[variable]
            for variable in inequal_variables
            if variable in assignment
        }

        inequal = constraint.inequal_values | comparison_inequal

        logging.debug(f'Sample a variable ({variable}, {index})')
        logging.debug(f'[{lower_bound}, {upper_bound}]')
        logging.debug(comparison)

        # At this point, we knows follows:
        # * the target variable is in a bound [`lower_bound`, `upper_bound`],
        # * the target variable is greater than `lower_variables`,
        # * the target variable is smaller than `upper_variables`, and
        # * `lower_` and `upper_variables` "may be" in the bound.
        # Heuristically, we choose the `k+1`-th one of `n` sampled values.

        n = lower_inner_variables + upper_inner_variables + 1
        k = lower_inner_variables

        # XXX: It depends on max iteration
        for _ in range(MAX_ITER):
            values = [
                random.randint(lower_bound, upper_bound) for _ in range(n)
            ]
            value = sorted(values)[k]
            if value not in inequal:
                return value

        raise RuntimeError(
            f"Failed to sample variable: {variable} with index {index}\n"
            + f"Assignment: {assignment}")

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
        elif token[0] == '[' and token[-1] == ']':
            return TokenType.VARIABLE
        elif token in self.constraints:
            return TokenType.VARIABLE
        elif token in self.productions:
            return TokenType.VARIABLE
        elif '_' in token:
            return TokenType.VARIABLE
        else:
            return TokenType.TERMINAL

    def _get_subscript_type(self, subscript: str) -> SubscriptType:
        if subscript.isdecimal():
            return SubscriptType.CONSTANT
        elif subscript in self.constraints:
            return SubscriptType.VARIABLE
        elif subscript in self.placeholders:
            return SubscriptType.PLACEHOLDER

        if subscript[-2:] == "-1":
            if subscript[:-2] in self.constraints:
                return SubscriptType.VARIABLE_DECREASING
            elif subscript[:-2] in self.placeholders:
                return SubscriptType.PLACEHOLDER_DECREASING
        elif subscript[-2:] == "+1":
            if subscript[:-2] in self.constraints:
                return SubscriptType.VARIABLE_INCREASING
            elif subscript[:-2] in self.placeholders:
                return SubscriptType.PLACEHOLDER_INCREASING

        raise ValueError(f"Invalid subscript {subscript}")

    def _parse_counter_operator(
        self,
        counter_operator: str,
        assignment: Assignment
    ) -> int:
        if _RE_NUMBER_CBE.fullmatch(counter_operator):
            parsed = constraint.parse_comparand(counter_operator)
            assert type(parsed) is int
            return parsed
        elif counter_operator in assignment:
            return assignment[Variable(counter_operator)]
        elif counter_operator in self.constraints:
            variable = Variable(counter_operator)
            value = self._sample_variable(variable, assignment)
            assert value is not None
            assignment[variable] = value
            return value
        elif counter_operator.isdecimal():
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
        counter_operands = _parse_counter_oparands(match.group(1))
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
            terminal_string += random.choice(list(counter_operands))
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


def _split_token(token: TToken) -> tuple[str, Optional[str]]:

    tmp_string = str(token)
    if token[0] == '<' and token[-1] == '>':
        tmp_string = token[1:-1]

    tmp = tmp_string.rsplit('_', 1)
    if len(tmp) != 2:
        return token, None
    else:
        return tmp[0], tmp[1]


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
    parsed = sre_parse.parse(regex_string)  # type: ignore[attr-defined]

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


def _is_counter(variable: Variable) -> bool:
    return variable[0] == '[' and variable[-1] == ']'


def _is_placeholder(subscript: str) -> bool:
    return subscript is not None and not subscript.isdecimal()


if __name__ == '__main__':
    # 192, 1162
    # Invalid Grammar: 1186, 1188

    logging.basicConfig(level=logging.DEBUG)

    if len(sys.argv) < 1:
        raise ValueError("Invalid arguments")

    production_strings: list[str]
    constraint_strings: list[str]

    with jsonlines.open('data/train_grammer.jsonl') as problems:
        for problem in problems:
            problem_idx = problem['name']['index']

            if problem_idx != int(sys.argv[1]):
                continue

            specification = problem['spec']
            production_strings = cast(list[str], specification['grammer'])
            constraint_strings = cast(list[str], specification['constraints'])
            break

    print(production_strings)
    print(constraint_strings)
    ccfg = CountingContextFreeGrammar(
        production_strings, constraint_strings, testmode=True)
    print(ccfg)
    print(ccfg.generate())
