"""A module for generating strings of counting context-free grammar."""

from enum import Enum
from functools import cmp_to_key
import logging
import math
import random
import re
from types import ModuleType
import typing
from typing import Any, Callable, cast, Optional, Protocol, Union

from .constraint import ExtInt
from .constraint import Normalization
from .constraint import normalize_variable
from .constraint import parse
from .constraint import parse_comparand
from .constraint import Placeholder
from .constraint import Variable
from .invalid_grammar_error import InvalidConstraintError
from .invalid_grammar_error import InvalidProductionError

logger = logging.getLogger(__name__)

sre_parse: ModuleType
try:
    import sre_parse  # pylint: disable=deprecated-module
except ImportError:
    sre_parse = re.sre_parse  # type: ignore [attr-defined]


Nonterminal = typing.NewType("Nonterminal", str)
Terminal = typing.NewType("Terminal", str)

Token = Union[Nonterminal, Variable, Terminal]
Assignment = dict[Variable, int]
Production = list[Token]

_T_contra = typing.TypeVar("_T_contra", contravariant=True)


class Comparator(Protocol[_T_contra]):
    def __call__(self, o1: _T_contra, o2: _T_contra) -> int:
        pass


MAX_SAMPLING_ITER = 2**16

_RE_REGEX_TERMINAL = re.compile(r"(.+?)\{([\w\-\*\^,]*)\}")
_RE_NUMBER_CBE = re.compile(r"(?:(-?\d+)\*)?(-?\d+)(?:\^(-?\d+))?")


class TokenType(Enum):
    TERMINAL = 0
    NONTERMINAL = 1  # <A>, <A_i>, <A_3>
    VARIABLE = 2  # [N], X, X_i, X_2, X_3, X_N


class SubscriptType(Enum):
    PLACEHOLDER = 0  # X_i, <A_i>
    VARIABLE = 1  # X_N, X_N-1, <A_N>
    CONSTANT = 2  # X_3, <A_3>
    PLACEHOLDER_DECREASING = 3  # X_i-1
    VARIABLE_DECREASING = 4  # X_N-1
    PLACEHOLDER_INCREASING = 5  # X_i+1
    VARIABLE_INCREASING = 6  # X_N+1


class CountingContextFreeGrammar:
    """A class for counting context-free grammar."""

    new_line_token = Terminal("<n>")
    space_token = Terminal("<s>")
    blank_token = Terminal("ε")

    derivation_token = "->"

    def __init__(
        self,
        productions: list[str],
        constraints: list[str],
    ):
        # Parse productions
        self.productions: dict[Token, list[Production]] = {}
        self.start_nonterminal: Nonterminal
        for i, rule_string in enumerate(productions):
            try:
                lhs, rhss = rule_string.split(self.derivation_token)
                lhs = lhs.strip()
                rhss = rhss.strip()
            except ValueError as e:
                raise InvalidProductionError("Improper production form") from e

            variable = Variable(lhs)
            if i == 0:
                self.start_nonterminal = Nonterminal(variable)
            self.productions[variable] = []
            for rhs in rhss.split("|"):
                tokenize = CountingContextFreeGrammar._tokenize
                production = [tokenize(e) for e in rhs.split()]
                self.productions[variable].append(production)

        # Parse constraints and comparisons
        parsed = parse(constraints)
        (
            self.constraints,
            self.comparisons,
            self.term_constraints,
            self.placeholders,
        ) = parsed

        # Add placeholders from constraints and comparisons
        for token in self.productions:
            _, subscript = self._split_token(token)
            if subscript is None:
                continue
            if self._is_placeholder(subscript):
                self.placeholders.add(cast(Placeholder, subscript))

        # Remove constraints and comparisons of placeholders
        for placeholder in self.placeholders:
            self.constraints.pop(Variable(placeholder), None)
            self.comparisons.pop(Variable(placeholder), None)

    def generate(self, *, degree: int = 0) -> str:
        """Randomly generate a string of the counting context-free grammar.

        Returns:
            A generated string in the counting context-free grammar.
        """

        string = ""

        derivation_queue: list[Token] = [self.start_nonterminal]
        assignment: Assignment = {}
        while derivation_queue:

            logger.debug("derivation_queue: %s", str(derivation_queue))
            logger.debug("assignment:\n%s", str(assignment))
            # logger.debug(f'string:\n{string}')

            token = derivation_queue.pop()
            production = None
            token_type = self._get_token_type(token)

            logger.debug("Current token: %s", str(token))
            logger.debug("Token type: %s", str(token_type))

            if token_type == TokenType.TERMINAL:
                terminal = cast(Terminal, token)
                string += self._sample_terminal(terminal, assignment, degree)

            elif token_type == TokenType.NONTERMINAL:
                nonterminal = cast(Nonterminal, token)
                production = self._derivate_nonterminal(nonterminal, assignment)

            elif token_type == TokenType.VARIABLE:
                variable = cast(Variable, token)

                production_, value = self._derivate_variable(
                    variable, assignment, degree
                )

                assignment_form_variable = variable

                if production_ is None and value is None:
                    raise InvalidConstraintError(
                        f"{variable} has no constraint"
                    )

                if self._is_counter(variable):
                    assignment_form_variable = Variable(variable[1:-1])
                    production = production_
                elif production_ is not None:
                    string += " ".join(production_)

                if value is not None:
                    assignment[assignment_form_variable] = value
                    string += str(value)

            if production is not None:
                derivation_queue += production[::-1]

        if string[-1] != "\n":
            string += "\n"
        return string

    def _substitute(
        self,
        token: Token,
        substitutable: Union[Variable, Placeholder],
        index: int,
    ) -> Token:
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

        fragment, subscript = self._split_token(token)
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

        if token[0] == "<" and token[-1] == ">":
            substituted = f"<{substituted}>"

        return cast(Token, substituted)

    def _substitute_production(
        self, production: Production, placeholder: Placeholder, index: int
    ) -> Production:

        return [
            self._substitute(token, placeholder, index) for token in production
        ]

    def _derivate_nonterminal(
        self, nonterminal: Nonterminal, assignment: Assignment
    ) -> Production:

        fragment, subscript = self._split_token(nonterminal)

        if subscript is None:
            return random.choice(self.productions[nonterminal])

        # XXX: Ad Hoc Grammar Errors
        if subscript == "-1":
            raise InvalidProductionError(
                f"Invalid indexed nonterminal {nonterminal}"
            )
        if subscript[-1] == ">":
            raise InvalidProductionError(
                f"Invalid indexed nonterminal {nonterminal}"
            )

        subscript_type = self._get_subscript_type(subscript)

        if subscript_type == SubscriptType.VARIABLE:
            variable = cast(Variable, subscript)
            return [
                self._substitute(nonterminal, variable, assignment[variable])
            ]
        elif subscript_type == SubscriptType.VARIABLE_DECREASING:
            variable = cast(Variable, subscript[:-2])
            return [
                self._substitute(nonterminal, variable, assignment[variable])
            ]
        elif subscript_type == SubscriptType.CONSTANT:
            index = int(subscript)

            if nonterminal in self.productions:  # E.g., <T_0> -> ...
                production = random.choice(list(self.productions[nonterminal]))

                # XXX: Implicit indexing
                if len(self.placeholders) == 1:
                    return self._substitute_production(
                        production, sorted(list(self.placeholders))[0], index
                    )
                return production

            placeholder: Optional[Placeholder] = None
            opt_production: Optional[Production] = None
            for placeholder_ in self.placeholders:
                unindexed = Nonterminal(f"<{fragment}_{placeholder_}>")
                if unindexed in self.productions:
                    opt_production = random.choice(self.productions[unindexed])
                    placeholder = placeholder_
                    break

            if opt_production is None or placeholder is None:
                raise ValueError(f"Cannot find production of {nonterminal}")
            production = cast(Production, opt_production)

            return self._substitute_production(production, placeholder, index)

        raise ValueError(f"Invalid nonterminal: {nonterminal}")

    def _derivate_variable(
        self, variable: Variable, assignment: Assignment, degree: int
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
        constraint_form_variables, index = self._to_constraint_form(variable)

        production_form_variables = constraint_form_variables
        if self._is_counter(variable):
            production_form_variables = [variable]

        if len(constraint_form_variables) == 0:
            value = None
        else:
            value = self._sample_variable(
                constraint_form_variables[0], assignment, index, degree=degree
            )

        # Find production of the variable
        productions = []
        for production_form_variable in production_form_variables:
            if production_form_variable in self.productions:
                production = random.choice(
                    self.productions[production_form_variable]
                )
                productions.append(production)
        productions = list(filter(lambda e: e is not None, productions))
        if len(productions) > 1:
            raise RuntimeError(f"Production is ambiguous for {variable}.")

        if len(productions) == 0:
            return None, value
        return productions[0], value

    def _to_constraint_form(
        self, variable: Variable
    ) -> tuple[list[Variable], Optional[int]]:
        """Format a variable to find the variable in constraints.

        Args:
            variable: A variable

        Returns:
            A tuple consists of the list of formatted variable and ``index``.
            If there is no index, ``index`` is None.
        Raises:
            ValueError: The given variable is in unindexed or variable-indexed
            form.
        """

        fragment, subscript = self._split_token(variable)
        if subscript is None:
            if self._is_counter(variable):
                new_variable = Variable(variable[1:-1])
                return [new_variable], None
            else:
                return [variable], None

        constraint_form_variables = []
        index = None

        subscript_type = self._get_subscript_type(subscript)

        if subscript_type == SubscriptType.CONSTANT:
            index = int(subscript)
            for placeholder in sorted(list(self.placeholders)):
                variable = Variable(f"{fragment}_{placeholder}")
                constraint_form_variables.append(variable)
        else:
            raise ValueError(f"Invalid variable: {variable}")

        return constraint_form_variables, index

    def _get_bound_and_unassigneds(
        self,
        bound_variables: set[tuple[Variable, bool]],
        constraint_bound: ExtInt,
        indexing: Optional[tuple[Placeholder, int]],
        tighten: Callable[[ExtInt], ExtInt],
        tighter_than: Comparator[ExtInt],
        assignment: Assignment,
    ) -> tuple[ExtInt, set[tuple[Variable, bool]]]:

        def tightest(bounds: list[ExtInt]) -> ExtInt:
            return max(bounds, key=cmp_to_key(tighter_than))

        index_variable: Callable[[Variable], Variable]
        if indexing is not None:
            placeholder, index = indexing

            def index_variable(variable: Variable) -> Variable:
                return cast(
                    Variable, self._substitute(variable, placeholder, index)
                )

        else:

            def index_variable(variable: Variable) -> Variable:
                return variable

        bounds = [constraint_bound]
        unassigned_bound_variables = set()
        for bound_variable, inclusive in bound_variables:
            bound = assignment.get(index_variable(bound_variable), None)
            if bound is not None:
                bounds.append(bound if inclusive else tighten(bound))
            else:
                unassigned_bound_variables.add((bound_variable, inclusive))
        bound = cast(int, tightest(bounds))
        return bound, unassigned_bound_variables

    def _filter_inner_variables(
        self,
        bound: int,
        unassigned_bound_variables: set[tuple[Variable, bool]],
        tighter_than: Comparator[ExtInt],
        get_target_bound: Callable[[Variable], ExtInt],
    ) -> set[tuple[Normalization, bool]]:
        inner_variables: set[tuple[Normalization, bool]] = set()
        for bound_variable, inclusive in unassigned_bound_variables:
            normalization = normalize_variable(bound_variable)
            normalized_variable, _, _ = normalization
            if tighter_than(get_target_bound(normalized_variable), bound) >= 0:
                inner_variables.add((normalization, inclusive))
        return inner_variables

    def _update_bound_and_count_inner_variables(
        self,
        variable: Variable,
        bound: int,
        indexing: Optional[tuple[Placeholder, int]],
        inner_variables: set[tuple[Normalization, bool]],
        tighten: Callable[[ExtInt, int], ExtInt],
    ) -> tuple[int, int]:
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
                if indexing is None or delta >= 0:
                    continue

                _, index = indexing

                # E.g., if a_i-1's are inner variables, a_N has N inner
                # variables
                number_of_indexed_inner_variables = index // (-delta)
                number_of_inner_variables += number_of_indexed_inner_variables
                if not inclusive:
                    opt_bound = tighten(
                        bound, number_of_indexed_inner_variables
                    )
                    bound = cast(int, opt_bound)
            else:
                raise NotImplementedError(
                    f"Constraint between {variable} and {inner_variable}"
                )
        return bound, number_of_inner_variables

    def _get_variable_bound(
        self,
        variable: Variable,
        assignment: Assignment,
        indexing: Optional[tuple[Placeholder, int]],
        reverse: bool = False,
    ) -> tuple[int, int]:
        logger.debug(
            "Get %s bound of %s", "lower" if reverse else "upper", variable
        )

        comparison = self.comparisons[variable]

        bound_variables: set[tuple[Variable, bool]]
        constraint_bound: ExtInt
        get_target_bound: Callable[[Variable], ExtInt]
        tighter_than: Comparator[ExtInt]
        tighten: Callable[[ExtInt, int], ExtInt]
        if not reverse:
            constraint_bound = self.constraints[variable].lower_bound
            bound_variables = comparison.lower_variables

            def get_target_bound(v: Variable) -> ExtInt:
                return self.constraints[v].upper_bound

            def tighter_than(o1: ExtInt, o2: ExtInt) -> int:
                if o1 == o2:
                    return 0
                elif o1 > o2:
                    return 1
                else:
                    return -1

            def tighten(e: ExtInt, n: ExtInt = 1) -> ExtInt:
                return e + n

        else:
            constraint_bound = self.constraints[variable].upper_bound
            bound_variables = comparison.upper_variables

            def get_target_bound(v: Variable) -> ExtInt:
                return self.constraints[v].lower_bound

            def tighter_than(o1: ExtInt, o2: ExtInt) -> int:
                if o1 == o2:
                    return 0
                elif o1 > o2:
                    return -1
                else:
                    return 1

            def tighten(e: ExtInt, n: ExtInt = 1) -> ExtInt:
                return e - n

        bound, unassigned_bound_variables = self._get_bound_and_unassigneds(
            bound_variables,
            constraint_bound,
            indexing,
            tighten,
            tighter_than,
            assignment,
        )

        if bound in [math.inf, -math.inf]:
            raise RuntimeError(f"Unbounded variable: {variable}")
        bound = cast(int, bound)

        # If there exists unassigned variables whose bounds are intersect with
        # the current variable, we have to consider them.

        # E.g., For a variable (a_i, N) (= a_N), inner variables
        # [(a_i, i, +1), (b, None, 0), ...]
        inner_variables = self._filter_inner_variables(
            bound, unassigned_bound_variables, tighter_than, get_target_bound
        )

        logger.debug(
            "unassigned bound variables: %s", unassigned_bound_variables
        )
        logger.debug("inner variables: %s", inner_variables)

        bound, number_of_inner_variables = (
            self._update_bound_and_count_inner_variables(
                variable, bound, indexing, inner_variables, tighten
            )
        )

        return bound, number_of_inner_variables

    def check_term_constraint(
        self, variable: str, value: int, assignment: Assignment
    ) -> bool:
        new_assignment = {str(k): v for k, v in assignment.items()}
        new_assignment[variable] = value
        for term_constraint in self.term_constraints:
            left_str, comparator, right_str = term_constraint
            try:
                # pylint: disable-next=eval-used
                left = eval(left_str, {}, new_assignment)
                # pylint: disable-next=eval-used
                right = eval(right_str, {}, new_assignment)
            except NameError:
                continue
            if comparator == "==" and not left == right:
                return False
            elif comparator == "!=" and not left != right:
                return False
            elif comparator == "<" and not left < right:
                return False
            elif comparator == ">" and not left > right:
                return False
            elif comparator == ">=" and not left >= right:
                return False
            elif comparator == "<=" and not left <= right:
                return False
        return True

    def _get_comparison_inequals(
        self,
        variable: Variable,
        assignment: Assignment,
        indexing: Optional[tuple[Placeholder, int]],
    ) -> set[int]:

        comparison = self.comparisons[variable]
        inequal_variables: set[Variable]

        if indexing is not None:
            placeholder, index = indexing
            inequal_variables = {
                cast(Variable, self._substitute(token, placeholder, index))
                for token in comparison.inequal_variables
            }
        else:
            inequal_variables = comparison.inequal_variables

        comparison_inequals: set[int] = set()
        for inequal_variable in inequal_variables:
            if inequal_variable in assignment:
                comparison_inequals.add(assignment[inequal_variable])
            else:
                fragment_1, _ = self._split_token(inequal_variable)
                fragment_2, _ = self._split_token(variable)
                if fragment_1 == fragment_2:
                    for key, value in assignment.items():
                        fragment_3, _ = self._split_token(key)
                        if fragment_3 == fragment_1:
                            comparison_inequals.add(value)
        return comparison_inequals

    def _sample_variable(
        self,
        variable: Variable,
        assignment: Assignment,
        index: Optional[int] = None,
        *,
        degree: int = 0,
    ) -> int:
        """Sample a value of a constraint-form variable

        Args:
            variable: A variable occurs in constraints.,
            index: Index of variable, which is omitted in ``variable``.

        Returns:
            Return an integer or ``None`` if the variable is not in
            constraints.
        """

        _, placeholder = self._split_token(variable)
        assert (placeholder is None) == (index is None)
        indexing = None
        if placeholder is not None and index is not None:
            indexing = Placeholder(placeholder), index

        constraint = self.constraints[variable]
        lower_bound, lower_inner_variables = self._get_variable_bound(
            variable, assignment, indexing
        )
        upper_bound, upper_inner_variables = self._get_variable_bound(
            variable, assignment, indexing, reverse=True
        )

        logger.debug(
            "(%s, %s) in [%s, %s]", variable, index, lower_bound, upper_bound
        )

        comparison_inequal_values = self._get_comparison_inequals(
            variable, assignment, indexing
        )
        inequal = constraint.inequal_values | comparison_inequal_values

        logger.debug("Sample a variable (%s, %s)", variable, index)
        logger.debug("[%s, %s]", lower_bound, upper_bound)
        logger.debug(self.comparisons[variable])

        # At this point, we know follows:
        # * the target variable is in a bound [`lower_bound`, `upper_bound`],
        # * the target variable is greater than `lower_variables`,
        # * the target variable is smaller than `upper_variables`, and
        # * `lower_` and `upper_variables` "may be" in the bound.
        # Heuristically, we choose the `k+1`-th one of `n` sampled values.

        n = lower_inner_variables + upper_inner_variables + 1
        k = lower_inner_variables

        def _sample_value() -> int:
            logger.debug("Sample a value of %s", variable)
            logger.debug("Index: %s", index)
            logger.debug("Lower bound: %s", lower_bound)
            logger.debug("Upper bound: %s", upper_bound)

            if degree == -1:
                return lower_bound

            if lower_bound < 0:
                return random.randint(lower_bound, upper_bound)

            distance = upper_bound - lower_bound
            for _ in range(degree):
                distance = int(math.log2(1 + abs(distance)))
            return random.randint(lower_bound, lower_bound + distance)

        # XXX: It depends on max iteration
        for _ in range(MAX_SAMPLING_ITER):
            values = [_sample_value() for _ in range(n)]
            value = sorted(values)[k]
            term_constraint_flag = self.check_term_constraint(
                variable, value, assignment
            )

            if value not in inequal and term_constraint_flag:
                return value

        raise RuntimeError(
            f"Failed to sample variable: {variable} with index {index}\n"
            + f"Assignment: {assignment}"
        )

    @staticmethod
    def _tokenize(string: str) -> Token:
        if string in {
            CountingContextFreeGrammar.new_line_token,
            CountingContextFreeGrammar.space_token,
            CountingContextFreeGrammar.blank_token,
        }:
            return Terminal(string)
        elif _RE_REGEX_TERMINAL.fullmatch(string):
            return Terminal(string)
        elif string[0] == "<" and string[-1] == ">":
            return Nonterminal(string)
        else:
            return Variable(string)

    def _get_token_type(self, token: str) -> TokenType:
        # raise DeprecationWarning("Now use typing instead")
        if token in {
            CountingContextFreeGrammar.new_line_token,
            CountingContextFreeGrammar.space_token,
            CountingContextFreeGrammar.blank_token,
        }:
            return TokenType.TERMINAL
        elif _RE_REGEX_TERMINAL.fullmatch(token):
            return TokenType.TERMINAL
        elif token[0] == "<" and token[-1] == ">":
            return TokenType.NONTERMINAL
        elif token[0] == "[" and token[-1] == "]":
            return TokenType.VARIABLE
        elif token in self.constraints:
            return TokenType.VARIABLE
        elif token in self.productions:
            return TokenType.VARIABLE
        elif "_" in token:
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
        self, counter_operator: str, assignment: Assignment, degree: int
    ) -> int:
        if _RE_NUMBER_CBE.fullmatch(counter_operator):
            parsed = parse_comparand(counter_operator)
            assert isinstance(parsed, int)
            return parsed
        elif counter_operator in assignment:
            return assignment[Variable(counter_operator)]
        elif counter_operator in self.constraints:
            variable = Variable(counter_operator)
            value = self._sample_variable(variable, assignment, degree=degree)
            assert value is not None
            assignment[variable] = value
            return value
        elif counter_operator.isdecimal():
            return int(counter_operator)
        else:
            raise ValueError(
                f"Counter operater parse failed: {counter_operator}"
            )

    def _sample_terminal(
        self, terminal: Terminal, assignment: Assignment, degree: int
    ) -> str:
        if terminal == CountingContextFreeGrammar.new_line_token:
            return "\n"
        elif terminal == CountingContextFreeGrammar.space_token:
            return " "
        elif terminal == CountingContextFreeGrammar.blank_token:
            return ""

        match = _RE_REGEX_TERMINAL.fullmatch(terminal)
        if not match:
            return terminal

        # Parse regex operands
        counter_operands = self._parse_counter_oparands(match.group(1))
        counter_operators = match.group(2).split(",")

        if len(counter_operators) > 2:
            raise ValueError(f"Too many counter operators: {terminal}")

        values = []
        for counter_operator in counter_operators:
            value = self._parse_counter_operator(
                counter_operator, assignment, degree
            )
            values.append(value)

        start = 0
        end = 0
        if len(values) == 1:
            start = values[0]
            end = values[0]
        elif len(values) == 2:
            start = values[0]
            end = values[1]

        terminal_len = random.choice(range(start, end + 1))
        terminal_string = ""
        for _ in range(terminal_len):
            terminal_string += random.choice(list(counter_operands))
        return terminal_string

    def __str__(self) -> str:
        return "\n".join(
            [
                "Productions:",
                str(self.productions),
                "Constraints:",
                str({k: str(v) for k, v in self.constraints.items()}),
                str(self.term_constraints),
                "Comparisons:",
                str({k: str(v) for k, v in self.comparisons.items()}),
                "Placeholders:",
                str(self.placeholders),
            ]
        )

    @staticmethod
    def _split_token(token: Token) -> tuple[str, Optional[str]]:

        tmp_string = str(token)
        if token[0] == "<" and token[-1] == ">":
            tmp_string = token[1:-1]

        tmp = tmp_string.rsplit("_", 1)
        if len(tmp) != 2:
            return token, None
        else:
            return tmp[0], tmp[1]

    @staticmethod
    def _get_alphabet_from_charclass(regexes: list[Any]) -> set[str]:
        alphabet = set()
        for opcode, value in regexes:
            if str(opcode) == "LITERAL":
                alphabet.add(chr(value))
            elif str(opcode) == "RANGE":
                for n in range(value[0], value[1] + 1):
                    alphabet.add(chr(n))
            else:
                raise ValueError(f"Unsupported opcode: {opcode}")
        return alphabet

    @staticmethod
    def _parse_counter_oparands(regex_string: str) -> set[str]:
        counter_operands = set()
        parsed = sre_parse.parse(regex_string)

        if len(parsed) != 1:
            raise ValueError(f"Too many nodes: {regex_string}")

        opcode, value = parsed[0]
        if str(opcode) == "LITERAL":
            counter_operands.add(chr(value))
        elif str(opcode) == "IN":
            counter_operands |= (
                CountingContextFreeGrammar._get_alphabet_from_charclass(value)
            )
        else:
            raise ValueError(f"Unsupported opcode: {opcode}")

        return counter_operands

    @staticmethod
    def _is_counter(variable: Variable) -> bool:
        return variable[0] == "[" and variable[-1] == "]"

    @staticmethod
    def _is_placeholder(subscript: str) -> bool:
        return subscript is not None and not subscript.isdecimal()
