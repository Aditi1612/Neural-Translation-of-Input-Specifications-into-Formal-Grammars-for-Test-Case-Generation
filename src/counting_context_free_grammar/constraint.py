"""Module for constraints"""

import itertools
import logging
import math
import re
import typing
from typing import Callable, cast, Optional, Union

ExtInt = Union[int, float]
Variable = typing.NewType("Variable", str)
Placeholder = typing.NewType("Placeholder", str)
Normalization = tuple[Variable, Optional[Placeholder], int]


class Constraint:
    """Class for constraints"""

    def __init__(self) -> None:
        self.lower_bound: ExtInt = -math.inf
        self.upper_bound: ExtInt = math.inf
        self.inequal_values: set[int] = set()

    def update_upper_bound(
        self: "Constraint", upper_bound: ExtInt, inclusive: bool = True
    ) -> "Constraint":
        if not inclusive:
            upper_bound -= 1
        self.upper_bound = min(self.upper_bound, upper_bound)
        return self

    def update_lower_bound(
        self: "Constraint", lower_bound: ExtInt, inclusive: bool = True
    ) -> "Constraint":
        if not inclusive:
            lower_bound += 1
        self.lower_bound = max(self.lower_bound, lower_bound)
        return self

    def add_inequal_value(self: "Constraint", not_equal: int) -> "Constraint":
        self.inequal_values.add(not_equal)
        return self

    def update(
        self: "Constraint", comparator: int, number: int
    ) -> "Constraint":
        inclusive = abs(comparator) == 1

        if comparator == 0:
            self.add_inequal_value(number)
        elif comparator == 3:
            self.update_upper_bound(number, True)
            self.update_lower_bound(number, True)
        elif comparator < 0:
            self.update_upper_bound(number, inclusive)
        else:
            self.update_lower_bound(number, inclusive)
        return self

    def update_constraint(
        self: "Constraint", constraint: "Constraint"
    ) -> "Constraint":
        self.update_upper_bound(constraint.upper_bound)
        self.update_lower_bound(constraint.lower_bound)
        for inequal in constraint.inequal_values:
            self.add_inequal_value(inequal)
        return self

    def __str__(self) -> str:
        return f"{self.lower_bound} <= ... <= {self.upper_bound}"


class Comparison:
    """Class for comparisons"""

    def __init__(self) -> None:
        self.lower_variables: set[tuple[Variable, bool]] = set()
        self.upper_variables: set[tuple[Variable, bool]] = set()
        self.inequal_variables: set[Variable] = set()

    def add_lower_variable(
        self: "Comparison", lower_bound: Variable, inclusive: bool
    ) -> "Comparison":
        self.lower_variables.add((lower_bound, inclusive))
        return self

    def add_upper_variable(
        self: "Comparison", upper_bound: Variable, inclusive: bool
    ) -> "Comparison":
        self.upper_variables.add((upper_bound, inclusive))
        return self

    def add_inequal_variable(
        self: "Comparison", inequal: Variable
    ) -> "Comparison":
        self.inequal_variables.add(inequal)
        return self

    def update(
        self: "Comparison", comparator: int, target: Variable
    ) -> "Comparison":

        # update compare dict
        inclusive = abs(comparator) == 1

        if comparator == 3:
            self.add_upper_variable(target, True)
            self.add_lower_variable(target, True)
        elif comparator < 0:
            self.add_upper_variable(target, inclusive)
        elif comparator > 0:
            self.add_lower_variable(target, inclusive)
        else:
            self.add_inequal_variable(target)
        return self

    def update_comparison(
        self: "Comparison", comparison: "Comparison"
    ) -> "Comparison":
        for lower_variable, inclusive in self.lower_variables:
            comparison.add_lower_variable(lower_variable, inclusive)
        for upper_variable, inclusive in self.upper_variables:
            comparison.add_upper_variable(upper_variable, inclusive)
        for inequal_variable in self.inequal_variables:
            comparison.add_inequal_variable(inequal_variable)
        return self

    def __str__(self) -> str:
        return " ".join(
            [
                f"{self.lower_variables}",
                f"<= {{}} (!= {self.inequal_variables})",
                f"<= {self.upper_variables}",
            ]
        )


Parsed = tuple[
    dict[Variable, Constraint],
    dict[Variable, Comparison],
    list[tuple[str, str, str]],
    set[Placeholder],
]


def parse_comparand(text: str) -> Union[Variable, int]:
    number_cbe_match = _RE_NUMBER_CBE.fullmatch(text)

    if not number_cbe_match:
        return Variable(text)

    coefficient, base, exponent = number_cbe_match.group(1, 2, 3)

    if coefficient is None:
        coefficient = "1"
    if exponent is None:
        exponent = "1"

    return int(int(coefficient) * (int(base) ** int(exponent)))


def _update_constraints_and_comparisons(
    text: str,
    constraints: dict[Variable, Constraint],
    comparisons: dict[Variable, Comparison],
    term_constraints: list[tuple[str, str, str]],
) -> None:

    text = text.replace(" ", "")

    comparators = list(map(_parse_comparator, _RE_COMPARATOR.findall(text)))
    comparandss = [
        _parse_comparands(piece) for piece in _RE_COMPARATOR.split(text)
    ]

    term_indicators = ["min", "max", "^", "*", "+", "-"]

    for i, comparator in enumerate(comparators):
        lefts = comparandss[i]
        rights = comparandss[i + 1]

        comparator_str = _comparator_to_str(comparator)

        for left, right in itertools.product(lefts, rights):

            is_left_number = isinstance(left, int)
            is_right_number = isinstance(right, int)

            # XXX: Hard-coded
            if not is_right_number:
                assert isinstance(right, str)
                if right[-2:] == "-1" and comparator_str == "<=":
                    right = right[:-2]
                    comparator = _parse_comparator("<")
                elif right[-2:] == "+1" and comparator_str == ">=":
                    right = right[:-2]
                    comparator = _parse_comparator(">")
            elif not is_left_number:
                assert isinstance(left, str)
                if left[-2:] == "+1" and comparator_str == "<=":
                    left = left[:-2]
                    comparator = _parse_comparator("<")
                elif left[-2:] == "-1" and comparator_str == ">=":
                    left = left[:-2]
                    comparator = _parse_comparator(">")

            # update term constraint
            def is_term(term: Union[str, int]) -> bool:
                if isinstance(term, int):
                    return False
                return any(indicator in term for indicator in term_indicators)

            if is_term(left) or is_term(right):
                assert isinstance(left, str)
                assert isinstance(right, str)
                term_constraints.append(
                    (left, _comparator_to_str(comparator), right)
                )
                continue

            if not is_left_number and not is_right_number:

                left_variable = cast(Variable, left)
                right_variable = cast(Variable, right)

                comparison = comparisons.setdefault(left_variable, Comparison())
                comparison.update(comparator, right_variable)

                comparison = comparisons.setdefault(
                    right_variable, Comparison()
                )
                comparison.update(-comparator, left_variable)

            elif is_left_number != is_right_number:
                # update constraint dict
                comparator_ = comparator
                if is_left_number:
                    left, right = right, left
                    comparator_ *= -1

                variable = cast(Variable, left)
                number = cast(int, right)

                constraints.setdefault(variable, Constraint())
                constraints[variable].update(comparator_, number)

            else:  # E.g., 1 < 3
                continue


def _to_transitive_bound(
    variable: Variable,
    constraints: dict[Variable, Constraint],
    comparisons: dict[Variable, Comparison],
    visited: Optional[set[Variable]],
    reverse: bool = False,
) -> tuple[ExtInt, set[tuple[Variable, bool]]]:
    """Make ``comparisons`` be transitive closure and update ``constraints``

    Args:
        variable: A variable
        comparisons: Constraints
        comparisons: Comparisons
        reverse: The function is called for upper bounds
        visited: Visited variables

    Returns:
        Updated lower (upper) bound and lower (upper) variables of ``variable``
    """
    if visited is None:
        visited = set()

    constraint = constraints[variable]
    comparison = comparisons[variable]
    if variable in visited:
        if not reverse:
            return constraint.lower_bound, comparison.lower_variables
        else:
            return constraint.upper_bound, comparison.upper_variables

    visited.add(variable)

    get_bound: Callable[[Constraint], ExtInt]
    update_bound: Callable[[ExtInt, bool], Constraint]
    bound_variables = None
    add_bound_variable: Callable[[Variable, bool], Comparison]
    if not reverse:

        def get_bound(constraint: Constraint) -> ExtInt:
            return constraint.lower_bound

        update_bound = constraint.update_lower_bound
        bound_variables = comparison.lower_variables
        add_bound_variable = comparison.add_lower_variable
    else:

        def get_bound(constraint: Constraint) -> ExtInt:
            return constraint.upper_bound

        update_bound = constraint.update_upper_bound
        bound_variables = comparison.upper_variables
        add_bound_variable = comparison.add_upper_variable

    bound = get_bound(constraint)
    for bound_variable, inclusive1 in bound_variables.copy():
        updated_bound, updated_variables = _to_transitive_bound(
            bound_variable, constraints, comparisons, visited, reverse
        )
        bound = get_bound(update_bound(updated_bound, inclusive1))

        for next_variable, inclusive2 in updated_variables:
            inclusive = inclusive1 and inclusive2
            add_bound_variable(next_variable, inclusive)

    return bound, bound_variables


def _split(variable: Variable) -> tuple[str, Optional[str]]:
    tmp = variable.rsplit("_", 1)
    if len(tmp) != 2:
        return tmp[0], None
    return tmp[0], tmp[1]


def _normalize_comparison(
    placeholder: Placeholder,
    comparison: Comparison,
    decrease: bool,
) -> Comparison:

    normalized_comparison = Comparison()

    modify_subscript: Callable[[str], str]
    if decrease:

        def modify_subscript(subscript: str) -> str:
            if subscript == f"{placeholder}+1":
                return f"{placeholder}"
            elif subscript == f"{placeholder}":
                return f"{placeholder}-1"
            else:
                return subscript

    else:

        def modify_subscript(subscript: str) -> str:
            if subscript == f"{placeholder}-1":
                return f"{placeholder}"
            elif subscript == f"{placeholder}":
                return f"{placeholder}+1"
            else:
                return subscript

    def modify(variable: Variable) -> Variable:
        fragment, subscript = _split(variable)
        if subscript is None:
            return variable
        subscript = modify_subscript(subscript)
        return Variable(f"{fragment}_{subscript}")

    for lower_variable, inclusive in comparison.lower_variables:
        lower_variable = modify(lower_variable)
        normalized_comparison.add_lower_variable(lower_variable, inclusive)

    for upper_variable, inclusive in comparison.upper_variables:
        upper_variable = modify(upper_variable)
        normalized_comparison.add_upper_variable(upper_variable, inclusive)

    for inequal_variable in comparison.inequal_variables:
        inequal_variable = modify(inequal_variable)
        normalized_comparison.add_inequal_variable(inequal_variable)

    return normalized_comparison


def normalize_variable(variable: Variable) -> Normalization:
    fragment, subscript = _split(variable)
    if subscript is None or len(subscript) < 3:
        return variable, None, 0

    # XXX: Hard-coded
    if not subscript[-2:] in {"-1", "+1"}:
        return variable, None, 0

    delta = int(subscript[-2:])
    placeholder = Placeholder(subscript[:-2])

    return Variable(f"{fragment}_{placeholder}"), placeholder, delta


def parse(constraint_strings: list[str]) -> Parsed:
    constraints: dict[Variable, Constraint] = {}
    comparisons: dict[Variable, Comparison] = {}
    term_constraints: list[tuple[str, str, str]] = []
    placeholders: set[Placeholder] = set()

    for constraint_string in constraint_strings:
        _update_constraints_and_comparisons(
            constraint_string, constraints, comparisons, term_constraints
        )

    variables = set(constraints.keys()) | set(comparisons.keys())
    for variable in variables:
        comparisons.setdefault(variable, Comparison())
        constraints.setdefault(variable, Constraint())

    lower_visited: set[Variable] = set()
    upper_visited: set[Variable] = set()
    for variable in variables:
        _to_transitive_bound(variable, constraints, comparisons, lower_visited)
        _to_transitive_bound(
            variable, constraints, comparisons, upper_visited, reverse=True
        )

    subscripts: set[str] = set()
    for variable in variables:
        _, subscript = _split(variable)
        if subscript is not None and subscript not in variables:
            subscripts.add(subscript)

    def proper_substring(a: str, b: str) -> bool:
        return a != b and a in b

    for subscript in subscripts:
        if not any(proper_substring(e, subscript) for e in subscripts):
            placeholders.add(Placeholder(subscript))

    for variable in variables:
        variable_normalization = normalize_variable(variable)
        normalized_variable, placeholder, delta = variable_normalization

        if placeholder is None or delta == 0:
            continue

        if delta not in [-1, 0, 1]:
            raise NotImplementedError("Placeholder delta must be -1, 0, or 1")

        placeholders.add(placeholder)

        if delta == 0:
            continue

        decrease = delta == 1

        constraint = constraints[variable]
        constraint.update_constraint(
            constraints.setdefault(normalized_variable, Constraint())
        )

        comparison = _normalize_comparison(
            placeholder, comparisons[variable], decrease
        )
        comparison.update_comparison(
            comparisons.setdefault(normalized_variable, Comparison())
        )

        constraints.pop(variable, None)
        comparisons.pop(variable, None)

    return constraints, comparisons, term_constraints, placeholders


_RE_COMPARATOR = re.compile(r"<=|>=|<|>|!=")
_RE_NUMBER_CBE = re.compile(r"(?:(-?\d+)\*)?(-?\d+)(?:\^(-?\d+))?")
_RE_MIN_OR_MAX = re.compile(r"(?:min|max)\((\w+(,\w+)*)\)")


def _comparator_to_str(comparator: int) -> str:
    return {-2: "<", -1: "<=", 0: "!=", 1: ">=", 2: ">", 3: "="}[comparator]


def _parse_comparator(text: str) -> int:
    return {"<": -2, "<=": -1, "!=": 0, ">=": 1, ">": 2, "=": 3}[text]


def _parse_comparands(text: str) -> list[Union[int, Variable, str]]:
    pieces: list[str] = []
    # FIXME: Currently, we do not consider minimum or maximum.
    match = _RE_MIN_OR_MAX.fullmatch(text)
    if match:
        pieces = match.group(1).split(",")
    else:
        pieces = text.split(",")
    return [parse_comparand(e) for e in pieces]


def main() -> None:
    # constraint_strings = ['0 <= a_i < 100', 'a_i < a_i+1']
    constraint_strings = ["0 <= A < B < C < D < 4"]
    constraints, comparisons, _, placeholders = parse(constraint_strings)

    print({k: str(v) for k, v in constraints.items()})
    print({k: str(v) for k, v in comparisons.items()})
    print(placeholders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
