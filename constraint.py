import re
import logging
import itertools
import math
from typing import Union


ExtInt = Union[int, float]


class Constraint():

    def __init__(self):
        self.lower_bound = -math.inf
        self.upper_bound = math.inf
        self.inequal = set()

    def update_upper_bound(
        self, upper_bound: ExtInt, inclusive: bool
    ) -> None:
        if not inclusive:
            upper_bound -= 1
        self.upper_bound = min(self.upper_bound, upper_bound)

    def update_lower_bound(
        self, lower_bound: ExtInt, inclusive: bool
    ) -> None:
        if not inclusive:
            lower_bound += 1
        self.lower_bound = max(self.lower_bound, lower_bound)

    def update_inequal(self, not_equal: int) -> None:
        self.inequal.add(not_equal)

    def update(self, comparator: int, number: int) -> None:
        inclusive = (abs(comparator) == 1)

        if comparator == 0:
            self.update_inequal(number)
        elif comparator < 0:
            self.update_upper_bound(number, inclusive)
        else:
            self.update_lower_bound(number, inclusive)

    def __str__(self) -> str:
        return f"{self.lower_bound} <= ... <= {self.upper_bound}"


class Comparison():

    def __init__(self):
        self.lower_bounds = set()
        self.upper_bounds = set()
        self.inequal = set()

    def add_lower_bound(self, lower_bound: str, inclusive: bool) -> None:
        self.lower_bounds.add((lower_bound, inclusive))

    def add_upper_bound(self, upper_bound: str, inclusive: bool) -> None:
        self.upper_bounds.add((upper_bound, inclusive))

    def add_inequal(self, inequal: str) -> None:
        self.inequal.add(inequal)

    def update(self, comparator: int, target: str) -> None:

        # update compare dict
        inclusive = (abs(comparator) == 1)

        if comparator < 0:
            self.add_upper_bound(target, inclusive)
        if comparator > 0:
            self.add_lower_bound(target, inclusive)
        if not inclusive:
            self.add_inequal(target)

    def __str__(self) -> str:
        return " ".join([
            f"{self.lower_bounds}",
            f"<= {{}} (!= {self.inequal})",
            f"<= {self.upper_bounds}",
        ])


def parse_comparand(text: str) -> ExtInt:
    number_cbe_match = _RE_NUMBER_CBE.fullmatch(text)

    if not number_cbe_match:
        return text

    coefficient, base, exponent = number_cbe_match.group(1, 2, 3)

    if coefficient is None:
        coefficient = "1"
    if exponent is None:
        exponent = "1"

    return int(coefficient) * (int(base) ** int(exponent))


def _update_constraints_and_comparisons(
    text: str,
    constraints: dict[str, Constraint],
    comparisons: dict[str, Comparison]
) -> None:

    text = text.replace(" ", "")

    comparators = list(map(_parse_comparator, _RE_COMPARATOR.findall(text)))
    comparandss = [
        _parse_comparands(piece) for piece in _RE_COMPARATOR.split(text)
    ]

    for i, comparator in enumerate(comparators):
        leftss = comparandss[:i+1]
        rightss = comparandss[i+1:]

        logging.debug(
            f"{leftss} {_comparator_to_str(comparator)} {rightss}")

        for lefts, rights in itertools.product(leftss, rightss):
            for left, right in itertools.product(lefts, rights):
                logging.debug(
                    f"{left} {_comparator_to_str(comparator)} {right}")

                is_left_number = (type(left) is int)
                is_right_number = (type(right) is int)

                if not is_left_number and not is_right_number:

                    comparisons.setdefault(left, Comparison())
                    comparisons.setdefault(right, Comparison())

                    if left in comparisons:
                        comparisons[left].update(comparator, right)
                    if right in comparisons:
                        comparisons[right].update(comparator * -1, left)

                elif is_left_number != is_right_number:
                    # update constraint dict
                    _comparator = comparator
                    if is_left_number:
                        left, right = right, left
                        _comparator *= -1

                    constraints.setdefault(left, Constraint())
                    constraints[left].update(_comparator, right)

                else:  # E.g., 1 < 3
                    continue


def get_constraints_and_comparisons(
    constraint_strings: list[str]
) -> tuple[dict[str, Constraint], dict[str, Comparison]]:
    constraints = {}
    comparisons = {}
    for constraint_string in constraint_strings:
        _update_constraints_and_comparisons(
            constraint_string, constraints, comparisons)
    return constraints, comparisons


_RE_COMPARATOR = re.compile(r'<=|>=|<|>|!=')
_RE_NUMBER_CBE = re.compile(r'(?:(-?\d+)\*)?(-?\d+)(?:\^(-?\d+))?')
_RE_MIN_OR_MAX = re.compile(r'(?:min|max)\((\w+),(\w+)\)')


def _comparator_to_str(comparator: int) -> str:
    return {-2: "<", -1: "<=", 0: "!=", 1: ">=", 2: ">"}[comparator]


def _parse_comparator(text: str) -> int:
    return {"<": -2, "<=": -1, "!=": 0, ">=": 1, ">": 2}[text]


def _parse_comparands(text: str) -> list[ExtInt]:
    pieces = []
    # FIXME: Currently, we do not consider minimum or maximum.
    match = _RE_MIN_OR_MAX.fullmatch(text)
    if match:
        pieces = [match.group(1), match.group(2)]
    else:
        pieces = text.split(',')
    return [parse_comparand(e) for e in pieces]
