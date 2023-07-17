import re
import logging
import itertools
import math

import jsonlines

from generator import TestCaseGenerator

_RE_COMPARATOR = re.compile("<=|>=|<|>|!=")
_RE_NUMBER_CBE = re.compile(r'(?:(-?\d+)\*)?(-?\d+)(?:\^(-?\d+))?')


def _comparator_to_str(comparator: int) -> str:
    return {-2: "<", -1: "<=", 0: "!=", 1: ">=", 2: ">"}[comparator]


def _parse_comparator(text: str) -> int:
    return {"<": -2, "<=": -1, "!=": 0, ">=": 1, ">": 2}[text]


def _parse_comparand(text: str):
    number_cbe_match = _RE_NUMBER_CBE.fullmatch(text)

    if not number_cbe_match:
        return text

    coefficient, base, exponent = number_cbe_match.group(1, 2, 3)

    if coefficient is None:
        coefficient = "1"
    if exponent is None:
        exponent = "1"

    return int(coefficient) * (int(base) ** int(exponent))


class Constraint():

    def __init__(self):
        self.lower_bound = -math.inf
        self.upper_bound = math.inf
        self.inequal = set()

    def update_upper_bound(self, upper_bound, inclusive):
        if not inclusive:
            upper_bound -= 1
        self.upper_bound = min(self.upper_bound, upper_bound)

    def update_lower_bound(self, lower_bound, inclusive):
        if not inclusive:
            lower_bound += 1
        self.lower_bound = max(self.lower_bound, lower_bound)

    def update_inequal(self, not_equal):
        self.inequal.add(not_equal)

    def __str__(self) -> str:
        return f"{self.lower_bound} <= ... <= {self.upper_bound}"


class Comparison():

    def __init__(self):
        self.lower_bounds = set()
        self.upper_bounds = set()
        self.inequal = set()

    def add_lower_bound(self, lower_bound, inclusive):
        self.lower_bounds.add((lower_bound, inclusive))

    def add_upper_bound(self, upper_bound, inclusive):
        self.upper_bounds.add((upper_bound, inclusive))

    def add_inequal(self, inequal):
        self.inequal.add(inequal)

    def __str__(self):
        return " ".join([
            f"{self.lower_bounds}",
            f"<= {{}} (!= {self.inequal})",
            f"<= {self.upper_bounds}",
        ])


def _update_constraints(
        variable: str, number: int, comparator: int, constraints: dict):

    logging.debug("Update constraints")
    logging.debug(
        f"{variable} {_comparator_to_str(comparator)} {number}")

    if variable not in constraints:
        constraints[variable] = Constraint()

    constraint = constraints[variable]
    inclusive = (abs(comparator) == 1)

    if comparator == 0:
        constraint.update_inequal(number)
    elif comparator < 0:
        constraint.update_upper_bound(number, inclusive)
    else:
        constraint.update_lower_bound(number, inclusive)


def update_constraints_and_comparisons(
        text: str, constraints: dict, comparisons: dict) -> None:
    text = text.replace(" ", "")

    comparators = list(map(_parse_comparator, _RE_COMPARATOR.findall(text)))
    comparandss = [
        list(map(_parse_comparand, piece.split(",")))
        for piece in _RE_COMPARATOR.split(text)
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

                    # update compare dict
                    if left not in constraints:
                        constraints[left] = Constraint()
                    if right not in constraints:
                        constraints[right] = Constraint()

                    inclusive = (abs(comparator) == 1)
                    if left not in comparisons:
                        comparisons[left] = Comparison()
                    if right not in comparisons:
                        comparisons[right] = Comparison()
                    if comparator < 0:
                        comparisons[left].add_upper_bound(right, inclusive)
                        comparisons[right].add_lower_bound(left, inclusive)
                    if comparator > 0:
                        comparisons[right].add_upper_bound(left, inclusive)
                        comparisons[left].add_lower_bound(right, inclusive)
                    if not inclusive:
                        comparisons[left].add_inequal(right)
                        comparisons[right].add_inequal(left)

                elif is_left_number != is_right_number:
                    # update constraint dict
                    local_comparator = comparator
                    if is_left_number:
                        left, right = right, left
                        local_comparator *= -1
                    _update_constraints(
                        left, right, local_comparator, constraints)
                else:  # 1 < 3
                    continue


def get_constraints_and_comparisons(constraint_strings):
    constraints = {}
    comparisons = {}

    for constraint_string in constraint_strings:
        update_constraints_and_comparisons(
            constraint_string, constraints, comparisons)
    return constraints, comparisons


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    with jsonlines.open('data/grammer_sample.jsonl') as f:
        for problem in f:
            name = problem['name']
            grammer = problem['grammer']
            constraint_strings = problem['constraints']

            generator = TestCaseGenerator()
            generator(grammer, constraint_strings)

            constraints, comparisons = (
                get_constraints_and_comparisons(constraint_strings))

            print()
            print("Constraints:")
            print(constraints)
            print()
            print("Original Const Dict:")
            print(generator.const_dict)
            print("Const Dict:")
            print({k: v.to_dict() for k, v in constraints.items()})
            print()
            print("Original Compare Dict:")
            print(generator.compare_dict)
            print("Compare Dict:")
            try:
                print(Comparison.get_dict(comparisons))
            except Exception:
                input()
