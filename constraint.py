import re
import logging
import itertools
from typing import Union, List

import jsonlines

from generator import test_case_generator

_RE_COMPARATOR = re.compile("<=|>=|<|>|!=")
_RE_NUMBER_CBE = re.compile(r'(?:(-?\d+)\*)?(-?\d+)(?:\^(-?\d+))?')

def _comparator_to_str(comparator: int) -> str:
    return { -2: "<", -1: "<=", 0: "!=", 1: ">=", 2: ">" }[comparator]

def _parse_comparator(text: str) -> int:
    return { "<": -2, "<=": -1, "!=": 0, ">=": 1, ">": 2 }[text]

def _parse_comparand(text: str) -> Union[int, str]:
    number_cbe_match = _RE_NUMBER_CBE.fullmatch(text)

    if not number_cbe_match: return text

    coefficient, base, exponent = number_cbe_match.group(1, 2, 3)

    if coefficient is None:
        coefficient = "1"
    if exponent is None:
        exponent = "1"

    return int(coefficient) * (int(base) ** int(exponent))

class Constraint():

    def __init__(
            self,
            start=None,
            end=None,
            start_inclusive=False,
            end_inclusive=False,
            not_equal_to=[]) -> None:

        self.start = start
        self.end = end
        self.start_inclusive = start_inclusive
        self.end_inclusive = end_inclusive
        self.not_equal_to = not_equal_to

    def __str__(self) -> str:
        return " ".join([
            str(self.start) if self.start is not None else "-inf",
            "<=" if self.start_inclusive else "<",
            "{}",
            "<=" if self.end_inclusive else "<",
            str(self.end) if self.end is not None else "+inf"
        ])

    def to_dict(self) -> dict:
        return {
            'start': str(self.start),
            'end': str(self.end),
            'include1': self.start_inclusive,
            'include2': self.end_inclusive,
            # 'not_equal_to': list(map(str, self.not_equal_to)),
        }


class Comparison():

    def __init__(
            self,
            source: str,
            target: str,
            comparator: int,
            same_variable: bool) -> None:

        self.source = source
        self.target = target
        self.comparator = comparator
        self.same_variable = same_variable


    @staticmethod
    def get_dict(comparisons: list) -> dict:
        compare_dict = {}

        for comparison in comparisons:
            if comparison.source in compare_dict:
                # raise NotImplementedError(
                    # "More than two comparisons for a single source")
                logging.warn("More than two comparisons for a single source")
            compare_dict[comparison.source] = {
                'target': comparison.target,
                'symbol': _comparator_to_str(comparison.comparator),
                'include': abs(comparison.comparator) == 1,
                'type': (
                    'same_variable'
                    if comparison.same_variable
                    else 'different_variable'
                )
            }

        return compare_dict


def _update_constraints(
        variable: str, number: int, comparator: int, constr_dict: dict) -> None:

    logging.debug("Update constr_dict")
    logging.debug(
        f"{variable} {_comparator_to_str(comparator)} {number}")

    if variable not in constr_dict:
        constr_dict[variable] = Constraint()

    constraint = constr_dict[variable]
    inclusive = (abs(comparator) == 1)

    if comparator == 0:
        constraint.not_equal_to.append(number)
    elif comparator < 0:
        if constraint.end is None or number < constraint.end:
            constraint.end = number
            constraint.end_inclusive = inclusive
    else:
        if constraint.start is None or constraint.start < number:
            constraint.start = number
            constraint.start_inclusive = inclusive


def update_constraints_and_comparisons(
        text: str, constr_dict: dict, comparisons: dict) -> None:
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
                logging.debug(f"{left} {_comparator_to_str(comparator)} {right}")

                is_left_number = (type(left) is int)
                is_right_number = (type(right) is int)

                if not is_left_number and not is_right_number:
                    # update compare dict
                    inclusive = (abs(comparator) == 1)

                    comparisons.append(
                        Comparison(
                            left, right, comparator,
                            left.split('_')[0] == right.split('_')[0]
                        )
                    )
                elif is_left_number != is_right_number:
                    # update constraint dict
                    local_comparator = comparator
                    if is_left_number:
                        left, right = right, left
                        local_comparator *= -1
                    _update_constraints(
                        left, right, local_comparator, constr_dict)
                else: # 1 < 3
                    continue


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)

    with jsonlines.open('data/grammer_sample.jsonl') as f:
        for problem in f:
            name = problem['name']
            grammer = problem['grammer']
            constraints = problem['constraints']

            generator = test_case_generator()
            generator(grammer, constraints)

            constr_dict = {}
            comparisons = []

            for constraint in constraints:
                update_constraints_and_comparisons(
                    constraint, constr_dict, comparisons)

            print()
            print("Constraints:")
            print(constraints)
            print()
            print("Original Const Dict:")
            print(generator.const_dict)
            print("Const Dict:")
            print({k: v.to_dict() for k, v in constr_dict.items()})
            print()
            print("Original Compare Dict:")
            print(generator.compare_dict)
            print("Compare Dict:")
            try:
                print(Comparison.get_dict(comparisons))
            except Exception:
                input()
