"""Test the ground truth grammars."""

import argparse
import os
import sys
from pathlib import Path

import jsonlines
from tqdm import tqdm

from counting_context_free_grammar import CountingContextFreeGrammar as Ccfg
from counting_context_free_grammar import Discriminator

if __name__ == "__main__":
    ground_truth_dir = Path(os.environ["GROUND_TRUTH_GRAMMAR_DIR"])
    public_testcase_dir = Path(os.environ["PUBLIC_TESTCASE_DIR"])
    private_testcase_dir = Path(os.environ["PRIVATE_TESTCASE_DIR"])

    for filename in ["test.jsonl"]:
        ground_truth_file = jsonlines.open(ground_truth_dir / filename)
        public_testcase_file = jsonlines.open(public_testcase_dir / filename)
        private_testcase_file = jsonlines.open(private_testcase_dir / filename)

        for ground_truth_obj, public_obj, private_obj in tqdm(
            zip(ground_truth_file, public_testcase_file, private_testcase_file)
        ):
            name = ground_truth_obj["name"]
            assert name == public_obj["name"]
            assert name == private_obj["name"]

            grammar = ground_truth_obj["grammar"]
            productions = grammar["productions"]
            constraints = grammar["constraints"]

            d = Discriminator()
            testcase = None
            ccfg = None

            for testcase in public_obj["testcase"]:
                try:
                    d(productions, constraints, testcase)
                except Exception as e:
                    print(name)
                    print(grammar)
                    print("fails to parse public testcase")
                    print(testcase)
                    continue

            for testcase in private_obj["testcase"]:
                try:
                    d(productions, constraints, testcase)
                except Exception as e:
                    print(name)
                    print(grammar)
                    print("fails to parse private testcase")
                    print(testcase)
                    continue

            try:
                ccfg = Ccfg(productions, constraints)
                try:
                    testcase = ccfg.generate(degree=-1)
                except Exception as e:  # pylint: disable=broad-except
                    testcase = ccfg.generate(degree=1)
                d(productions, constraints, testcase)
            except Exception as e:  # pylint: disable=broad-except
                continue
                print(f"{type(e)}: {e}")
                print(f"productions: {productions}")
                print(f"constraints: {constraints}")
                if ccfg is not None:
                    print(f"ccfg: {ccfg}")
                if testcase is not None:
                    print(f"testcase: {testcase}")
