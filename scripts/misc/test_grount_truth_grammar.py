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
    for filename in ["train.jsonl", "test.jsonl", "new_train.jsonl"]:
        print("filename:", filename)
        for obj in tqdm(jsonlines.open(ground_truth_dir / filename)):
            grammar = obj["grammar"]
            productions = grammar["productions"]
            constraints = grammar["constraints"]

            d = Discriminator()
            testcase = None
            ccfg = None
            try:
                ccfg = Ccfg(productions, constraints)
                testcase = ccfg.generate(degree=2)
                d(productions, constraints, testcase)
            except Exception as e:  # pylint: disable=broad-except
                print(f"Error: {e}")
                print(f"productions: {productions}")
                print(f"constraints: {constraints}")
                if ccfg is not None:
                    print(f"ccfg: {ccfg}")
                if testcase is not None:
                    print(f"testcase: {testcase}")
