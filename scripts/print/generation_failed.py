import os

import jsonlines

filename = os.environ["GROUND_TRUTH_TESTCASE"]
grammar_filename = os.environ["GROUND_TRUTH_GRAMMAR"]

pairs = zip(jsonlines.open(filename), jsonlines.open(grammar_filename))

for testcase_result, grammar_object in pairs:
    grammar = grammar_object["grammar"]
    name = testcase_result["name"]
    error = testcase_result.get("error", None)
    if error is not None:
        print("Name: ", name)
        print("Grammar: ", grammar)
        print("Error: ", error)
