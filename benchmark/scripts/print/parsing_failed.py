import os

import jsonlines


def main():
    generation_result_filename = os.environ["GROUND_TRUTH_GENERATION_RESULT"]
    parsing_result_filename = os.environ["GROUND_TRUTH_PARSING_RESULT"]

    grammar_filename = os.environ["GROUND_TRUTH_GRAMMAR"]
    testcase_filename = os.environ["GROUND_TRUTH_TESTCASE"]

    it = zip(
        jsonlines.open(generation_result_filename),
        jsonlines.open(parsing_result_filename),
        jsonlines.open(grammar_filename),
        jsonlines.open(testcase_filename)
    )
    for generation_obj, parsing_obj, grammar_obj, testcase_obj in it:
        grammar = grammar_obj["grammar"]
        name = grammar_obj["name"]

        assert name == testcase_obj["name"]
        assert name == generation_obj["name"]

        generation_results = generation_obj["results"]
        parsing_results = parsing_obj["results"]

        testcases = testcase_obj.get("testcase", [])
        assert len(generation_results) == len(parsing_results)
        assert len(generation_results) == len(testcases)

        it2 = zip(generation_results, parsing_results, testcases)
        for generation_result, parsing_result, testcase in it2:
            parsable = True
            error = None
            if not generation_result['parsable']:
                parsable = False
                error = generation_result["error"]
            elif not parsing_result['parsable']:
                parsable = False
                error = parsing_result["error"]

            if not parsable:
                print("Name: ", name)
                print("Grammar: ", grammar)
                if error is None:
                    print("Testcase: ", testcase)
                else:
                    print("Error: ", error)
                break


if __name__ == "__main__":
    main()
