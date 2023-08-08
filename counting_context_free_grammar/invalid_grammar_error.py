class InvalidGrammarError(Exception):
    pass


class InvalidConstraintError(InvalidGrammarError):
    pass


class InvalidProductionError(InvalidGrammarError):
    pass
