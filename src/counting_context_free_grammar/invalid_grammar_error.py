class InvalidGrammarError(Exception):
    """Class for invalid grammar errors."""

    pass


class InvalidConstraintError(InvalidGrammarError):
    """Class for invalid constraint errors."""

    pass


class InvalidProductionError(InvalidGrammarError):
    """Class for invalid production errors."""

    pass
