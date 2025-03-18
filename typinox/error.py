class TypinoxError(Exception):
    """Base class for all exceptions raised by Typinox."""

    pass


class TypinoxAnnotationError(TypinoxError, TypeError):
    """Raised at annotation-time to indicate an error in the annotation."""

    pass


class TypinoxInvalidTypeToCheck(TypinoxError, TypeError):
    """Raised when the type being checked should not be checked at run-time."""

    pass


class TypinoxTypeViolation(TypinoxError, TypeError):
    """Raised when the typechecking failed."""

    pass


class TypinoxNotImplementedError(TypinoxError, NotImplementedError):
    """Raised when a feature is not yet implemented."""

    pass
