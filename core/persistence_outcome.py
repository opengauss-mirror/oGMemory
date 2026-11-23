"""Completion evidence carried with the original storage error."""
from enum import Enum, auto


class WriteCompletion(Enum):
    UNKNOWN = auto()
    CLEANUP_REQUIRED = auto()
    SETTLED = auto()


def record_write_completion(error: BaseException, outcome: WriteCompletion) -> None:
    error._write_completion = outcome


def write_completion(error: BaseException) -> WriteCompletion:
    return getattr(error, "_write_completion", WriteCompletion.UNKNOWN)
