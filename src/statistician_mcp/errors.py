from __future__ import annotations


class StatMcpError(Exception):
    """Base class for errors that should reach the caller as a structured
    `error_envelope` rather than a raw traceback."""

    code = "internal_error"

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class DatasetNotFoundError(StatMcpError):
    code = "dataset_not_found"

    def __init__(self, handle: str) -> None:
        super().__init__(
            f"no dataset found with handle '{handle}'",
            hint="call list_datasets to see valid handles in this workspace",
        )


class ColumnNotFoundError(StatMcpError):
    code = "column_not_found"

    def __init__(self, column: str, available: list[str]) -> None:
        super().__init__(
            f"column '{column}' not found",
            hint=f"available columns: {', '.join(available)}",
        )


class ValidationError(StatMcpError):
    code = "validation_error"


class QuotaExceededError(StatMcpError):
    code = "quota_exceeded"
