class AQAError(Exception):
    """Base error with a stable machine-readable code."""

    code = "aqa_error"

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(AQAError):
    code = "configuration_error"


class ValidationError(AQAError):
    code = "validation_error"


class NotFoundError(AQAError):
    code = "not_found"


class DataUnavailableError(AQAError):
    code = "data_unavailable"


class IngestionError(AQAError):
    code = "ingestion_error"

