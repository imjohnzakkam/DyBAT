class BaseTurboException:
    """Base Exception"""

class InvalidBackboneError(BaseTurboException):
    """Raised when the choice of backbone is invalid."""

class InvalidDatasetSelection(BaseTurboException):
    """Raised when the choice of dataset is invalid."""

class InvalidCheckpointPath(BaseTurboException):
    """Raised when the checkpoint path is invalid."""
