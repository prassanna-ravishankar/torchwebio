class BaseTorchWebioException(Exception):
    """Base exception of the library"""

    pass


class ComingSoonException(BaseTorchWebioException):
    """A special exception for features that are not yet implemented, but are planned"""

    def __init__(self, feature, error):
        self.message = f"The '{feature}' is coming soon, it is not yet implemented"
        super().__init__(self.message)
