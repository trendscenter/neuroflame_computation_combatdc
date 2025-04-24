class ValidationException(Exception):
    def __init__(self, message: str, *args) -> None:
        super().__init__(message)
        self.message = message
        self.args = args

    def __str__(self) -> str:
        return f"{self.message} {self.args}"