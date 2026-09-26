class TrackioAPIError(Exception):
    pass


class TrackioConflictError(TrackioAPIError, RuntimeError):
    """A write lost an optimistic-concurrency check and must be re-read.

    The server answers it with HTTP 409 and the remote client raises it again,
    so callers can distinguish a stale write from any other failure. It is a
    ``RuntimeError`` because remote-call failures have always surfaced as one.
    """

    status_code = 409

    def __init__(self, message: str, *, detail: dict | None = None) -> None:
        super().__init__(message)
        self.detail = dict(detail or {})
