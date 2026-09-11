class TalkDBError(Exception):
    """Base error for all TalkDB operations.

    Error code format: TALKDB-[TYPE]-[NUM]
    Types: VAL (validation), NOT (not found), CON (conflict),
           EXT (external), INT (internal)
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")
