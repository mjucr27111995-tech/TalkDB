import pytest

from talkdb.utils.errors import TalkDBError


class TestTalkDBError:
    def test_should_format_error_with_code_and_message(self) -> None:
        error = TalkDBError("TALKDB-VAL-001", "API key is required")
        assert error.code == "TALKDB-VAL-001"
        assert error.message == "API key is required"
        assert str(error) == "[TALKDB-VAL-001] API key is required"

    def test_should_be_catchable_as_exception(self) -> None:
        with pytest.raises(Exception):
            raise TalkDBError("TALKDB-INT-001", "Something went wrong")
