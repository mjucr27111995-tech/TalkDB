import re

import sqlparse

from talkdb.utils.errors import TalkDBError


def extract_sql(llm_response: str) -> str:
    """Extract a SQL query from an LLM response that may contain markdown or prose."""
    code_block = re.search(r"```(?:sql)?\s*\n?(.*?)```", llm_response, re.DOTALL)
    if code_block:
        candidate = code_block.group(1).strip()
    else:
        select_match = re.search(r"(SELECT\b.*?)(?:;|$)", llm_response, re.DOTALL | re.IGNORECASE)
        if select_match:
            candidate = select_match.group(1).strip()
        else:
            raise TalkDBError("TALKDB-INT-002", "Could not extract SQL from LLM response")

    if candidate.endswith(";"):
        candidate = candidate[:-1].strip()

    parsed = sqlparse.parse(candidate)
    if not parsed:
        raise TalkDBError("TALKDB-INT-002", "Could not parse extracted SQL")

    return candidate


def validate_sql_is_select(sql: str) -> None:
    """Ensure the SQL is a SELECT statement for safety."""
    parsed = sqlparse.parse(sql)
    if not parsed:
        raise TalkDBError("TALKDB-VAL-004", "Invalid SQL syntax")
    stmt_type = parsed[0].get_type()
    if stmt_type and stmt_type.upper() != "SELECT":
        raise TalkDBError(
            "TALKDB-VAL-004",
            f"Only SELECT queries are allowed, got: {stmt_type}",
        )
