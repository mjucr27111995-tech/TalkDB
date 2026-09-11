import pytest

from talkdb.utils.sql_utils import extract_sql, validate_sql_is_select
from talkdb.utils.errors import TalkDBError


class TestExtractSql:
    def test_should_extract_sql_from_markdown_code_block(self) -> None:
        response = "Here is the query:\n```sql\nSELECT * FROM users\n```"
        assert extract_sql(response) == "SELECT * FROM users"

    def test_should_extract_sql_from_plain_code_block(self) -> None:
        response = "```\nSELECT id, name FROM products WHERE price > 10\n```"
        assert extract_sql(response) == "SELECT id, name FROM products WHERE price > 10"

    def test_should_extract_sql_from_raw_select(self) -> None:
        response = "SELECT name FROM employees WHERE department = 'Engineering'"
        assert extract_sql(response) == "SELECT name FROM employees WHERE department = 'Engineering'"

    def test_should_strip_trailing_semicolon(self) -> None:
        response = "```sql\nSELECT COUNT(*) FROM orders;\n```"
        assert extract_sql(response) == "SELECT COUNT(*) FROM orders"

    def test_should_raise_when_no_sql_found(self) -> None:
        with pytest.raises(TalkDBError, match="TALKDB-INT-002"):
            extract_sql("I cannot answer this question.")

    def test_should_extract_multiline_sql(self) -> None:
        response = """```sql
SELECT
    o.id,
    o.total,
    c.name
FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE o.total > 100
```"""
        result = extract_sql(response)
        assert "SELECT" in result
        assert "JOIN" in result


class TestValidateSqlIsSelect:
    def test_should_accept_select_query(self) -> None:
        validate_sql_is_select("SELECT * FROM users")

    def test_should_reject_insert_query(self) -> None:
        with pytest.raises(TalkDBError, match="TALKDB-VAL-004"):
            validate_sql_is_select("INSERT INTO users (name) VALUES ('test')")

    def test_should_reject_delete_query(self) -> None:
        with pytest.raises(TalkDBError, match="TALKDB-VAL-004"):
            validate_sql_is_select("DELETE FROM users WHERE id = 1")

    def test_should_reject_drop_query(self) -> None:
        with pytest.raises(TalkDBError, match="TALKDB-VAL-004"):
            validate_sql_is_select("DROP TABLE users")
