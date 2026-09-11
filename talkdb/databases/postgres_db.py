import pandas as pd
import psycopg2

from talkdb.databases.base import BaseDatabase
from talkdb.utils.errors import TalkDBError


class PostgresDatabase(BaseDatabase):
    """PostgreSQL database adapter."""

    def __init__(self) -> None:
        self._connection = None

    def connect(self, **kwargs) -> None:
        required = ("host", "database", "user", "password")
        missing = [k for k in required if not kwargs.get(k)]
        if missing:
            raise TalkDBError(
                "TALKDB-VAL-003",
                f"Missing required PostgreSQL config: {', '.join(missing)}",
            )
        try:
            self._connection = psycopg2.connect(
                host=kwargs["host"],
                database=kwargs["database"],
                user=kwargs["user"],
                password=kwargs["password"],
                port=kwargs.get("port", 5432),
            )
        except psycopg2.Error as exc:
            raise TalkDBError("TALKDB-EXT-004", f"PostgreSQL connection failed: {exc}") from exc

    def execute(self, sql: str) -> pd.DataFrame:
        if self._connection is None:
            raise TalkDBError("TALKDB-INT-001", "Not connected. Call connect() first.")
        try:
            return pd.read_sql_query(sql, self._connection)
        except Exception as exc:
            raise TalkDBError("TALKDB-EXT-003", f"SQL execution failed: {exc}") from exc

    def get_table_ddl(self, table_name: str) -> str:
        if self._connection is None:
            raise TalkDBError("TALKDB-INT-001", "Not connected. Call connect() first.")
        sql = """
        SELECT
            'CREATE TABLE ' || c.table_name || ' (' ||
            string_agg(
                c.column_name || ' ' || c.data_type ||
                CASE WHEN c.character_maximum_length IS NOT NULL
                     THEN '(' || c.character_maximum_length || ')'
                     ELSE '' END ||
                CASE WHEN c.is_nullable = 'NO' THEN ' NOT NULL' ELSE '' END,
                ', '
            ) || ');'
        FROM information_schema.columns c
        WHERE c.table_schema = 'public' AND c.table_name = %s
        GROUP BY c.table_name
        """
        cursor = self._connection.cursor()
        cursor.execute(sql, (table_name,))
        row = cursor.fetchone()
        cursor.close()
        if row is None:
            raise TalkDBError("TALKDB-NOT-002", f"Table not found: {table_name}")
        return row[0]

    def get_all_ddls(self) -> dict[str, str]:
        if self._connection is None:
            raise TalkDBError("TALKDB-INT-001", "Not connected. Call connect() first.")
        cursor = self._connection.cursor()
        cursor.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        )
        tables = [row[0] for row in cursor.fetchall()]
        cursor.close()
        return {table: self.get_table_ddl(table) for table in tables}

    def get_dialect(self) -> str:
        return "postgresql"

    def close(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None
