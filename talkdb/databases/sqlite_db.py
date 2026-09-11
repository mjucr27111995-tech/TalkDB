import sqlite3
from pathlib import Path

import pandas as pd

from talkdb.databases.base import BaseDatabase
from talkdb.utils.errors import TalkDBError


class SQLiteDatabase(BaseDatabase):
    """SQLite database adapter."""

    def __init__(self) -> None:
        self._connection: sqlite3.Connection | None = None
        self._db_path: str = ""

    def connect(self, **kwargs) -> None:
        db_path = kwargs.get("db_path", "")
        if not db_path:
            raise TalkDBError("TALKDB-VAL-002", "db_path is required for SQLite")
        if not Path(db_path).exists():
            raise TalkDBError("TALKDB-NOT-001", f"Database file not found: {db_path}")
        self._db_path = db_path
        self._connection = sqlite3.connect(db_path, check_same_thread=False)

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
        cursor = self._connection.cursor()
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        row = cursor.fetchone()
        if row is None:
            raise TalkDBError("TALKDB-NOT-002", f"Table not found: {table_name}")
        return row[0]

    def get_all_ddls(self) -> dict[str, str]:
        if self._connection is None:
            raise TalkDBError("TALKDB-INT-001", "Not connected. Call connect() first.")
        cursor = self._connection.cursor()
        cursor.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return {row[0]: row[1] for row in cursor.fetchall()}

    def get_dialect(self) -> str:
        return "sqlite"

    def close(self) -> None:
        if self._connection:
            self._connection.close()
            self._connection = None
