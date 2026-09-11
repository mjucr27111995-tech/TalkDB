import sqlite3
import tempfile
from pathlib import Path

import pytest

from talkdb.databases.sqlite_db import SQLiteDatabase
from talkdb.utils.errors import TalkDBError


@pytest.fixture()
def sample_db(tmp_path: Path) -> str:
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT, price REAL)"
    )
    conn.execute("INSERT INTO products VALUES (1, 'Widget', 9.99)")
    conn.execute("INSERT INTO products VALUES (2, 'Gadget', 24.99)")
    conn.execute("INSERT INTO products VALUES (3, 'Doohickey', 4.50)")
    conn.commit()
    conn.close()
    return db_path


class TestSQLiteDatabase:
    def test_should_connect_with_valid_path(self, sample_db: str) -> None:
        db = SQLiteDatabase()
        db.connect(db_path=sample_db)
        assert db._connection is not None
        db.close()

    def test_should_raise_when_path_missing(self) -> None:
        db = SQLiteDatabase()
        with pytest.raises(TalkDBError, match="TALKDB-VAL-002"):
            db.connect()

    def test_should_raise_when_file_not_found(self) -> None:
        db = SQLiteDatabase()
        with pytest.raises(TalkDBError, match="TALKDB-NOT-001"):
            db.connect(db_path="/nonexistent/path.db")

    def test_should_execute_select_query(self, sample_db: str) -> None:
        db = SQLiteDatabase()
        db.connect(db_path=sample_db)
        df = db.execute("SELECT * FROM products WHERE price > 5")
        assert len(df) == 2
        db.close()

    def test_should_get_table_ddl(self, sample_db: str) -> None:
        db = SQLiteDatabase()
        db.connect(db_path=sample_db)
        ddl = db.get_table_ddl("products")
        assert "CREATE TABLE" in ddl
        assert "products" in ddl
        db.close()

    def test_should_get_all_ddls(self, sample_db: str) -> None:
        db = SQLiteDatabase()
        db.connect(db_path=sample_db)
        ddls = db.get_all_ddls()
        assert "products" in ddls
        db.close()

    def test_should_return_sqlite_dialect(self, sample_db: str) -> None:
        db = SQLiteDatabase()
        db.connect(db_path=sample_db)
        assert db.get_dialect() == "sqlite"
        db.close()

    def test_should_raise_when_not_connected(self) -> None:
        db = SQLiteDatabase()
        with pytest.raises(TalkDBError, match="TALKDB-INT-001"):
            db.execute("SELECT 1")

    def test_should_raise_on_nonexistent_table(self, sample_db: str) -> None:
        db = SQLiteDatabase()
        db.connect(db_path=sample_db)
        with pytest.raises(TalkDBError, match="TALKDB-NOT-002"):
            db.get_table_ddl("nonexistent")
        db.close()
