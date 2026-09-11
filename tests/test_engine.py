import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from talkdb.core.engine import TalkDB, QueryResult, _QueryCache
from talkdb.databases.sqlite_db import SQLiteDatabase
from talkdb.llms.base import BaseLLM
from talkdb.utils.errors import TalkDBError
from talkdb.vectorstores.base import BaseVectorStore, RetrievalResult


class FakeLLM(BaseLLM):
    """Test double that returns predetermined responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses) if responses else []
        self._call_index = 0

    def generate(self, prompt: str) -> str:
        return self._next_response()

    def generate_with_history(self, messages: list[dict[str, str]]) -> str:
        return self._next_response()

    def get_provider_name(self) -> str:
        return "FakeLLM"

    def _next_response(self) -> str:
        if self._call_index < len(self._responses):
            resp = self._responses[self._call_index]
            self._call_index += 1
            return resp
        return "No more responses"


class FakeVectorStore(BaseVectorStore):
    """Test double with simple in-memory storage."""

    def __init__(self) -> None:
        self._ddls: list[RetrievalResult] = []
        self._examples: list[RetrievalResult] = []

    def add_ddl(self, table_name: str, ddl: str) -> None:
        self._ddls.append(
            RetrievalResult(content=ddl, score=1.0, metadata={"table_name": table_name})
        )

    def add_example(self, question: str, sql: str) -> None:
        combined = f"Question: {question}\nSQL: {sql}"
        self._examples.append(
            RetrievalResult(content=combined, score=1.0, metadata={"question": question, "sql": sql})
        )

    def add_documentation(self, doc: str) -> None:
        pass

    def find_relevant_ddls(self, question: str, top_k: int = 5) -> list[RetrievalResult]:
        return self._ddls[:top_k]

    def find_relevant_examples(self, question: str, top_k: int = 3) -> list[RetrievalResult]:
        return self._examples[:top_k]

    def find_relevant_docs(self, question: str, top_k: int = 3) -> list[RetrievalResult]:
        return []

    def clear(self) -> None:
        self._ddls.clear()
        self._examples.clear()


@pytest.fixture()
def sample_db(tmp_path: Path) -> SQLiteDatabase:
    db_path = str(tmp_path / "test.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE products (id INTEGER PRIMARY KEY, name TEXT NOT NULL, price REAL NOT NULL)"
    )
    conn.execute("INSERT INTO products VALUES (1, 'Widget', 9.99)")
    conn.execute("INSERT INTO products VALUES (2, 'Gadget', 24.99)")
    conn.execute("INSERT INTO products VALUES (3, 'Doohickey', 4.50)")
    conn.commit()
    conn.close()
    db = SQLiteDatabase()
    db.connect(db_path=db_path)
    return db


# ── Init tests ────────────────────────────────────────────


class TestTalkDBInit:
    def test_should_raise_when_llm_is_none(self) -> None:
        with pytest.raises(TalkDBError, match="TALKDB-VAL-005"):
            TalkDB(llm=None, vectorstore=FakeVectorStore(), database=MagicMock())  # type: ignore[arg-type]

    def test_should_raise_when_vectorstore_is_none(self) -> None:
        with pytest.raises(TalkDBError, match="TALKDB-VAL-005"):
            TalkDB(llm=FakeLLM(), vectorstore=None, database=MagicMock())  # type: ignore[arg-type]

    def test_should_raise_when_database_is_none(self) -> None:
        with pytest.raises(TalkDBError, match="TALKDB-VAL-005"):
            TalkDB(llm=FakeLLM(), vectorstore=FakeVectorStore(), database=None)  # type: ignore[arg-type]


# ── Ask tests ─────────────────────────────────────────────


class TestTalkDBAsk:
    def test_should_return_query_result_for_valid_question(
        self, sample_db: SQLiteDatabase
    ) -> None:
        llm = FakeLLM(responses=[
            "```sql\nSELECT * FROM products WHERE price > 5\n```",
            "There are 2 products priced above $5: Widget and Gadget.",
        ])
        engine = TalkDB(llm=llm, vectorstore=FakeVectorStore(), database=sample_db)
        result = engine.ask("Which products cost more than 5?")

        assert isinstance(result, QueryResult)
        assert result.sql == "SELECT * FROM products WHERE price > 5"
        assert len(result.dataframe) == 2

    def test_should_maintain_conversation_history(
        self, sample_db: SQLiteDatabase
    ) -> None:
        llm = FakeLLM(responses=[
            "```sql\nSELECT * FROM products\n```",
            "There are 3 products total.",
            "```sql\nSELECT * FROM products WHERE price > 20\n```",
            "Only Gadget costs more than $20.",
        ])
        engine = TalkDB(llm=llm, vectorstore=FakeVectorStore(), database=sample_db)

        engine.ask("Show all products")
        assert len(engine._history) == 1

        engine.ask("Which of those cost more than 20?")
        assert len(engine._history) == 2

    def test_should_limit_history_to_max_turns(
        self, sample_db: SQLiteDatabase
    ) -> None:
        responses = []
        for _ in range(7):
            responses.append("```sql\nSELECT COUNT(*) as cnt FROM products\n```")
            responses.append("There are 3 products.")

        llm = FakeLLM(responses=responses)
        engine = TalkDB(llm=llm, vectorstore=FakeVectorStore(), database=sample_db, max_history_turns=3)

        for i in range(7):
            engine.ask(f"Question {i}")

        assert len(engine._history) == 3

    def test_should_clear_history(self, sample_db: SQLiteDatabase) -> None:
        llm = FakeLLM(responses=[
            "```sql\nSELECT * FROM products\n```",
            "All products.",
        ])
        engine = TalkDB(llm=llm, vectorstore=FakeVectorStore(), database=sample_db)

        engine.ask("Show all products")
        assert len(engine._history) == 1

        engine.clear_history()
        assert len(engine._history) == 0


# ── Caching tests ─────────────────────────────────────────


class TestQueryCache:
    def test_should_return_none_on_cache_miss(self) -> None:
        cache = _QueryCache(max_size=10)
        assert cache.get("unknown question") is None

    def test_should_return_cached_result_on_hit(self) -> None:
        cache = _QueryCache(max_size=10)
        result = QueryResult(question="test", sql="SELECT 1", dataframe=pd.DataFrame(), answer="one")
        cache.put("test", result)
        assert cache.get("test") is result

    def test_should_be_case_insensitive(self) -> None:
        cache = _QueryCache(max_size=10)
        result = QueryResult(question="Test", sql="SELECT 1", dataframe=pd.DataFrame(), answer="one")
        cache.put("Test", result)
        assert cache.get("test") is result
        assert cache.get("TEST") is result

    def test_should_evict_oldest_when_full(self) -> None:
        cache = _QueryCache(max_size=2)
        r1 = QueryResult(question="q1", sql="SELECT 1", dataframe=pd.DataFrame(), answer="a1")
        r2 = QueryResult(question="q2", sql="SELECT 2", dataframe=pd.DataFrame(), answer="a2")
        r3 = QueryResult(question="q3", sql="SELECT 3", dataframe=pd.DataFrame(), answer="a3")
        cache.put("q1", r1)
        cache.put("q2", r2)
        cache.put("q3", r3)

        assert cache.get("q1") is None
        assert cache.get("q2") is r2
        assert cache.get("q3") is r3

    def test_should_clear_all_entries(self) -> None:
        cache = _QueryCache(max_size=10)
        result = QueryResult(question="test", sql="SELECT 1", dataframe=pd.DataFrame(), answer="one")
        cache.put("test", result)
        cache.clear()
        assert cache.get("test") is None


class TestTalkDBCaching:
    def test_should_serve_from_cache_on_repeated_question(
        self, sample_db: SQLiteDatabase
    ) -> None:
        llm = FakeLLM(responses=[
            "```sql\nSELECT COUNT(*) as cnt FROM products\n```",
            "There are 3 products.",
        ])
        engine = TalkDB(llm=llm, vectorstore=FakeVectorStore(), database=sample_db)

        result1 = engine.ask("How many products?")
        engine.clear_history()
        result2 = engine.ask("How many products?")

        assert result1.sql == result2.sql
        assert result1.answer == result2.answer
        assert llm._call_index == 2  # LLM was only called for the first ask

    def test_should_clear_cache(self, sample_db: SQLiteDatabase) -> None:
        llm = FakeLLM(responses=[
            "```sql\nSELECT COUNT(*) as cnt FROM products\n```",
            "There are 3 products.",
            "```sql\nSELECT COUNT(*) as cnt FROM products\n```",
            "There are 3 products.",
        ])
        engine = TalkDB(llm=llm, vectorstore=FakeVectorStore(), database=sample_db)

        engine.ask("How many products?")
        engine.clear_history()
        engine.clear_cache()
        engine.ask("How many products?")

        assert llm._call_index == 4  # LLM called twice (cache was cleared)


# ── Explain tests ─────────────────────────────────────────


class TestTalkDBExplain:
    def test_should_include_explanation_when_requested(
        self, sample_db: SQLiteDatabase
    ) -> None:
        llm = FakeLLM(responses=[
            "```sql\nSELECT * FROM products\n```",
            "All products listed.",
            "This query selects all columns from the products table.",
        ])
        engine = TalkDB(llm=llm, vectorstore=FakeVectorStore(), database=sample_db)
        result = engine.ask("Show all products", explain=True)

        assert result.explanation is not None
        assert "products" in result.explanation

    def test_should_not_include_explanation_by_default(
        self, sample_db: SQLiteDatabase
    ) -> None:
        llm = FakeLLM(responses=[
            "```sql\nSELECT * FROM products\n```",
            "All products.",
        ])
        engine = TalkDB(llm=llm, vectorstore=FakeVectorStore(), database=sample_db)
        result = engine.ask("Show all products")

        assert result.explanation is None

    def test_should_explain_standalone_sql(
        self, sample_db: SQLiteDatabase
    ) -> None:
        llm = FakeLLM(responses=[
            "This query counts all rows in products.",
        ])
        engine = TalkDB(llm=llm, vectorstore=FakeVectorStore(), database=sample_db)
        explanation = engine.explain("SELECT COUNT(*) FROM products")

        assert "products" in explanation


# ── Indexing tests ────────────────────────────────────────


class TestTalkDBIndexing:
    def test_should_index_all_ddls_from_database(
        self, sample_db: SQLiteDatabase
    ) -> None:
        vs = FakeVectorStore()
        engine = TalkDB(llm=FakeLLM(), vectorstore=vs, database=sample_db)

        count = engine.index_all_ddls()
        assert count == 1
        assert len(vs._ddls) == 1

    def test_should_bulk_index_examples(self, sample_db: SQLiteDatabase) -> None:
        vs = FakeVectorStore()
        engine = TalkDB(llm=FakeLLM(), vectorstore=vs, database=sample_db)

        examples = [
            {"question": "Show all products", "sql": "SELECT * FROM products"},
            {"question": "Count products", "sql": "SELECT COUNT(*) FROM products"},
            {"question": "invalid entry"},
        ]
        count = engine.index_examples_from_list(examples)
        assert count == 2
        assert len(vs._examples) == 2

    def test_should_generate_sql_only_without_executing(
        self, sample_db: SQLiteDatabase
    ) -> None:
        llm = FakeLLM(responses=[
            "```sql\nSELECT name FROM products\n```",
        ])
        engine = TalkDB(llm=llm, vectorstore=FakeVectorStore(), database=sample_db)
        sql = engine.generate_sql_only("List product names")

        assert sql == "SELECT name FROM products"
