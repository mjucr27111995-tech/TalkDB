import hashlib
import re
from collections import OrderedDict
from dataclasses import dataclass

import pandas as pd

from talkdb.databases.base import BaseDatabase
from talkdb.llms.base import BaseLLM
from talkdb.prompts.templates import (
    CHART_PROMPT,
    CONTEXT_PROMPT,
    CONVERSATION_CONTEXT,
    DOCS_SECTION,
    EXAMPLES_SECTION,
    EXPLAIN_SQL_PROMPT,
    RESPONSE_PROMPT,
    SYSTEM_PROMPT,
)
from talkdb.utils.errors import TalkDBError
from talkdb.utils.sql_utils import extract_sql, validate_sql_is_select
from talkdb.vectorstores.base import BaseVectorStore


@dataclass
class QueryResult:
    """Result of a natural language query against the database."""
    question: str
    sql: str
    dataframe: pd.DataFrame
    answer: str
    explanation: str | None = None
    chart: object | None = None


@dataclass
class _ConversationTurn:
    question: str
    sql: str
    answer: str


class _QueryCache:
    """LRU cache for question -> (sql, dataframe, answer) lookups."""

    def __init__(self, max_size: int = 100) -> None:
        self._max_size = max_size
        self._cache: OrderedDict[str, QueryResult] = OrderedDict()

    @staticmethod
    def _key(question: str) -> str:
        normalized = question.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, question: str) -> QueryResult | None:
        key = self._key(question)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, question: str, result: QueryResult) -> None:
        key = self._key(question)
        self._cache[key] = result
        self._cache.move_to_end(key)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        self._cache.clear()


class TalkDB:
    """Core orchestrator that converts natural language to SQL using RAG.

    Composes an LLM, a vector store, and a database adapter.
    Supports conversation memory, query caching, SQL explanation, and chart generation.
    """

    def __init__(
        self,
        llm: BaseLLM,
        vectorstore: BaseVectorStore,
        database: BaseDatabase,
        max_history_turns: int = 5,
        cache_size: int = 100,
    ) -> None:
        if llm is None:
            raise TalkDBError("TALKDB-VAL-005", "llm is required")
        if vectorstore is None:
            raise TalkDBError("TALKDB-VAL-005", "vectorstore is required")
        if database is None:
            raise TalkDBError("TALKDB-VAL-005", "database is required")

        self._llm = llm
        self._vectorstore = vectorstore
        self._database = database
        self._max_history = max_history_turns
        self._history: list[_ConversationTurn] = []
        self._cache = _QueryCache(max_size=cache_size)

    # ── Indexing ──────────────────────────────────────────────

    def index_ddl(self, table_name: str, ddl: str) -> None:
        """Index a single table DDL into the vector store."""
        self._vectorstore.add_ddl(table_name, ddl)

    def index_all_ddls(self) -> int:
        """Auto-index all table DDLs from the connected database.
        Returns the number of tables indexed."""
        ddls = self._database.get_all_ddls()
        for table_name, ddl in ddls.items():
            self._vectorstore.add_ddl(table_name, ddl)
        return len(ddls)

    def index_example(self, question: str, sql: str) -> None:
        """Index a question-SQL pair as a few-shot example."""
        self._vectorstore.add_example(question, sql)

    def index_examples_from_list(self, examples: list[dict[str, str]]) -> int:
        """Bulk-index question-SQL pairs. Each dict needs 'question' and 'sql' keys.
        Returns the number of examples indexed."""
        count = 0
        for ex in examples:
            q = ex.get("question", "")
            s = ex.get("sql", "")
            if q and s:
                self._vectorstore.add_example(q, s)
                count += 1
        return count

    def index_documentation(self, doc: str) -> None:
        """Index a documentation snippet for additional context."""
        self._vectorstore.add_documentation(doc)

    # ── Querying ──────────────────────────────────────────────

    def ask(
        self,
        question: str,
        explain: bool = False,
        visualize: bool = False,
    ) -> QueryResult:
        """Convert a natural language question to SQL, execute it, and return results.

        Args:
            question: Natural language question.
            explain: If True, include a plain-English explanation of the SQL.
            visualize: If True, generate a Plotly chart from the results.
        """
        cached = self._cache.get(question)
        if cached and not self._history:
            return cached

        sql = self._generate_sql(question)
        validate_sql_is_select(sql)
        df = self._database.execute(sql)
        answer = self._generate_answer(question, sql, df)

        explanation = None
        if explain:
            explanation = self._explain_sql(sql)

        chart = None
        if visualize and not df.empty:
            chart = self._generate_chart(question, sql, df)

        result = QueryResult(
            question=question,
            sql=sql,
            dataframe=df,
            answer=answer,
            explanation=explanation,
            chart=chart,
        )

        self._cache.put(question, result)

        self._history.append(_ConversationTurn(question=question, sql=sql, answer=answer))
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        return result

    def generate_sql_only(self, question: str) -> str:
        """Generate SQL without executing it. Useful for preview/review."""
        sql = self._generate_sql(question)
        validate_sql_is_select(sql)
        return sql

    def explain(self, sql: str) -> str:
        """Explain a SQL query in plain English."""
        return self._explain_sql(sql)

    def clear_history(self) -> None:
        """Reset conversation memory."""
        self._history.clear()

    def clear_cache(self) -> None:
        """Clear the query cache."""
        self._cache.clear()

    # ── Internal pipeline ─────────────────────────────────────

    def _generate_sql(self, question: str) -> str:
        dialect = self._database.get_dialect()

        relevant_ddls = self._vectorstore.find_relevant_ddls(question)
        relevant_examples = self._vectorstore.find_relevant_examples(question)
        relevant_docs = self._vectorstore.find_relevant_docs(question)

        ddl_context = "\n\n".join(r.content for r in relevant_ddls) or "No schema available."

        examples_text = ""
        if relevant_examples:
            examples_text = EXAMPLES_SECTION.format(
                examples="\n\n".join(r.content for r in relevant_examples)
            )

        docs_text = ""
        if relevant_docs:
            docs_text = DOCS_SECTION.format(
                docs="\n\n".join(r.content for r in relevant_docs)
            )

        history_text = ""
        if self._history:
            turns = [
                f"User: {t.question}\nSQL: {t.sql}\nAnswer: {t.answer}"
                for t in self._history
            ]
            history_text = CONVERSATION_CONTEXT.format(history="\n\n".join(turns))

        system = SYSTEM_PROMPT.format(dialect=dialect)
        user_prompt = CONTEXT_PROMPT.format(
            ddl_context=ddl_context,
            examples_section=examples_text,
            docs_section=docs_text,
            question=question,
        )

        if history_text:
            user_prompt = history_text + "\n\n" + user_prompt

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ]
        response = self._llm.generate_with_history(messages)
        return extract_sql(response)

    def _generate_answer(self, question: str, sql: str, df: pd.DataFrame) -> str:
        if df.empty:
            return "The query returned no results."
        results_preview = df.head(20).to_string(index=False)
        prompt = RESPONSE_PROMPT.format(
            sql=sql, results=results_preview, question=question
        )
        return self._llm.generate(prompt)

    def _explain_sql(self, sql: str) -> str:
        relevant_ddls = self._vectorstore.find_relevant_ddls(sql, top_k=3)
        ddl_context = "\n\n".join(r.content for r in relevant_ddls) or "No schema available."
        prompt = EXPLAIN_SQL_PROMPT.format(sql=sql, ddl_context=ddl_context)
        return self._llm.generate(prompt)

    def _generate_chart(self, question: str, sql: str, df: pd.DataFrame) -> object | None:
        prompt = CHART_PROMPT.format(
            question=question,
            sql=sql,
            columns=list(df.columns),
            preview=df.head(5).to_string(index=False),
        )
        response = self._llm.generate(prompt)
        return self._execute_chart_code(response, df)

    @staticmethod
    def _execute_chart_code(response: str, df: pd.DataFrame) -> object | None:
        code_match = re.search(r"```(?:python)?\s*\n?(.*?)```", response, re.DOTALL)
        if not code_match:
            return None
        code = code_match.group(1).strip()

        import plotly.express as px
        import plotly.graph_objects as go

        safe_globals: dict = {
            "__builtins__": {"len": len, "list": list, "dict": dict, "str": str,
                             "int": int, "float": float, "range": range, "enumerate": enumerate,
                             "zip": zip, "sorted": sorted, "min": min, "max": max, "sum": sum,
                             "round": round, "abs": abs, "True": True, "False": False, "None": None},
            "pd": pd,
            "px": px,
            "go": go,
            "df": df,
        }
        local_vars: dict = {"df": df}
        try:
            exec(code, safe_globals, local_vars)  # noqa: S102
        except Exception:
            return None
        return local_vars.get("fig") or safe_globals.get("fig")
