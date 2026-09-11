from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalResult:
    """A single item retrieved from the vector store."""
    content: str
    score: float
    metadata: dict


class BaseVectorStore(ABC):
    """Abstract base class for all vector store adapters."""

    @abstractmethod
    def add_ddl(self, table_name: str, ddl: str) -> None:
        """Index a table's DDL statement."""

    @abstractmethod
    def add_example(self, question: str, sql: str) -> None:
        """Index a question-SQL pair as a few-shot example."""

    @abstractmethod
    def add_documentation(self, doc: str) -> None:
        """Index a documentation snippet for additional context."""

    @abstractmethod
    def find_relevant_ddls(self, question: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve DDLs most relevant to the question."""

    @abstractmethod
    def find_relevant_examples(self, question: str, top_k: int = 3) -> list[RetrievalResult]:
        """Retrieve question-SQL examples most relevant to the question."""

    @abstractmethod
    def find_relevant_docs(self, question: str, top_k: int = 3) -> list[RetrievalResult]:
        """Retrieve documentation snippets most relevant to the question."""

    @abstractmethod
    def clear(self) -> None:
        """Remove all indexed data from the store."""
