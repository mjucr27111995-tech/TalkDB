import hashlib
import uuid
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from talkdb.vectorstores.base import BaseVectorStore, RetrievalResult


def _deterministic_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-backed vector store with persistent local storage."""

    _COLLECTION_DDLS = "talkdb_ddls"
    _COLLECTION_EXAMPLES = "talkdb_examples"
    _COLLECTION_DOCS = "talkdb_docs"

    def __init__(
        self,
        persist_dir: str = ".talkdb_chroma",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self._embedder = SentenceTransformer(embedding_model)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._ddls = self._client.get_or_create_collection(self._COLLECTION_DDLS)
        self._examples = self._client.get_or_create_collection(self._COLLECTION_EXAMPLES)
        self._docs = self._client.get_or_create_collection(self._COLLECTION_DOCS)

    def _embed(self, text: str) -> list[float]:
        return self._embedder.encode(text).tolist()

    def add_ddl(self, table_name: str, ddl: str) -> None:
        self._ddls.upsert(
            ids=[_deterministic_id(table_name)],
            embeddings=[self._embed(ddl)],
            documents=[ddl],
            metadatas=[{"table_name": table_name}],
        )

    def add_example(self, question: str, sql: str) -> None:
        combined = f"Question: {question}\nSQL: {sql}"
        self._examples.upsert(
            ids=[_deterministic_id(combined)],
            embeddings=[self._embed(question)],
            documents=[combined],
            metadatas=[{"question": question, "sql": sql}],
        )

    def add_documentation(self, doc: str) -> None:
        self._docs.upsert(
            ids=[_deterministic_id(doc)],
            embeddings=[self._embed(doc)],
            documents=[doc],
            metadatas=[{"type": "documentation"}],
        )

    def _query(
        self, collection: Any, question: str, top_k: int
    ) -> list[RetrievalResult]:
        if collection.count() == 0:
            return []
        results = collection.query(
            query_embeddings=[self._embed(question)],
            n_results=min(top_k, collection.count()),
        )
        output: list[RetrievalResult] = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                score = 1.0 - (results["distances"][0][i] if results.get("distances") else 0.0)
                meta = results["metadatas"][0][i] if results.get("metadatas") else {}
                output.append(RetrievalResult(content=doc, score=score, metadata=meta))
        return output

    def find_relevant_ddls(self, question: str, top_k: int = 5) -> list[RetrievalResult]:
        return self._query(self._ddls, question, top_k)

    def find_relevant_examples(self, question: str, top_k: int = 3) -> list[RetrievalResult]:
        return self._query(self._examples, question, top_k)

    def find_relevant_docs(self, question: str, top_k: int = 3) -> list[RetrievalResult]:
        return self._query(self._docs, question, top_k)

    def clear(self) -> None:
        self._client.delete_collection(self._COLLECTION_DDLS)
        self._client.delete_collection(self._COLLECTION_EXAMPLES)
        self._client.delete_collection(self._COLLECTION_DOCS)
        self._ddls = self._client.get_or_create_collection(self._COLLECTION_DDLS)
        self._examples = self._client.get_or_create_collection(self._COLLECTION_EXAMPLES)
        self._docs = self._client.get_or_create_collection(self._COLLECTION_DOCS)
