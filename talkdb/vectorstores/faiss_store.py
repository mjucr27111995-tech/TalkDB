import hashlib

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from talkdb.vectorstores.base import BaseVectorStore, RetrievalResult


class _FAISSCollection:
    """In-memory FAISS index with metadata tracking."""

    def __init__(self, dimension: int) -> None:
        self._index = faiss.IndexFlatL2(dimension)
        self._documents: list[str] = []
        self._metadatas: list[dict] = []
        self._id_to_pos: dict[str, int] = {}

    def upsert(
        self,
        doc_id: str,
        embedding: np.ndarray,
        document: str,
        metadata: dict,
    ) -> None:
        if doc_id in self._id_to_pos:
            pos = self._id_to_pos[doc_id]
            self._documents[pos] = document
            self._metadatas[pos] = metadata
            return
        self._index.add(embedding.reshape(1, -1))
        self._id_to_pos[doc_id] = len(self._documents)
        self._documents.append(document)
        self._metadatas.append(metadata)

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[RetrievalResult]:
        if self._index.ntotal == 0:
            return []
        k = min(top_k, self._index.ntotal)
        distances, indices = self._index.search(query_embedding.reshape(1, -1), k)
        results: list[RetrievalResult] = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            score = 1.0 / (1.0 + float(distances[0][i]))
            results.append(
                RetrievalResult(
                    content=self._documents[idx],
                    score=score,
                    metadata=self._metadatas[idx],
                )
            )
        return results

    def clear(self) -> None:
        self._index.reset()
        self._documents.clear()
        self._metadatas.clear()
        self._id_to_pos.clear()


def _deterministic_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class FAISSVectorStore(BaseVectorStore):
    """FAISS-backed in-memory vector store for fast similarity search."""

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2") -> None:
        self._embedder = SentenceTransformer(embedding_model)
        dimension = self._embedder.get_sentence_embedding_dimension()
        self._ddls = _FAISSCollection(dimension)
        self._examples = _FAISSCollection(dimension)
        self._docs = _FAISSCollection(dimension)

    def _embed(self, text: str) -> np.ndarray:
        return self._embedder.encode(text)

    def add_ddl(self, table_name: str, ddl: str) -> None:
        self._ddls.upsert(
            doc_id=_deterministic_id(table_name),
            embedding=self._embed(ddl),
            document=ddl,
            metadata={"table_name": table_name},
        )

    def add_example(self, question: str, sql: str) -> None:
        combined = f"Question: {question}\nSQL: {sql}"
        self._examples.upsert(
            doc_id=_deterministic_id(combined),
            embedding=self._embed(question),
            document=combined,
            metadata={"question": question, "sql": sql},
        )

    def add_documentation(self, doc: str) -> None:
        self._docs.upsert(
            doc_id=_deterministic_id(doc),
            embedding=self._embed(doc),
            document=doc,
            metadata={"type": "documentation"},
        )

    def find_relevant_ddls(self, question: str, top_k: int = 5) -> list[RetrievalResult]:
        return self._ddls.search(self._embed(question), top_k)

    def find_relevant_examples(self, question: str, top_k: int = 3) -> list[RetrievalResult]:
        return self._examples.search(self._embed(question), top_k)

    def find_relevant_docs(self, question: str, top_k: int = 3) -> list[RetrievalResult]:
        return self._docs.search(self._embed(question), top_k)

    def clear(self) -> None:
        self._ddls.clear()
        self._examples.clear()
        self._docs.clear()
