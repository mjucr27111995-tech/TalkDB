import hashlib
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from talkdb.vectorstores.base import BaseVectorStore, RetrievalResult


def _deterministic_int_id(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:15], 16)


class QdrantVectorStore(BaseVectorStore):
    """Qdrant-backed vector store -- production-grade, supports local and server mode."""

    _COLLECTION_DDLS = "talkdb_ddls"
    _COLLECTION_EXAMPLES = "talkdb_examples"
    _COLLECTION_DOCS = "talkdb_docs"

    def __init__(
        self,
        url: str | None = None,
        path: str | None = ".talkdb_qdrant",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        """Initialize Qdrant store.

        Args:
            url: Qdrant server URL (e.g. 'http://localhost:6333'). If set, uses server mode.
            path: Local storage path. Used only when url is None (embedded mode).
            embedding_model: Sentence transformer model name.
        """
        self._embedder = SentenceTransformer(embedding_model)
        self._dimension = self._embedder.get_sentence_embedding_dimension()

        if url:
            self._client = QdrantClient(url=url)
        else:
            self._client = QdrantClient(path=path)

        self._ensure_collection(self._COLLECTION_DDLS)
        self._ensure_collection(self._COLLECTION_EXAMPLES)
        self._ensure_collection(self._COLLECTION_DOCS)

    def _ensure_collection(self, name: str) -> None:
        collections = [c.name for c in self._client.get_collections().collections]
        if name not in collections:
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=self._dimension,
                    distance=Distance.COSINE,
                ),
            )

    def _embed(self, text: str) -> list[float]:
        return self._embedder.encode(text).tolist()

    def _upsert(self, collection: str, doc_id: str, text: str, vector_text: str, metadata: dict) -> None:
        point_id = _deterministic_int_id(doc_id)
        self._client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=self._embed(vector_text),
                    payload={"content": text, **metadata},
                )
            ],
        )

    def _search(self, collection: str, question: str, top_k: int) -> list[RetrievalResult]:
        results = self._client.search(
            collection_name=collection,
            query_vector=self._embed(question),
            limit=top_k,
        )
        return [
            RetrievalResult(
                content=hit.payload.get("content", ""),
                score=hit.score,
                metadata={k: v for k, v in hit.payload.items() if k != "content"},
            )
            for hit in results
            if hit.payload
        ]

    def add_ddl(self, table_name: str, ddl: str) -> None:
        self._upsert(
            self._COLLECTION_DDLS,
            doc_id=table_name,
            text=ddl,
            vector_text=ddl,
            metadata={"table_name": table_name},
        )

    def add_example(self, question: str, sql: str) -> None:
        combined = f"Question: {question}\nSQL: {sql}"
        self._upsert(
            self._COLLECTION_EXAMPLES,
            doc_id=combined,
            text=combined,
            vector_text=question,
            metadata={"question": question, "sql": sql},
        )

    def add_documentation(self, doc: str) -> None:
        self._upsert(
            self._COLLECTION_DOCS,
            doc_id=doc,
            text=doc,
            vector_text=doc,
            metadata={"type": "documentation"},
        )

    def find_relevant_ddls(self, question: str, top_k: int = 5) -> list[RetrievalResult]:
        return self._search(self._COLLECTION_DDLS, question, top_k)

    def find_relevant_examples(self, question: str, top_k: int = 3) -> list[RetrievalResult]:
        return self._search(self._COLLECTION_EXAMPLES, question, top_k)

    def find_relevant_docs(self, question: str, top_k: int = 3) -> list[RetrievalResult]:
        return self._search(self._COLLECTION_DOCS, question, top_k)

    def clear(self) -> None:
        for name in (self._COLLECTION_DDLS, self._COLLECTION_EXAMPLES, self._COLLECTION_DOCS):
            self._client.delete_collection(name)
            self._ensure_collection(name)
