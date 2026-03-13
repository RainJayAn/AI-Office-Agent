import math
from functools import lru_cache
from pathlib import Path

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import DashScopeRerank
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from app.core.config import get_settings
from app.core.path import get_vector_db_dir


def get_vectorstore() -> Chroma | None:
    settings = get_settings()
    persist_directory = _get_store_path()
    if not _has_vector_store_data(persist_directory):
        return None

    return Chroma(
        collection_name=settings.CHROMA_COLLECTION_NAME,
        persist_directory=str(persist_directory),
        embedding_function=get_embedding_model(),
    )


@lru_cache(maxsize=1)
def get_embedding_model():
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_base_retriever(vectorstore: Chroma, top_k: int):
    return vectorstore.as_retriever(search_kwargs={"k": top_k})


def get_reranker(top_n: int) -> DashScopeRerank:
    settings = get_settings()
    return DashScopeRerank(
        api_key=settings.API_KEY,
        model=settings.RAG_RERANK_MODEL,
        top_n=top_n,
    )


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return []

    settings = get_settings()
    effective_top_k = top_k or settings.RAG_DEFAULT_TOP_K
    fetch_k = max(
        settings.RAG_BASE_RETRIEVAL_K,
        effective_top_k * settings.RAG_RETRIEVAL_MULTIPLIER,
        effective_top_k,
    )
    base_retriever = get_base_retriever(vectorstore=vectorstore, top_k=fetch_k)
    score_lookup = _build_vector_score_lookup(vectorstore=vectorstore, query=query, fetch_k=fetch_k)

    if settings.API_KEY:
        try:
            compression_retriever = ContextualCompressionRetriever(
                base_retriever=base_retriever,
                base_compressor=get_reranker(top_n=effective_top_k),
            )
            reranked_docs = compression_retriever.invoke(query)
            return _format_dashscope_reranked_documents(
                documents=reranked_docs,
                score_lookup=score_lookup,
            )
        except Exception:
            pass

    candidate_docs = base_retriever.invoke(query)
    return _fallback_local_rerank(
        query=query,
        documents=candidate_docs,
        score_lookup=score_lookup,
        top_k=effective_top_k,
    )


def _build_vector_score_lookup(vectorstore: Chroma, query: str, fetch_k: int) -> dict[str, float]:
    lookup: dict[str, float] = {}
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=fetch_k)
    for document, raw_score in docs_with_scores:
        lookup[_document_key(document)] = _normalize_vector_score(raw_score)
    return lookup


def _format_dashscope_reranked_documents(
    *,
    documents: list[Document],
    score_lookup: dict[str, float],
) -> list[dict]:
    results: list[dict] = []
    for document in documents:
        vector_score = score_lookup.get(_document_key(document), 0.0)
        rerank_score = float(document.metadata.get("relevance_score", 0.0))
        results.append(
            {
                "chunk_id": document.metadata.get("chunk_id", ""),
                "source": document.metadata.get("source", ""),
                "page": document.metadata.get("page"),
                "file_type": document.metadata.get("file_type", ""),
                "content": document.page_content,
                "vector_score": vector_score,
                "rerank_score": rerank_score,
                "score": rerank_score,
            }
        )
    return results


def _fallback_local_rerank(
    *,
    query: str,
    documents: list[Document],
    score_lookup: dict[str, float],
    top_k: int,
) -> list[dict]:
    if not documents:
        return []

    embeddings = get_embedding_model()
    query_embedding = embeddings.embed_query(query)
    doc_embeddings = embeddings.embed_documents([document.page_content for document in documents])

    reranked: list[dict] = []
    for document, doc_embedding in zip(documents, doc_embeddings):
        vector_score = score_lookup.get(_document_key(document), 0.0)
        rerank_score = _cosine_similarity(query_embedding, doc_embedding)
        final_score = (vector_score * 0.4) + (rerank_score * 0.6)
        reranked.append(
            {
                "chunk_id": document.metadata.get("chunk_id", ""),
                "source": document.metadata.get("source", ""),
                "page": document.metadata.get("page"),
                "file_type": document.metadata.get("file_type", ""),
                "content": document.page_content,
                "vector_score": vector_score,
                "rerank_score": rerank_score,
                "score": final_score,
            }
        )

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def _document_key(document: Document) -> str:
    return "::".join(
        [
            document.metadata.get("chunk_id", ""),
            document.metadata.get("source", ""),
            document.page_content[:120],
        ]
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _normalize_vector_score(raw_score: float) -> float:
    return 1.0 / (1.0 + abs(float(raw_score)))


def _get_store_path() -> Path:
    settings = get_settings()
    return get_vector_db_dir(settings.VECTOR_DB_PATH)


def _has_vector_store_data(path: Path) -> bool:
    if not path.exists():
        return False
    return any(path.iterdir())
