from typing import Any

from app.core.config import get_settings
from app.rag.retriever import retrieve


def retrieve_docs(query: str, top_k: int = 3) -> str:
    settings = get_settings()
    docs = retrieve(query=query, top_k=top_k or settings.RAG_DEFAULT_TOP_K)
    if not docs:
        return "未检索到相关文档内容，请先导入本地 md/txt/pdf 文档。"

    return "\n\n".join(
        _format_doc(item)
        for item in docs
    )


def build_retrieve_docs_tool() -> dict[str, Any]:
    return {
        "name": "retrieve_docs",
        "description": "Retrieve relevant local document snippets from the RAG store.",
        "func": retrieve_docs,
    }


def _format_doc(item: dict) -> str:
    page = item.get("page")
    page_line = f"\n页码：{page}" if page is not None else ""
    return (
        f"来源：{item['source']}"
        f"{page_line}\n"
        f"片段：{item['content'][:240]}\n"
        f"向量召回分：{item.get('vector_score', 0):.4f}\n"
        f"重排分：{item.get('rerank_score', 0):.4f}"
    )
