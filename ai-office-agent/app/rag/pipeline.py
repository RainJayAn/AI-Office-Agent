from app.llm.factory import LLMFactory
from app.llm.router import TaskRequirements
from app.prompts import build_rag_answer_prompt, build_rag_fallback_answer
from app.rag.retriever import retrieve


def build_rag_chain():
    return run_rag


def run_rag(query: str, top_k: int = 3) -> dict:
    docs = retrieve(query=query, top_k=top_k)
    citations = format_citations(docs)

    if not docs:
        return {
            "answer": "未检索到相关文档内容，请先导入本地 md/txt/pdf 文档。",
            "docs": [],
            "citations": [],
        }

    answer = _generate_rag_answer(query=query, docs=docs)
    return {
        "answer": answer,
        "docs": docs,
        "citations": citations,
    }


def format_citations(docs: list[dict]) -> list[dict]:
    citations: list[dict] = []
    for item in docs:
        citations.append(
            {
                "source": item["source"],
                "chunk_id": item["chunk_id"],
                "page": item.get("page"),
                "file_type": item.get("file_type", ""),
                "score": item.get("score", 0),
                "vector_score": item.get("vector_score", 0),
                "rerank_score": item.get("rerank_score", 0),
            }
        )
    return citations


def _generate_rag_answer(query: str, docs: list[dict]) -> str:
    context = "\n\n".join(
        _format_context_line(index=index, doc=doc)
        for index, doc in enumerate(docs)
    )
    model = LLMFactory.get_chat_model(
        task_requirements=TaskRequirements(
            prompt_length=len(query) + len(context),
            message_count=1,
            tool_iteration_count=0,
            use_rag=True,
            stream=False,
            requires_reasoning=True,
        )
    )
    if model is not None:
        prompt = build_rag_answer_prompt(query=query, context=context)
        try:
            response = model.invoke(prompt)
            content = getattr(response, "content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
        except Exception:
            pass

    preview = "\n\n".join(
        _format_preview_line(index=index, doc=doc)
        for index, doc in enumerate(docs)
    )
    return build_rag_fallback_answer(query=query, preview=preview)


def _format_context_line(index: int, doc: dict) -> str:
    page = doc.get("page")
    page_suffix = f"（第 {page} 页）" if page is not None else ""
    return f"[{index + 1}] 来源：{doc['source']}{page_suffix}\n{doc['content']}"


def _format_preview_line(index: int, doc: dict) -> str:
    page = doc.get("page")
    page_suffix = f"（第 {page} 页）" if page is not None else ""
    return f"[{index + 1}] 来源：{doc['source']}{page_suffix}\n{doc['content'][:200]}"
