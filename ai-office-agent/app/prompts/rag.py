def build_rag_answer_prompt(query: str, context: str) -> str:
    return (
        "你是一个办公助手，请仅依据提供的文档片段回答问题。"
        "如果文档信息不足，请明确说明。\n\n"
        f"问题：{query}\n\n"
        f"文档片段：\n{context}"
    )


def build_rag_fallback_answer(query: str, preview: str) -> str:
    return f"基于检索到的文档片段，和“{query}”最相关的内容如下：\n\n{preview}"
