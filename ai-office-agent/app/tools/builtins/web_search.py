import re
from typing import Any

from app.core.config import get_settings


def web_search(query: str, max_results: int = 5) -> str:
    settings = get_settings()
    effective_max_results = max_results or settings.WEB_SEARCH_MAX_RESULTS
    rewritten_queries = rewrite_search_queries(query)
    ddgs = _create_ddgs()

    try:
        results = _search_with_rewrites(
            ddgs=ddgs,
            queries=rewritten_queries,
            max_results=effective_max_results,
        )
    finally:
        close = getattr(ddgs, "close", None)
        if callable(close):
            close()

    if not results:
        return "未检索到联网搜索结果。"

    formatted_results: list[str] = []
    for index, item in enumerate(results, start=1):
        title = str(item.get("title") or "无标题").strip()
        url = str(item.get("href") or item.get("url") or "").strip()
        snippet = str(item.get("body") or item.get("snippet") or "无摘要").strip()
        search_query = str(item.get("_query") or query).strip()
        formatted_results.append(
            "\n".join(
                [
                    f"结果 {index}",
                    f"检索词：{search_query}",
                    f"标题：{title}",
                    f"链接：{url or '无链接'}",
                    f"摘要：{snippet}",
                ]
            )
        )

    return "\n\n".join(formatted_results)


def build_web_search_tool() -> dict[str, Any]:
    return {
        "name": "web_search",
        "description": (
            "Search the public web for up-to-date information, news, current events, "
            "recent changes, and time-sensitive facts."
        ),
        "func": web_search,
    }


def rewrite_search_queries(query: str) -> list[str]:
    normalized = " ".join(query.strip().split())
    if not normalized:
        return []

    candidates: list[str] = [normalized]
    cleaned = _remove_chatty_suffixes(normalized)
    if cleaned and cleaned != normalized:
        candidates.append(cleaned)

    date_variant = _replace_cn_date_with_iso(cleaned or normalized)
    if date_variant and date_variant not in candidates:
        candidates.append(date_variant)

    english_variant = _expand_common_cn_keywords(cleaned or normalized)
    if english_variant and english_variant not in candidates:
        candidates.append(english_variant)

    return _dedupe_preserve_order(candidates)[:4]


def _search_with_rewrites(ddgs, queries: list[str], max_results: int) -> list[dict]:
    aggregated: list[dict] = []
    seen_keys: set[str] = set()

    for search_query in queries:
        try:
            results = list(ddgs.text(search_query, max_results=max_results))
        except Exception:
            continue

        for item in results:
            url = str(item.get("href") or item.get("url") or "").strip()
            title = str(item.get("title") or "").strip()
            key = url or title
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            aggregated.append({**item, "_query": search_query})
            if len(aggregated) >= max_results:
                return aggregated

    return aggregated


def _remove_chatty_suffixes(query: str) -> str:
    cleaned = re.sub(
        r"(这是真的吗|真的假的|是真的吗\??|对吗|是否属实)$",
        "",
        query,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^(请帮我查一下|请搜索|帮我搜索|查一下|搜索一下)\s*", "", cleaned)
    return " ".join(cleaned.strip(" ？。！!；;").split())


def _replace_cn_date_with_iso(query: str) -> str:
    def replace(match: re.Match[str]) -> str:
        year = match.group("year")
        month = int(match.group("month"))
        day = int(match.group("day"))
        return f"{year}-{month:02d}-{day:02d}"

    return re.sub(
        r"(?P<year>\d{4})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日",
        replace,
        query,
    )


def _expand_common_cn_keywords(query: str) -> str:
    replacements = {
        "融资": "funding financing",
        "宣布": "announced announcement",
        "新闻": "news",
        "最新": "latest",
        "发布": "release launched",
        "收购": "acquisition acquired",
        "财报": "earnings results",
        "价格": "price pricing",
        "模型": "model",
    }
    expanded = query
    changed = False
    for source, target in replacements.items():
        if source in expanded:
            expanded = expanded.replace(source, f"{source} {target}")
            changed = True
    return expanded if changed else ""


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _create_ddgs():
    try:
        from ddgs import DDGS
    except ImportError as exc:
        raise RuntimeError(
            "ddgs is not installed. Please install project dependencies."
        ) from exc

    return DDGS()
