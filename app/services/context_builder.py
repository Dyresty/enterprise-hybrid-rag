from app import config


class ContextBuilder:
    """Turns reranked child chunks into numbered sources for the prompt.

    Children sharing a parent are merged, and the parent text is sent instead of the child,
    so the LLM gets full context while retrieval stayed precise.
    """

    def build(self, documents, max_chars=None):
        max_chars = max_chars or config.MAX_CONTEXT_CHARS
        sources, seen, used = [], set(), 0
        for doc in documents:
            if doc["parent_id"] in seen:
                continue
            text = doc.get("parent_text") or doc["text"]
            if sources and used + len(text) > max_chars:
                break
            seen.add(doc["parent_id"])
            used += len(text)
            sources.append({**doc, "n": len(sources) + 1, "context_text": text})

        context = "\n\n".join(
            f"[{s['n']}] {s.get('title') or s['filename']} (page {s['page']})\n{s['context_text']}"
            for s in sources
        )
        return context, sources
