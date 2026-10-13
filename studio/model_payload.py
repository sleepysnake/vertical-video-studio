"""Compact long word/timestamp inputs without losing global source indices."""
from __future__ import annotations

import json


def compact_model_input(value: str) -> str:
    # Preserve repair messages after the original context JSON. They contain
    # the validation error and previous model response, not another context.
    decoder = json.JSONDecoder()
    try:
        context, end = decoder.raw_decode(value)
    except (ValueError, TypeError):
        return value
    if not isinstance(context, dict):
        return value
    words = context.get("words")
    tokens = context.get("kept_tokens")
    if max(len(words or []), len(tokens or [])) < 200:
        return value
    context = dict(context)
    if isinstance(words, list):
        context["word_table"] = {
            "columns": ["i", "text", "start_ms", "end_ms"],
            "rows": [[w["i"], w["text"], w["start_ms"], w["end_ms"]] for w in words],
        }
        del context["words"]
    if isinstance(tokens, list):
        context["token_table"] = {
            "columns": ["token", "word_index", "text", "duration_ms"],
            "rows": [[t["token"], t["word_index"], t["text"], t["end_ms"] - t["start_ms"]]
                     for t in tokens],
        }
        del context["kept_tokens"]
    if isinstance(context.get("applied_cuts"), list):
        context["applied_cuts"] = [{key: cut[key] for key in
                                   ("kind", "start_word", "end_word") if key in cut}
                                  for cut in context["applied_cuts"]]
    return json.dumps(context, ensure_ascii=False, separators=(",", ":")) + value[end:]
