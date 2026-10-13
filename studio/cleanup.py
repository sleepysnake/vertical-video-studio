"""Bounded spoken disfluencies; no sentence rewriting or semantic deletion."""
from __future__ import annotations

from .captions import PUNCT, Word


FILLER_TEXTS = {"嗯", "呃", "额", "咦", "唔", "啊", "哎", "哦", "那个"}


def is_filler(value: str) -> bool:
    return value in FILLER_TEXTS or (0 < len(value) <= 4 and all(c in "嗯呃额咦唔啊哎哦" for c in value))


def is_short_stutter(text: str, first: int, last: int) -> bool:
    value = text[first:last]
    return (0 < len(value) <= 3 and not PUNCT.search(value) and
            (text[last:last + len(value)] == value or
             (first >= len(value) and text[first - len(value):first] == value)))


def basic_proposal_kind(words: list[Word], proposal: dict) -> str | None:
    # Punctuation is retained here: repeated words in separate clauses cannot
    # qualify as a short, contiguous stutter just by sharing their spelling.
    offsets = [0]
    for word in words:
        offsets.append(offsets[-1] + len(word.text) + len(word.punctuation))
    text = "".join(word.text + word.punctuation for word in words)
    if "word_index" in proposal:
        index = int(proposal["word_index"])
        first = offsets[index] + int(proposal["start_char"])
        last = offsets[index] + int(proposal["end_char"])
    else:
        begin, end = int(proposal["start_word"]), int(proposal["end_word"])
        first = offsets[begin]
        last = offsets[end - 1] + len(words[end - 1].text)
    if is_filler(text[first:last]):
        return "filler"
    if is_short_stutter(text, first, last):
        size = last - first
        repeat_first, repeat_last = ((first, last + size) if text[last:last + size] == text[first:last]
                                     else (first - size, last))
        for index in range(len(words) - 1):
            if repeat_first < offsets[index + 1] < repeat_last and words[index + 1].start - words[index].end > .15:
                return None
        return "stutter"
    return None


def display_cleanup(value: object, tokens: list[dict], max_seconds: float) -> tuple[str, list[dict]]:
    """Validate model-declared omissions structurally, without judging their wording."""
    original = "".join(token["text"] for token in tokens)
    if value is None:
        return original, []
    if not isinstance(value, list):
        raise ValueError("display_cleanup 必须是数组")
    clock = []
    for token in tokens:
        length = len(token["text"])
        for offset in range(length):
            start = token["start_ms"] / 1000
            duration = (token["end_ms"] - token["start_ms"]) / 1000
            clock.append((start + duration * offset / length,
                          start + duration * (offset + 1) / length, token))
    omissions = []
    occupied = set()
    for item in value:
        try:
            first, last = int(item["start_char"]), int(item["end_char"])
            kind, reason = str(item["kind"]), str(item["reason"]).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("字幕修整缺少字符索引、类型或原因") from exc
        if not 0 <= first < last <= len(original) or not reason:
            raise ValueError("字幕修整字符范围无效")
        if occupied.intersection(range(first, last)):
            raise ValueError("字幕修整范围重叠")
        removed = original[first:last]
        if kind not in {"stutter", "filler", "model_cleanup"}:
            raise ValueError("字幕修整类型无效")
        duration = clock[last - 1][1] - clock[first][0]
        if duration > max_seconds + .001:
            raise ValueError("字幕修整涉及较长原声，应保留文字或在剪辑阶段处理")
        # The model owns the semantic decision. ASR punctuation, vocabulary and
        # exact neighboring characters do not establish whether speech is a
        # hesitation or deliberate emphasis. Keep only index/time validation.
        occupied.update(range(first, last))
        omissions.append({"start_char": first, "end_char": last, "kind": kind, "reason": reason,
                          "source_text": removed, "source_start": round(clock[first][0], 3),
                          "source_end": round(clock[last - 1][1], 3), "audio_kept": True,
                          "timing_method": "asr_character_interpolation"})
    return "".join(c for i, c in enumerate(original) if i not in occupied), omissions
