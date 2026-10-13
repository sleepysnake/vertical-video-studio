"""Repair model caption layout while preserving wording and source-token coverage."""
from __future__ import annotations

from bisect import bisect_right
from copy import deepcopy
from difflib import SequenceMatcher

from .captions import MAX_CAPTION_CHARS, PUNCT, Word, shorten_captions


def _local_source(caption: dict, words: list[Word]) -> tuple[list[int], list[Word], str]:
    indices = caption.get("source_word_indices")
    if indices is None:
        try:
            first, last = int(caption["start_word"]), int(caption["end_word"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("长字幕缺少可定位的源词范围") from exc
        if not 0 <= first < last <= len(words):
            raise ValueError("长字幕源词范围无效")
        indices = [index for index in range(first, last) if PUNCT.sub("", words[index].text)]
    if (not isinstance(indices, list) or not indices
            or any(isinstance(index, bool) or not isinstance(index, int)
                   or not 0 <= index < len(words) for index in indices)
            or indices != sorted(set(indices))):
        raise ValueError("长字幕源词索引必须有效且按原顺序唯一覆盖")
    values = [PUNCT.sub("", words[index].text) for index in indices]
    if any(not value for value in values):
        raise ValueError("长字幕源词列表包含无文字的标点项")
    raw = "".join(values)
    source = caption.get("source_text", raw)
    if not isinstance(source, str) or not source:
        raise ValueError("长字幕源文字为空或类型无效")
    if source != raw:
        # A retained token can have a word-internal audio cut. Allocate only
        # retained source characters; never guess which token owns inserted or
        # corrected source_text (display corrections belong in caption.text).
        boundaries = [0]
        for value in values:
            boundaries.append(boundaries[-1] + len(value))
        retained = ["" for _ in indices]
        for kind, begin, end, _, _ in SequenceMatcher(None, raw, source, autojunk=False).get_opcodes():
            if kind == "delete":
                continue
            if kind != "equal":
                raise ValueError("长字幕源文字无法可靠对应源词，需单条重做")
            for position in range(begin, end):
                retained[bisect_right(boundaries, position) - 1] += raw[position]
        if any(not value for value in retained) or "".join(retained) != source:
            raise ValueError("长字幕源文字缺少已分配的源词，需单条重做")
        values = retained
    local_words = [Word(words[index].start, words[index].end, value, words[index].punctuation)
                   for index, value in zip(indices, values)]
    return indices, local_words, source


def _cleanup_for_piece(cleanup: object, source: str, char_clock: list[tuple[float, float]],
                       first: int, last: int, segment_id: object) -> list[dict]:
    if not cleanup:
        return []
    if not isinstance(cleanup, list):
        raise ValueError("长字幕 display_cleanup 必须是数组")
    result = []
    for item in cleanup:
        try:
            begin, end = int(item["start_char"]), int(item["end_char"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("长字幕修整缺少源字符范围") from exc
        if not 0 <= begin < end <= len(source):
            raise ValueError("长字幕修整源字符范围无效")
        clipped_begin, clipped_end = max(first, begin), min(last, end)
        if clipped_begin >= clipped_end:
            continue
        part = deepcopy(item)
        part.update({"start_char": clipped_begin - first, "end_char": clipped_end - first,
                     "source_text": source[clipped_begin:clipped_end],
                     "source_start": round(char_clock[clipped_begin][0], 3),
                     "source_end": round(char_clock[clipped_end - 1][1], 3)})
        if clipped_begin != begin or clipped_end != end:
            part["original_segment_id"] = segment_id
            part["original_source_char_range"] = [begin, end]
        result.append(part)
    return result


def repair_caption_lengths(captions: list[dict], words: list[Word],
                           protected: tuple[str, ...] = ()) -> list[dict]:
    """Split only overlong captions that already passed token/semantic validation.

    This does not assess omissions or rewrite model wording. Short captions are
    returned unchanged. Long captions must have reliable retained source-word
    mapping; a ValueError lets the caller retry or fall back for that caption
    alone rather than discard a whole group's model output.
    """
    repaired = []
    for number, caption in enumerate(captions):
        text = str(caption["text"])
        if len(text.replace(" ", "")) <= MAX_CAPTION_CHARS:
            repaired.append(caption)
            continue
        if caption.get("timing_estimated"):
            raise ValueError("长字幕只有估算时间，不能可靠按源词拆分")
        indices, local_words, source = _local_source(caption, words)
        parts = shorten_captions([{"start_word": 0, "end_word": len(local_words),
                                  "start": local_words[0].start, "end": local_words[-1].end,
                                  "text": text}], local_words, tuple(protected))
        if "".join(part["text"] for part in parts) != text:
            raise ValueError("字幕排版修整改变了模型文字")
        cursor = 0
        for part in parts:
            if (part.get("timing_estimated") or "start_word" not in part
                    or "end_word" not in part or part["start_word"] != cursor
                    or not cursor < part["end_word"] <= len(indices)):
                raise ValueError("长字幕无法使用真实源词时间拆分，需单条重做")
            if len(part["text"].replace(" ", "")) > MAX_CAPTION_CHARS:
                raise ValueError("长字幕无法在保护专名的同时缩短，需单条重做")
            cursor = part["end_word"]
        if cursor != len(indices):
            raise ValueError("字幕排版修整未完整覆盖源词")
        boundaries = [0]
        char_clock = []
        for word in local_words:
            boundaries.append(boundaries[-1] + len(word.text))
            span = word.end - word.start
            char_clock.extend((word.start + span * offset / len(word.text),
                               word.start + span * (offset + 1) / len(word.text))
                              for offset in range(len(word.text)))
        segment_id = caption.get("segment_id", caption.get("id", number + 1))
        original_range = [indices[0], indices[-1] + 1]
        for part_number, part in enumerate(parts):
            first, last = part["start_word"], part["end_word"]
            char_first, char_last = boundaries[first], boundaries[last]
            assigned = indices[first:last]
            item = deepcopy(caption)
            item.update({"text": part["text"], "start_word": assigned[0],
                         "end_word": assigned[-1] + 1, "source_word_indices": assigned,
                         "start": round(local_words[first].start, 3),
                         "end": round(local_words[last - 1].end, 3),
                         "source_text": source[char_first:char_last],
                         "length_split": True, "formatting_fallback": True,
                         "formatting_source_segment_id": segment_id,
                         "formatting_source_word_range": original_range})
            if "source_word_range" in item:
                item["source_word_range"] = [assigned[0], assigned[-1] + 1]
            cleanup = _cleanup_for_piece(caption.get("display_cleanup"), source, char_clock,
                                         char_first, char_last, segment_id)
            if cleanup:
                item["display_cleanup"] = cleanup
            else:
                item.pop("display_cleanup", None)
            # These annotations were generated for the original model segment,
            # and may span the split. Preserve them once with explicit scope.
            if caption.get("model_text_changes"):
                if part_number == 0:
                    item["model_text_changes_scope"] = {"segment_id": segment_id,
                                                        "source_word_range": original_range,
                                                        "scope": "original_segment"}
                else:
                    item.pop("model_text_changes", None)
                    item.pop("model_text_changes_scope", None)
            repaired.append(item)
    return repaired
