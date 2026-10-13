"""Anchor caption text to ASR words and persist the source-to-output timeline."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from difflib import SequenceMatcher
from pathlib import Path

from .captions import PUNCT, Word, chunks
from .edits import keep_intervals, retime_captions


def _fully_removed(word: Word, cuts: list[dict]) -> bool:
    return any(float(cut["start"]) <= word.start + 0.02
               and float(cut["end"]) >= word.end - 0.02 for cut in cuts)


def align_captions(captions: list[dict], words: list[Word], cuts: list[dict]) -> list[dict]:
    """Correct model word-index drift using one monotonic text alignment per ASR chunk."""
    result = [dict(item) for item in captions]
    for caption in result:
        indices = caption.get("source_word_indices")
        if indices is None:
            continue
        if (not isinstance(indices, list) or not indices or
                any(type(i) is not int or not 0 <= i < len(words) for i in indices) or
                indices != sorted(set(indices)) or
                any(min(words[i].end, float(cut["end"])) -
                    max(words[i].start, float(cut["start"])) > 1e-6
                    for i in indices for cut in cuts)):
            raise ValueError("字幕的保留源词索引无效")
        # Token coverage was established before wording cleanup. Matching the
        # cleaned text again can incorrectly move a valid boundary to a later
        # repeated phrase, or drop the short hesitation's original duration.
        caption.update({"start_word": indices[0], "end_word": indices[-1] + 1,
                        "start": round(words[indices[0]].start, 3),
                        "end": round(words[indices[-1]].end, 3),
                        "alignment_method": "retained_source_tokens"})
    for section_start, section_end in chunks(words):
        positions = [position for position, item in enumerate(result)
                     if "source_word_indices" not in item and
                     section_start <= int(item.get("start_word", -1)) < section_end]
        kept = [(index, words[index]) for index in range(section_start, section_end)
                if not _fully_removed(words[index], cuts)]
        if not positions or len(kept) < len(positions):
            continue
        source = "".join(PUNCT.sub("", word.text) for _, word in kept)
        display = "".join(result[position]["text"] for position in positions)
        if not source or not display:
            continue
        score = SequenceMatcher(None, source, display, autojunk=False).ratio()
        # Very low agreement means the model may have supplied unrelated text.
        # Keep the original timestamps and expose the uncertainty for diagnosis.
        if score < 0.42:
            for position in positions:
                result[position]["alignment_confidence"] = round(score, 3)
                result[position]["alignment_method"] = "model_indices_low_confidence"
            continue
        boundaries = [0]
        for _, word in kept:
            boundaries.append(boundaries[-1] + len(PUNCT.sub("", word.text)))
        matches = [block for block in SequenceMatcher(None, source, display,
                   autojunk=False).get_matching_blocks() if block.size]
        text_cursor = 0
        previous_end = 0
        for position in positions:
            caption = result[position]
            caption_start = text_cursor
            text_cursor += len(caption["text"])
            if caption.get("display_cleanup"):
                # Caption-only cleanup covers the original speech, including
                # the omitted hesitation. Text matching must not trim it away.
                caption["alignment_method"] = "retained_tokens_with_display_cleanup"
                previous_end = max(previous_end, bisect_left([i for i, _ in kept], caption["end_word"]))
                continue
            overlapping = [(block.a + max(0, caption_start - block.b),
                            block.a + min(block.size, text_cursor - block.b))
                           for block in matches
                           if block.b < text_cursor and block.b + block.size > caption_start]
            matching_chars = sum(end - start for start, end in overlapping)
            if matching_chars < max(1, len(caption["text"]) // 2):
                caption["alignment_confidence"] = round(
                    matching_chars / max(1, len(caption["text"])), 3)
                caption["alignment_method"] = "model_indices_low_confidence"
                continue
            word_start = max(previous_end,
                             bisect_right(boundaries, overlapping[0][0]) - 1)
            word_end = bisect_left(boundaries, overlapping[-1][1])
            if word_start >= word_end or word_end > len(kept):
                caption["alignment_method"] = "model_indices_low_confidence"
                caption["alignment_confidence"] = 0.0
                continue
            assigned = kept[word_start:word_end]
            original_text = "".join(PUNCT.sub("", word.text) for _, word in assigned)
            if "model_start_word" not in caption:
                caption["model_start_word"] = caption.get("start_word")
                caption["model_end_word"] = caption.get("end_word")
                caption["model_start"] = caption.get("start")
                caption["model_end"] = caption.get("end")
            caption.update({
                "start_word": assigned[0][0], "end_word": assigned[-1][0] + 1,
                "start": round(assigned[0][1].start, 3),
                "end": round(assigned[-1][1].end, 3),
                "alignment_confidence": round(SequenceMatcher(
                    None, original_text, caption["text"], autojunk=False).ratio(), 3),
                "alignment_method": "asr_text_sequence",
            })
            previous_end = word_end
    return result


def build_timeline(source: Path, output: Path, words: list[Word], captions: list[dict],
                   cuts: list[dict], duration: float) -> dict:
    """Save an auditable mapping between source words, edits and final captions."""
    segments: list[dict] = []
    cursor = 0.0
    for start, end in keep_intervals(cuts, duration):
        segments.append({"source_start": round(start, 3), "source_end": round(end, 3),
                         "output_start": round(cursor, 3),
                         "output_end": round(cursor + end - start, 3)})
        cursor += end - start
    rendered = retime_captions(captions, cuts)
    entries: list[dict] = []
    for number, caption in enumerate(rendered, 1):
        first, last = caption.get("start_word"), caption.get("end_word")
        source_words = words[first:last] if isinstance(first, int) and isinstance(last, int) else []
        word_entries = []
        if isinstance(first, int) and isinstance(last, int):
            for word_index in range(first, last):
                word = words[word_index]
                shifted = retime_captions([{"start": word.start, "end": word.end,
                                            "text": word.text}], cuts)
                word_entries.append({"index": word_index, "text": word.text,
                                     "source_start": round(word.start, 3),
                                     "source_end": round(word.end, 3),
                                     "output_start": shifted[0]["start"] if shifted else None,
                                     "output_end": shifted[0]["end"] if shifted else None,
                                     "removed": not bool(shifted)})
        cleanups = []
        for cleanup in caption.get("display_cleanup", []):
            shifted = retime_captions([{"start": cleanup["source_start"],
                                        "end": cleanup["source_end"], "text": cleanup["source_text"]}], cuts)
            cleanups.append({**cleanup, "output_start": shifted[0]["start"] if shifted else None,
                             "output_end": shifted[0]["end"] if shifted else None})
        entries.append({"index": number, "text": caption["text"],
                        "manual_edited": bool(caption.get("manual_edited")),
                        "source_start": caption["source_start"],
                        "source_end": caption["source_end"],
                        "output_start": caption["start"], "output_end": caption["end"],
                        "start_word": first, "end_word": last,
                        "source_word_indices": caption.get("source_word_indices"),
                        "asr_text": "".join(word.text for word in source_words),
                        "asr_words": word_entries,
                        "alignment_confidence": caption.get("alignment_confidence"),
                        "model_start_word": caption.get("model_start_word"),
                        "model_end_word": caption.get("model_end_word"),
                        "display_cleanup": cleanups})
    return {"version": 1, "source_video": str(source), "output_video": str(output),
            "source_duration": round(duration, 3), "output_duration": round(cursor, 3),
            "keep_segments": segments, "cuts": cuts, "captions": entries}
