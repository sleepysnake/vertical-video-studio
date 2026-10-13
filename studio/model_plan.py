"""Compile model-authored edits, checking only executable source boundaries.

The model owns what to remove.  This module maps its explicit choices to source
times and verifies that execution cannot remove any unselected spoken word.
It does not classify speech, select silences, or apply editing policy limits.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re

from .captions import Word
from .edits import keep_intervals


PLAN_VERSION = 6
GAP_ID = re.compile(r"S([1-9][0-9]*)\Z")


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label}必须是整数，不能使用布尔值、小数或字符串")
    return value


def _seconds(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}必须是有限时间数值")
    try:
        converted = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label}必须是有限时间数值") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{label}必须是有限时间数值")
    return converted


def _word_times(word: Word, index: int) -> tuple[float, float]:
    start = _seconds(word.start, f"源词 {index} 起始时间")
    end = _seconds(word.end, f"源词 {index} 结束时间")
    if not 0 <= start < end:
        raise ValueError(f"源词 {index} 时间区间无效")
    return start, end


def _indices(item: dict) -> list[int]:
    return ([item["word_index"]] if "word_index" in item else
            list(range(item["start_word"], item["end_word"])))


def validate_executable_cuts(value: object, first: int, last: int,
                             words: list[Word]) -> list[dict]:
    """Validate execution indices; descriptive model metadata is preserved.

    Replacement/resume fields explain the model's decision and cannot veto a
    different decision.  Character cuts cannot be estimated by evenly dividing
    a word: the current ASR contract supplies exact word times only.
    """
    if not isinstance(value, list):
        raise ValueError("cuts 必须是数组")
    if (type(first) is not int or type(last) is not int or
            not 0 <= first <= last <= len(words)):
        raise ValueError("当前句组词索引无效")
    result = []
    for position, original in enumerate(value, 1):
        if not isinstance(original, dict):
            raise ValueError(f"第 {position} 条剪切必须是对象")
        item = deepcopy(original)
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ValueError(f"第 {position} 条剪切缺少 reason")
        if "word_index" in item:
            if any(key in item for key in ("start_word", "end_word")):
                raise ValueError("剪切不能混用整词与字符定位索引")
            index = _integer(item.get("word_index"), "word_index")
            if not first <= index < last:
                raise ValueError("剪切词索引不属于当前句组")
            begin = _integer(item.get("start_char"), "start_char")
            end = _integer(item.get("end_char"), "end_char")
            if not 0 <= begin < end <= len(words[index].text):
                raise ValueError("字符剪切索引无效")
            if begin != 0 or end != len(words[index].text):
                raise ValueError("ASR 没有准确字符时间，不能均分估算切音；请选择整词裁剪或只修整字幕")
        else:
            if any(key in item for key in ("start_char", "end_char")):
                raise ValueError("字符定位缺少 word_index")
            begin = _integer(item.get("start_word"), "start_word")
            end = _integer(item.get("end_word"), "end_word")
            if not first <= begin < end <= last:
                raise ValueError("剪切词索引不属于当前句组")
        for index in _indices(item):
            _word_times(words[index], index)
        if "end_before_word" in item:
            following = _integer(item["end_before_word"], "end_before_word")
            exclusive_end = item["word_index"] + 1 if "word_index" in item else item["end_word"]
            if following != exclusive_end or not 0 <= following < len(words):
                raise ValueError("end_before_word 必须是 end_word 指向的紧邻接续词索引")
            _word_times(words[following], following)
            padding = _seconds(item.get("keep_before_ms", 120), "keep_before_ms")
            if padding < 0:
                raise ValueError("接续词前的保留时间不能小于零")
            last_end = max(words[index].end for index in _indices(item))
            if words[following].start - padding / 1000 < last_end:
                raise ValueError("显式气口裁剪终点不能早于最后删除词结束，请调整 keep_before_ms")
        result.append(item)
    return result


def _inward(start: float, end: float) -> tuple[float, float]:
    # Preserve microsecond source boundaries without crossing a retained word.
    start = math.ceil((start - 1e-12) * 1_000_000) / 1_000_000
    end = math.floor((end + 1e-12) * 1_000_000) / 1_000_000
    if end <= start:
        raise ValueError("剪切区间不足以映射为可执行的时间边界")
    return start, end


def _unoccupied(start: float, end: float,
                retained: list[tuple[int, Word]]) -> list[tuple[float, float]]:
    """Subtract every retained word interval from a confirmed quiet region."""
    result = []
    cursor = start
    blocked = sorted((max(start, word.start), min(end, word.end))
                     for _, word in retained if word.start < end and word.end > start)
    for before, after in blocked:
        if before > cursor:
            result.append((cursor, before))
        cursor = max(cursor, after)
    if cursor < end:
        result.append((cursor, end))
    return result


def compile_model_plan(words: list[Word], cuts: object, duration: float,
                       silences: list[tuple[float, float]], pause_compressions: object,
                       estimated_timing: bool = False, strategy=None) -> dict:
    """Execute every model choice that has valid source timing.

    S1, S2, ... refer to the supplied confirmed-silence list in its original
    order.  Pauses are never selected automatically.  Each selected quiet gap
    is intersected with the complement of retained speech before the model's
    requested lead/trail padding is applied (120 ms each by default).
    ``strategy`` is accepted for caller compatibility and supplies no vetoes.
    """
    duration = _seconds(duration, "源视频时长")
    if duration <= 0:
        raise ValueError("源视频时长必须大于零")
    checked = validate_executable_cuts(cuts, 0, len(words), words)
    raw_cuts = deepcopy(checked)
    # A full character range has exact *word* boundaries. Normalize it for
    # downstream caption/source-word coverage; never invent partial timings.
    for item in checked:
        if "word_index" in item:
            index = item.pop("word_index")
            item.pop("start_char")
            item.pop("end_char")
            item.update({"start_word": index, "end_word": index + 1})
    if not isinstance(pause_compressions, list):
        raise ValueError("pause_compressions 必须是数组")
    if estimated_timing and (checked or pause_compressions):
        raise ValueError("ASR 词时间为估算，无法执行准确剪切；请使用词级时间戳或只修整字幕")
    for index, word in enumerate(words):
        _word_times(word, index)
    selected = {index for item in checked for index in _indices(item)}
    retained = [(index, word) for index, word in enumerate(words) if index not in selected]
    candidates, decisions = [], []

    def append_candidate(start: float, end: float, source_text: str, reason: str,
                         source: str, proposal: dict, position: int) -> None:
        if not 0 <= start < end <= duration:
            raise ValueError("模型剪切时间超出源视频可执行范围")
        start, end = _inward(start, end)
        identifier = proposal.get("id") or f"{source}-{position}"
        span = {"start": start, "end": end, "source_text": source_text,
                "reason": reason, "decision_ids": [identifier]}
        candidates.append(span)
        decisions.append({"id": identifier, "source": source, **span,
                          "proposal": deepcopy(proposal), "action": "CUT", "need_review": False,
                          "policy_reason": "", "execution_verified": True})

    for position, item in enumerate(checked, 1):
        indices = _indices(item)
        start = min(words[index].start for index in indices)
        end = max(words[index].end for index in indices)
        if "end_before_word" in item:
            end = words[item["end_before_word"]].start - item.get("keep_before_ms", 120) / 1000
        append_candidate(start, end, "".join(words[index].text for index in indices),
                         item["reason"], "llm", item, position)

    applied_pauses = []
    for position, original in enumerate(pause_compressions, 1):
        if not isinstance(original, dict):
            raise ValueError("静音压缩选择必须是包含 gap_id 的对象")
        item = deepcopy(original)
        gap_id = item.get("gap_id")
        match = GAP_ID.fullmatch(gap_id) if isinstance(gap_id, str) else None
        if match is None or int(match.group(1)) > len(silences):
            raise ValueError(f"静音 gap_id 不存在：{gap_id}")
        gap = silences[int(match.group(1)) - 1]
        if not isinstance(gap, (tuple, list)) or len(gap) != 2:
            raise ValueError(f"静音候选 {gap_id} 时间区间无效")
        start, end = (_seconds(gap[0], f"{gap_id} 起始时间"),
                      _seconds(gap[1], f"{gap_id} 结束时间"))
        if not start < end:
            raise ValueError(f"静音候选 {gap_id} 时间区间无效")
        before_ms = _seconds(item.get("keep_before_ms", 120), "keep_before_ms")
        after_ms = _seconds(item.get("keep_after_ms", 120), "keep_after_ms")
        if before_ms < 0 or after_ms < 0:
            raise ValueError("静音压缩保留时间不能小于零")
        reason = item.get("reason", "模型选择压缩确认静音")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("静音压缩 reason 必须是非空文字")
        parts = _unoccupied(max(0.0, start), min(duration, end), retained)
        part_count = 0
        for part_start, part_end in parts:
            a, b = part_start + before_ms / 1000, part_end - after_ms / 1000
            if b <= a:
                continue
            append_candidate(a, b, "", reason, "llm:pause", item, position)
            decisions[-1]["gap_id"] = gap_id
            decisions[-1]["confirmed_quiet"] = {"start": start, "end": end}
            part_count += 1
        if not part_count:
            raise ValueError(f"静音候选 {gap_id} 未留下可压缩的确认空白，请调整保留时间或取消选择")
        applied_pauses.append(item)

    candidates.sort(key=lambda span: (span["start"], span["end"]))
    merged = []
    for span in candidates:
        if not merged or span["start"] > merged[-1]["end"]:
            merged.append(deepcopy(span))
        else:
            previous = merged[-1]
            previous["end"] = max(previous["end"], span["end"])
            previous["reason"] += "；" + span["reason"]
            previous["decision_ids"].extend(span["decision_ids"])
    for span in merged:
        intrusions = [index for index, word in retained
                      if span["start"] < word.end - 1e-9 and span["end"] > word.start + 1e-9]
        if intrusions:
            raise ValueError(f"执行剪切侵入未选择删除的源词：{intrusions[:8]}")
        removed = [index for index in sorted(selected)
                   if span["start"] < words[index].end and span["end"] > words[index].start]
        span["source_text"] = "".join(words[index].text for index in removed)
        span["removed_word_indices"] = removed
        span["retained_word_check"] = {"verified": True, "checked": len(retained), "intrusions": []}
    keep_intervals(merged, duration)
    return {"version": PLAN_VERSION, "mode": "model_authored", "source_duration": duration,
            "decisions": sorted(decisions, key=lambda item: (item["start"], item["end"])),
            "applied_proposals": checked, "applied_pause_compressions": applied_pauses,
            "cuts": merged, "review_count": 0,
            "raw_model_decisions": {"cuts": raw_cuts,
                                    "pause_compressions": deepcopy(pause_compressions)},
            "retained_word_audit": {"verified": True, "word_indices": [i for i, _ in retained],
                                    "checked": len(retained),
                                    "method": "all_unselected_asr_words_outside_explicit_model_cuts"}}
