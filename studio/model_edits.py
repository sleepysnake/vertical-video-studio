"""Structural validation for model-authored spoken-content edits.

The model decides whether an utterance was abandoned or replaced. This module
checks its declared source ranges; it never infers meaning from a reason string
or from a fixed vocabulary of Chinese fillers.
"""
from __future__ import annotations

from .captions import Word


SEMANTIC_CUT_KINDS = frozenset({"repair_restart", "repeated_take",
                                "abandoned_start", "isolated_filler"})
REPLACEMENT_CUT_KINDS = frozenset({"repair_restart", "repeated_take"})


def cut_words(item: dict) -> set[int]:
    return ({item["word_index"]} if "word_index" in item else
            set(range(item["start_word"], item["end_word"])))


def protected_words(item: dict) -> set[int]:
    replacement = item.get("replacement_range")
    result = (set(range(replacement["start_word"], replacement["end_word"]))
              if isinstance(replacement, dict) else set())
    if item.get("kind") == "abandoned_start":
        result.add(item.get("resume_word", item["end_word"]))
    return result


def _index(value: object) -> int:
    if type(value) is not int:
        raise ValueError("剪切索引必须是整数，不能使用字符串、小数或布尔值")
    return value


def validate_model_cuts(value: object, first: int, last: int,
                        words: list[Word]) -> list[dict]:
    """Validate global, half-open indices in a target range.

    Replacement/resume ranges may refer to a following group, but must remain
    untouched by every proposal in this result. Duration limits belong to the
    configurable edit policy rather than the model-response schema.
    """
    if not isinstance(value, list):
        raise ValueError("cuts 必须是数组")
    if not (type(first) is int and type(last) is int and
            0 <= first <= last <= len(words)):
        raise ValueError("当前句组词索引无效")
    result: list[dict] = []
    occupied: dict[int, list[tuple[int, int]]] = {}
    for original in value:
        if not isinstance(original, dict):
            raise ValueError("剪切区间必须是对象")
        item = dict(original)
        allowed = {"kind", "start_word", "end_word", "word_index", "start_char", "end_char",
                   "resume_word", "replacement_range", "reason", "joined_text", "id"}
        if set(item) - allowed:
            raise ValueError("剪切区间包含未知字段；时间戳必须由程序根据原始词索引计算")
        if not isinstance(item.get("reason"), str) or not item["reason"].strip():
            raise ValueError("剪切区间缺少原因")
        if "id" in item and (not isinstance(item["id"], str) or not item["id"].strip()):
            raise ValueError("剪切 id 必须是非空文字")
        kind = item.get("kind")
        if kind is not None and (not isinstance(kind, str) or kind not in SEMANTIC_CUT_KINDS):
            raise ValueError("未知剪切类型；ASR 纠错只修改字幕，不能剪掉原声")
        if "joined_text" in item and item["joined_text"] is not None and not isinstance(item["joined_text"], str):
            raise ValueError("joined_text 必须是文字")
        if "word_index" in item:
            if kind is not None or any(key in item for key in ("start_word", "end_word", "resume_word", "replacement_range")):
                raise ValueError("语义裁剪必须使用完整词索引，不能使用词内切点")
            index = _index(item.get("word_index"))
            begin, end = _index(item.get("start_char")), _index(item.get("end_char"))
            if not (first <= index < last and 0 <= begin < end <= len(words[index].text)):
                raise ValueError("词内剪切索引无效")
            ranges = {index: (begin, end)}
        else:
            begin, end = _index(item.get("start_word")), _index(item.get("end_word"))
            if not first <= begin < end <= last:
                raise ValueError("剪切词索引不属于当前句组")
            if any(key in item for key in ("start_char", "end_char")):
                raise ValueError("完整词剪切不能混用字符索引")
            ranges = {index: (0, len(words[index].text)) for index in range(begin, end)}
            replacement = item.get("replacement_range")
            if kind in REPLACEMENT_CUT_KINDS and replacement is None:
                raise ValueError("改口或重复拍摄裁剪必须指出后续完整保留的 replacement_range")
            if replacement is not None:
                if not isinstance(replacement, dict):
                    raise ValueError("replacement_range 必须是词索引对象")
                replacement_first = _index(replacement.get("start_word"))
                replacement_last = _index(replacement.get("end_word"))
                if not end <= replacement_first < replacement_last <= len(words):
                    raise ValueError("替代表达必须位于删除片段之后并使用有效完整词范围")
            if kind == "abandoned_start":
                resume = _index(item.get("resume_word", end))
                if resume != end or not 0 <= resume < len(words):
                    raise ValueError("废弃起句必须指向紧接（紧邻）的下一个完整保留词")
            elif item.get("resume_word") is not None:
                raise ValueError("resume_word 只适用于废弃起句")
        for index, (a, b) in ranges.items():
            if any(a < occupied_end and b > occupied_start
                   for occupied_start, occupied_end in occupied.get(index, [])):
                raise ValueError("剪切词或字符区间重叠")
            occupied.setdefault(index, []).append((a, b))
        result.append(item)
    changed = set(occupied)
    if any(protected_words(item) & changed for item in result):
        raise ValueError("接续词和替代表达必须在最终剪辑中完整保留，不能同时删除或裁剪")
    return result


def partition_model_cuts(value: object, first: int, last: int,
                         words: list[Word]) -> tuple[list[dict], list[dict]]:
    """Keep independent valid proposals and isolate every conflicting party.

    Positions in issue records are one-based positions in the supplied array;
    optional or repeated model IDs do not affect proposal identity.  A conflict
    never selects a winner by array order or attempts to infer which edit is
    semantically correct.  The model can repair the isolated proposals later.
    """
    # Validate the outer contract even when no candidate can be accepted.
    if not isinstance(value, list):
        raise ValueError("cuts 必须是数组")
    validate_model_cuts([], first, last, words)
    candidates: dict[int, dict] = {}
    ranges: dict[int, dict[int, tuple[int, int]]] = {}
    issues: dict[int, dict] = {}

    def identity(position: int) -> dict:
        original = value[position - 1]
        return {"proposal_position": position,
                "proposal_id": original.get("id") if isinstance(original, dict) else None,
                "proposal": original}

    for position, original in enumerate(value, 1):
        try:
            item = validate_model_cuts([original], first, last, words)[0]
        except ValueError as exc:
            issues[position] = {**identity(position), "reason": str(exc),
                                "word_indices": [], "conflicts": []}
            continue
        candidates[position] = item
        ranges[position] = (
            {item["word_index"]: (item["start_char"], item["end_char"])}
            if "word_index" in item else
            {index: (0, len(words[index].text))
             for index in range(item["start_word"], item["end_word"])})

    def record_conflict(left: int, right: int, details: dict, reason: str) -> None:
        for position, other in ((left, right), (right, left)):
            issue = issues.setdefault(position, {**identity(position), "reason": "",
                                                 "word_indices": [], "conflicts": []})
            issue["conflicts"].append({**details, "other_proposal_position": other,
                                       "other_proposal_id": candidates[other].get("id"),
                                       "reason": reason})
            issue["word_indices"] = sorted(set(issue["word_indices"]) | set(details["word_indices"]))
            issue["reason"] = "；".join(conflict["reason"] for conflict in issue["conflicts"])

    positions = list(candidates)
    for number, left in enumerate(positions):
        for right in positions[number + 1:]:
            overlap = []
            for index in sorted(ranges[left].keys() & ranges[right].keys()):
                begin = max(ranges[left][index][0], ranges[right][index][0])
                end = min(ranges[left][index][1], ranges[right][index][1])
                if begin < end:
                    overlap.append({"word_index": index, "start_char": begin, "end_char": end})
            if overlap:
                indices = [part["word_index"] for part in overlap]
                record_conflict(left, right,
                    {"kind": "deletion_overlap", "word_indices": indices, "char_ranges": overlap},
                    f"第 {left}、{right} 条删除词或字符区间重叠：词 {indices}")

            for protector, deleter in ((left, right), (right, left)):
                item = candidates[protector]
                replacement = item.get("replacement_range")
                protections = []
                if isinstance(replacement, dict):
                    protections.append(("replacement", set(range(replacement["start_word"],
                                                                replacement["end_word"]))))
                if item.get("kind") == "abandoned_start":
                    protections.append(("resume", {item.get("resume_word", item["end_word"])}))
                for protection, protected in protections:
                    indices = sorted(protected & ranges[deleter].keys())
                    if not indices:
                        continue
                    label = "替代表达" if protection == "replacement" else "接续词"
                    record_conflict(protector, deleter,
                        {"kind": "protected_word_conflict", "protection": protection,
                         "word_indices": indices, "protected_by_proposal_position": protector,
                         "deleted_by_proposal_position": deleter},
                        f"第 {protector} 条要求保留的{label}被第 {deleter} 条删除或裁剪：词 {indices}")

    safe = [candidates[position] for position in positions if position not in issues]
    # This is an invariant check, not a second opportunity to pick winners.
    safe = validate_model_cuts(safe, first, last, words)
    return safe, [issues[position] for position in sorted(issues)]
