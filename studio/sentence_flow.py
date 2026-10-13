"""Sentence-first ASR editing: decide audio cuts before writing captions."""
from __future__ import annotations

from difflib import SequenceMatcher
import json
import re
from typing import Callable

from .bailian import ModelOutputError
from .captions import (MAX_CAPTION_CHARS, PUNCT, Word, extract_words,
                       has_estimated_timing, shorten_captions)
from .models import Person
from .prompts import PromptSet, load_prompts
from .cleanup import display_cleanup
from .strategy import Strategy
from .edit_plan import build_edit_plan
from .model_edits import partition_model_cuts, validate_model_cuts
from .caption_format_repair import repair_caption_lengths

def asr_sentences(transcription: dict, words: list[Word]) -> list[dict]:
    """Preserve ASR sentence boundaries and global word positions."""
    transcripts = transcription.get("transcripts") or []
    source = transcripts[0].get("sentences", []) if transcripts else []
    result: list[dict] = []
    cursor = 0
    for number, sentence in enumerate(source):
        originals = sentence.get("words")
        if isinstance(originals, list) and originals:
            size = sum(bool(str(item.get("text", "")).strip()) for item in originals)
        else:
            size = len(PUNCT.sub("", str(sentence.get("text", "")).strip()))
        if not size:
            continue
        first, last = cursor, cursor + size
        if last > len(words):
            raise ValueError("ASR 句子词索引超出转写结果")
        result.append({"id": number, "start_word": first, "end_word": last,
                       "start_ms": round(words[first].start * 1000),
                       "end_ms": round(words[last - 1].end * 1000),
                       "text": str(sentence.get("text") or
                                   "".join(word.text + word.punctuation for word in words[first:last]))})
        cursor = last
    if cursor != len(words) or not result:
        raise ValueError("ASR 句子与词级时间戳不一致")
    return result


def budget_sentences(sentences: list[dict], words: list[Word],
                     max_chars: int = 2400, max_words: int = 1200) -> list[dict]:
    """Split oversized ASR sentences along real word boundaries, retaining IDs."""
    result = []
    for sentence in sentences:
        first, last = sentence["start_word"], sentence["end_word"]
        if last - first <= max_words and len(sentence["text"]) <= max_chars:
            result.append(sentence)
            continue
        begin, count = first, 0
        for index in range(first, last):
            length = len(words[index].text)
            if length > max_chars:
                raise ValueError("ASR 单个词过长，缺少可定位的词级时间戳，无法可靠自动剪辑")
            if index > begin and (count + length > max_chars or index - begin >= max_words):
                result.append({**sentence, "asr_sentence_id": sentence["id"],
                               "start_word": begin, "end_word": index,
                               "start_ms": round(words[begin].start * 1000),
                               "end_ms": round(words[index - 1].end * 1000),
                               "text": "".join(w.text + w.punctuation for w in words[begin:index])})
                begin, count = index, 0
            count += length
        result.append({**sentence, "asr_sentence_id": sentence["id"],
                       "start_word": begin, "end_word": last,
                       "start_ms": round(words[begin].start * 1000),
                       "end_ms": round(words[last - 1].end * 1000),
                       "text": "".join(w.text + w.punctuation for w in words[begin:last])})
    return result


def sentence_groups(sentences: list[dict], max_chars: int = 2400,
                    max_seconds: float = 900, max_words: int = 1200) -> list[list[dict]]:
    """Review normal talking-head videos together; bound long-video chapters.

    For longer input, prefer existing sentence/long-pause boundaries near the
    request budget. The full transcript remains available to every chapter.
    These boundaries limit payload size, never decide which speech is deleted.
    """
    if not sentences:
        return []
    total_chars = sum(len(PUNCT.sub("", s["text"])) for s in sentences)
    duration = (sentences[-1]["end_ms"] - sentences[0]["start_ms"]) / 1000
    if (total_chars <= max_chars and duration <= max_seconds and
            sentences[-1]["end_word"] - sentences[0]["start_word"] <= max_words):
        return [sentences]
    groups: list[list[dict]] = []
    current: list[dict] = []
    count = 0
    for sentence in sentences:
        length = len(PUNCT.sub("", sentence["text"]))
        long_pause = (sentence["start_ms"] - current[-1]["end_ms"] >= 1200
                      if current else False)
        if current and (count + length > max_chars or
                        sentence["end_word"] - current[0]["start_word"] > max_words or
                        (count >= max_chars * .65 and long_pause) or
                        (sentence["end_ms"] - current[0]["start_ms"]) / 1000 > max_seconds):
            groups.append(current)
            current, count = [], 0
        current.append(sentence)
        count += length
    if current:
        groups.append(current)
    return groups


def _check_cuts(value: object, first: int, last: int, words: list[Word]) -> list[dict]:
    return validate_model_cuts(value, first, last, words)


def _model_input(context: dict) -> str:
    value = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    if len(value) > 200000:
        raise ValueError("完整文字上下文超过当前请求预算，请把超长视频分段制作；未发送文本模型请求")
    return value


def _retained_tokens(group: list[dict], words: list[Word], cuts: list[dict]) -> list[dict]:
    removed = {index for item in cuts if "word_index" not in item
               for index in range(int(item["start_word"]), int(item["end_word"]))}
    parts: dict[int, list[tuple[int, int]]] = {}
    for item in cuts:
        if "word_index" in item:
            parts.setdefault(int(item["word_index"]), []).append(
                (int(item["start_char"]), int(item["end_char"])))
    result: list[dict] = []
    for index in range(group[0]["start_word"], group[-1]["end_word"]):
        if index in removed:
            continue
        value = words[index].text
        for begin, end in sorted(parts.get(index, []), reverse=True):
            value = value[:begin] + value[end:]
        value = PUNCT.sub("", value)
        if value:
            result.append({"token": len(result), "word_index": index, "source_word_index": index, "text": value,
                           "punctuation": words[index].punctuation,
                           "start_ms": round(words[index].start * 1000),
                           "end_ms": round(words[index].end * 1000)})
    return result


def _caption_segments(response: dict, tokens: list[dict], words: list[Word],
                      max_cleanup_seconds: float = 1.0, *,
                      enforce_wording_policy: bool = True) -> list[dict]:
    if response.get("cuts"):
        raise ValueError("字幕阶段不能再修改音视频剪辑")
    source = response.get("segments")
    if not isinstance(source, list) or not source:
        raise ValueError("segments 为空")
    result: list[dict] = []
    cursor = 0
    for item in source:
        try:
            first, last = int(item["start_token"]), int(item["end_token"])
            value = PUNCT.sub("", re.sub(r"\s+", "", str(item["text"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("字幕缺少 token 索引或文字") from exc
        if first != cursor or not first < last <= len(tokens):
            raise ValueError("字幕 token 未按顺序覆盖保留语音")
        if not value:
            raise ValueError("字幕文字不能为空")
        selected_tokens = tokens[first:last]
        original = "".join(token["text"] for token in selected_tokens)
        raw_original = original
        declared = item.get("display_cleanup")
        cleanup, local_problem = [], ""
        try:
            if declared:
                original, cleanup = display_cleanup(declared, selected_tokens, max_cleanup_seconds)
        except ValueError:
            declared = None
        if not declared:
            # The model chooses the final wording. Mapping textual deletions to
            # character/time spans is bookkeeping, not a vocabulary judgment.
            omissions = [{"start_char": begin, "end_char": end, "kind": "model_cleanup",
                          "reason": "字幕模型结合上下文省略短促口头起音，原声保留"}
                         for kind, begin, end, _, _ in
                         SequenceMatcher(None, original, value, autojunk=False).get_opcodes()
                         if kind == "delete"]
            try:
                original, cleanup = display_cleanup(omissions, selected_tokens, max_cleanup_seconds)
            except ValueError as exc:
                # Keep this caption's speech instead of undoing other captions
                # in the group that the model successfully cleaned.
                original = value = raw_original
                cleanup = []
                local_problem = str(exc)
        # The caption model may correct ASR text, but must not summarize speech.
        if enforce_wording_policy and SequenceMatcher(None, original, value, autojunk=False).ratio() < 0.55:
            raise ValueError("字幕与保留语音差异过大，请保留原话")
        replacements = [{"from": original[begin:end], "to": value[a:b],
                         "reason": "文本模型修整或识别纠错"}
                        for kind, begin, end, a, b in
                        SequenceMatcher(None, original, value, autojunk=False).get_opcodes()
                        if kind in {"replace", "insert"}]
        first_word, last_word = tokens[first]["word_index"], tokens[last - 1]["word_index"] + 1
        result.append({"start_word": first_word, "end_word": last_word,
                       "start": round(words[first_word].start, 3),
                       "end": round(words[last_word - 1].end, 3), "text": value,
                       "source_word_indices": [token["word_index"] for token in selected_tokens],
                       "source_text": raw_original,
                       "model_wording": True,
                       **({"caption_fallback": True, "local_fallback_reason": local_problem} if local_problem else {}),
                       **({"model_text_changes": replacements} if replacements else {}),
                       **({"display_cleanup": cleanup} if cleanup else {})})
        cursor = last
    if cursor != len(tokens):
        raise ValueError("字幕漏掉了末尾保留语音")
    # Clause boundaries belong to the model; formatting/length can be repaired
    # separately without discarding a whole group's successful wording.
    return result


def _literal_captions(tokens: list[dict], words: list[Word]) -> list[dict]:
    """Keep spoken text if caption formatting repeatedly fails; never delete audio."""
    if not tokens:
        return []
    local_words = [Word(token["start_ms"] / 1000, token["end_ms"] / 1000,
                        token["text"], token["punctuation"]) for token in tokens]
    clauses: list[dict] = []
    first = 0
    for index, token in enumerate(tokens):
        next_gap = ((tokens[index + 1]["start_ms"] - token["end_ms"]) / 1000
                    if index + 1 < len(tokens) else 0)
        length = sum(len(part["text"]) for part in tokens[first:index + 1])
        boundary = bool(PUNCT.search(token["punctuation"])) or next_gap >= 0.5
        if index == len(tokens) - 1 or (boundary and length >= 4):
            clauses.append({"start_word": first, "end_word": index + 1,
                            "start": local_words[first].start, "end": local_words[index].end,
                            "text": "".join(part["text"] for part in tokens[first:index + 1])})
            first = index + 1
    try:
        pieces = shorten_captions(clauses, local_words)
    except (ValueError, IndexError):
        full_text = "".join(token["text"] for token in tokens)
        span = local_words[-1].end - local_words[0].start
        pieces = [{"text": full_text[position:position + MAX_CAPTION_CHARS],
                   "start": round(local_words[0].start + span * position / len(full_text), 3),
                   "end": round(local_words[0].start + span *
                                min(len(full_text), position + MAX_CAPTION_CHARS) / len(full_text), 3)}
                  for position in range(0, len(full_text), MAX_CAPTION_CHARS)]
    result: list[dict] = []
    for piece in pieces:
        if "start_word" not in piece:
            result.append({**piece, "caption_fallback": True, "timing_estimated": True})
            continue
        first, last = int(piece["start_word"]), int(piece["end_word"])
        source_start = tokens[first]["word_index"]
        source_end = tokens[last - 1]["word_index"] + 1
        result.append({"start_word": source_start, "end_word": source_end,
                       "source_word_indices": [token["word_index"] for token in tokens[first:last]],
                       "start": round(words[source_start].start, 3),
                       "end": round(words[source_end - 1].end, 3),
                       "text": piece["text"], "caption_fallback": True})
    return result


def build_sentence_project(transcription: dict, person: Person,
                           chat_json: Callable[[str, str], dict],
                           progress: Callable[[int, int], None] | None = None,
                           resume: dict | None = None,
                           checkpoint: Callable[[dict], None] | None = None,
                           plan_builder: Callable[[list[dict]], dict] | None = None,
                           prompts: PromptSet | None = None,
                           strategy: Strategy | None = None,
                           silences: list[tuple[float, float]] | None = None,
                           strict_model_failures: bool = False,
                           model_authored: bool = False,
                           model_plan_builder: Callable[[list[dict], list[dict]], dict] | None = None) -> dict:
    if model_authored:
        from .model_flow import build_model_project
        return build_model_project(transcription, person, chat_json, progress, resume,
                                   checkpoint, model_plan_builder, prompts, strategy, silences)
    prompts = prompts or load_prompts()
    strategy = strategy or Strategy()
    words = extract_words(transcription)
    sentences = budget_sentences(asr_sentences(transcription, words), words)
    transcripts = transcription.get("transcripts") or []
    full_transcript = str(transcripts[0].get("text") or "").strip() if transcripts else ""
    if not full_transcript:
        full_transcript = "".join(sentence["text"] for sentence in sentences)
    groups = sentence_groups(sentences)
    signature = [[group[0]["start_word"], group[-1]["end_word"]] for group in groups]
    state = {"signature": signature, "flow_version": 3, "edited": 0, "captioned": 0,
             "cuts": [], "captions": [], "changes": [], "uncertain": [], "fallback_groups": [],
             "caption_attempts": [], "edit_attempts": [], "edit_rejections": [],
             "caption_formatting_repairs": []}
    if isinstance(resume, dict) and resume.get("signature") == signature:
        if (isinstance(resume.get("edited"), int) and isinstance(resume.get("captioned"), int)
                and isinstance(resume.get("cuts"), list) and isinstance(resume.get("captions"), list)
                and 0 <= resume["captioned"] <= resume["edited"] <= len(groups)):
            state.update(resume)
    total = len(groups) * 2
    for number in range(state["edited"], len(groups)):
        group = groups[number]
        first, last = group[0]["start_word"], group[-1]["end_word"]
        group_first = sentences.index(group[0])
        group_last = sentences.index(group[-1]) + 1
        context = {"full_transcript": full_transcript,
                   "phase": "edit",
                   "context_scope": "whole_video" if len(groups) == 1 else "chapter_with_full_context",
                   "editable_word_ranges": [{"start_word": first, "end_word": last}],
                   "silence_gaps": [{"gap_id": f"S{i + 1}", "start_ms": round(a * 1000),
                                     "end_ms": round(b * 1000)} for i, (a, b) in enumerate(silences or [])],
                   "speaker": {"organization": person.organization,
                               "title": person.title, "name": person.name},
                   "target_sentences": group,
                   "min_audio_cut_seconds": strategy.min_model_cut_seconds,
                   "max_subtitle_cleanup_seconds": strategy.max_subtitle_cleanup_seconds,
                   "abandoned_start_constraints": {
                       "min_pause_seconds": strategy.abandoned_pause_seconds,
                       "max_spoken_seconds": strategy.max_abandoned_start_seconds},
                   "before": sentences[max(0, group_first - 2):group_first],
                   "after": sentences[group_last:group_last + 2],
                   "words": [{"i": index, "text": words[index].text,
                              "punctuation": words[index].punctuation,
                              "start_ms": round(words[index].start * 1000),
                              "end_ms": round(words[index].end * 1000),
                              "pause_after_ms": round(max(0, words[index + 1].start - words[index].end) * 1000)
                              if index + 1 < len(words) else 0,
                              "next_word_index": index + 1 if index + 1 < len(words) else None}
                             for index in range(first, last)]}
        error = ""
        previous = None
        response = None
        for attempt in range(2):
            decisions, issues = [], []
            try:
                response = chat_json(prompts.edit, _model_input(context)
                                     + ("\n上次剪切索引无效：" + error +
                                        "\n上次结果：" + json.dumps(previous, ensure_ascii=False)
                                        if error else ""))
            except ModelOutputError as exc:
                state["edit_attempts"].append({"group": number + 1, "attempt": attempt + 1,
                                               "response": None, "error": str(exc)})
                if strict_model_failures:
                    if checkpoint:
                        checkpoint(state)
                    raise ModelOutputError(f"剪辑模型调用未成功，尚未生成本次裁剪计划；ASR 结果已保存，可继续：{exc}") from exc
                state["uncertain"].append(f"第 {number + 1} 组剪辑分析失败，保留原声：{exc}")
                decisions = []
                break
            try:
                decisions, issues = partition_model_cuts(response.get("cuts"), first, last, words)
                if issues:
                    details = [{key: issue[key] for key in
                                ("proposal_position", "proposal_id", "reason", "word_indices", "conflicts")
                                if key in issue} for issue in issues]
                    raise ValueError("以下候选非法或互相冲突，请逐条修正，未涉及候选不必改动："
                                     + json.dumps(details, ensure_ascii=False))
                state["edit_attempts"].append({"group": number + 1, "attempt": attempt + 1,
                                               "response": response, "error": ""})
                break
            except ValueError as exc:
                error = str(exc)
                state["edit_attempts"].append({"group": number + 1, "attempt": attempt + 1,
                                               "response": response, "error": error,
                                               "rejected_proposals": issues,
                                               "independent_valid_proposals": decisions})
                previous = response
                if attempt == 1:
                    rejected = issues or [{"reason": error, "proposal": response.get("cuts")}]
                    state["edit_rejections"].extend({"group": number + 1, **issue} for issue in rejected)
                    state["uncertain"].append(
                        f"第 {number + 1} 组 {len(rejected)} 条剪辑建议未通过校验，相关原声保留；"
                        f"另有 {len(decisions)} 条独立合法建议继续进入时间校验")
                    break
        state["cuts"].extend(decisions)
        if isinstance(response, dict):
            state["uncertain"].extend(str(value) for value in response.get("uncertain", [])
                                       if str(value).strip())
        state["edited"] = number + 1
        if progress:
            progress(number + 1, total)
        if checkpoint:
            checkpoint(state)
    edit_plan = (plan_builder(state["cuts"]) if plan_builder else
                 build_edit_plan(words, state["cuts"], words[-1].end, [],
                                 estimated_timing=has_estimated_timing(transcription), strategy=strategy))
    edit_plan["rejected_model_proposals"] = state["edit_rejections"]
    applied_cuts = edit_plan["applied_proposals"]
    for number in range(state["captioned"], len(groups)):
        group = groups[number]
        tokens = _retained_tokens(group, words, applied_cuts)
        if tokens:
            kept_sentences = []
            for sentence in group:
                remaining = [token for token in tokens
                             if sentence["start_word"] <= token["word_index"] < sentence["end_word"]]
                if remaining:
                    kept_sentences.append({"id": sentence["id"],
                                           "start_token": remaining[0]["token"],
                                           "end_token": remaining[-1]["token"] + 1,
                                           "text": "".join(token["text"] for token in remaining),
                                           "start_ms": remaining[0]["start_ms"],
                                           "end_ms": remaining[-1]["end_ms"]})
            context = {"full_transcript": full_transcript,
                       "phase": "caption",
                       "applied_cuts": applied_cuts,
                       "caption_preferences": {"target_min_chars": 10, "target_max_chars": 14,
                                               "max_chars": MAX_CAPTION_CHARS, "semantic_boundaries_first": True},
                       "max_subtitle_cleanup_seconds": strategy.max_subtitle_cleanup_seconds,
                       "speaker": {"organization": person.organization,
                                   "title": person.title, "name": person.name},
                       "sentences": kept_sentences, "kept_tokens": tokens,
                       "preceding_sentence": sentences[sentences.index(group[0]) - 1]["text"]
                       if sentences.index(group[0]) else "",
                       "following_sentence": sentences[sentences.index(group[-1]) + 1]["text"]
                       if sentences.index(group[-1]) + 1 < len(sentences) else ""}
            error = ""
            previous = None
            pieces = None
            for attempt in range(2):
                try:
                    response = chat_json(prompts.caption, _model_input(context)
                                         + ("\n上次字幕无效：" + error +
                                            "\n上次结果：" + json.dumps(previous, ensure_ascii=False)
                                            if error else ""))
                except ModelOutputError as exc:
                    error = str(exc)
                    state["caption_attempts"].append({"group": number + 1, "attempt": attempt + 1,
                                                     "response": None, "error": error})
                    if strict_model_failures:
                        if checkpoint:
                            checkpoint(state)
                        raise ModelOutputError(f"字幕模型调用未成功；已完成的剪辑与 ASR 结果已保存，可继续：{exc}") from exc
                    break
                try:
                    pieces = _caption_segments(response, tokens, words, strategy.max_subtitle_cleanup_seconds)
                    too_long = [piece["text"] for piece in pieces if len(piece["text"]) > MAX_CAPTION_CHARS]
                    if too_long and attempt == 0:
                        raise ValueError(f"字幕超过 {MAX_CAPTION_CHARS} 字，请按自然意群重新拆分，不能拆专名或省略信息："
                                         + "；".join(too_long[:3]))
                    if too_long:
                        repaired = []
                        for piece in pieces:
                            if len(piece["text"]) <= MAX_CAPTION_CHARS:
                                repaired.append(piece)
                                continue
                            try:
                                local = repair_caption_lengths([piece], words,
                                    (person.organization, person.name, person.title))
                            except ValueError as exc:
                                indices = set(piece["source_word_indices"])
                                local = _literal_captions(
                                    [token for token in tokens if token["word_index"] in indices], words)
                                if number + 1 not in state["fallback_groups"]:
                                    state["fallback_groups"].append(number + 1)
                                state["uncertain"].append(
                                    f"第 {number + 1} 组源词 {piece['start_word']}–{piece['end_word']} "
                                    f"的单条字幕无法可靠拆分，局部保留原文：{exc}")
                            state["caption_formatting_repairs"].append({"group": number + 1,
                                "source_word_indices": piece["source_word_indices"],
                                "model_text": piece["text"], "render_texts": [p["text"] for p in local],
                                "literal_fallback": any(p.get("caption_fallback") for p in local)})
                            repaired.extend(local)
                        pieces = repaired
                    state["caption_attempts"].append({"group": number + 1, "attempt": attempt + 1,
                                                     "response": response, "error": ""})
                    state["changes"].extend(item for item in response.get("changes", [])
                                            if isinstance(item, dict))
                    state["uncertain"].extend(f"第 {number + 1} 组单条字幕保留原文：{piece['local_fallback_reason']}"
                                              for piece in pieces if piece.get("local_fallback_reason"))
                    state["uncertain"].extend(str(value) for value in response.get("uncertain", [])
                                               if str(value).strip())
                    break
                except ValueError as exc:
                    error = str(exc)
                    pieces = None
                    state["caption_attempts"].append({"group": number + 1, "attempt": attempt + 1,
                                                     "response": response, "error": error})
                    previous = response
            if pieces is None:
                pieces = _literal_captions(tokens, words)
                state["fallback_groups"].append(number + 1)
                state["uncertain"].append(f"第 {number + 1} 组字幕回退为 ASR 原文：{error}")
            state["captions"].extend(pieces)
        state["captioned"] = number + 1
        if progress:
            progress(len(groups) + number + 1, total)
        if checkpoint:
            checkpoint(state)
    removed = {i for cut in applied_cuts if "word_index" not in cut
               for i in range(cut["start_word"], cut["end_word"])}
    retained_source = [i for i in range(len(words)) if i not in removed]
    # ASR can put ':' in the word list for ratios. Its original index must
    # remain stable, but a punctuation-only item is not a caption speech token.
    retained = [token["word_index"] for token in _retained_tokens(
        [{"start_word": 0, "end_word": len(words)}], words, applied_cuts)]
    retained_set = set(retained)
    punctuation_only = [i for i in retained_source if i not in retained_set]
    assigned = [i for caption in state["captions"] for i in caption.get("source_word_indices", [])]
    coverage = {"original_words": len(words), "removed_words": len(removed),
                "retained_words": len(retained_source), "captionable_words": len(retained),
                "punctuation_only_word_indices": punctuation_only,
                "caption_count": len(state["captions"]),
                "retained_words_assigned_once_in_order": assigned == retained,
                "fallback_groups": state["fallback_groups"]}
    estimated_captions = any(caption.get("timing_estimated") for caption in state["captions"])
    coverage["estimated_caption_timing"] = estimated_captions
    if not estimated_captions and assigned != retained:
        raise ValueError("字幕未按原顺序完整覆盖实际保留的源词")
    return {"reviewed": True, "human_reviewed": False, "flow_version": 3,
            "context_scope": "whole_video" if len(groups) == 1 else "chapter_with_full_context",
            "coverage_audit": coverage,
            "estimated_timing": has_estimated_timing(transcription),
            "asr_sentences": sentences, "cut_proposals": applied_cuts,
            "model_cut_proposals": state["cuts"], "edit_plan": edit_plan,
            "captions": state["captions"], "changes": state["changes"],
            "uncertain": state["uncertain"], "caption_fallback_groups": state["fallback_groups"],
            "rejected_model_proposals": state["edit_rejections"],
            "caption_formatting_repairs": state["caption_formatting_repairs"],
            "caption_attempts": state["caption_attempts"], "edit_attempts": state["edit_attempts"]}
