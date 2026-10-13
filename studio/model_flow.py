"""Model draft -> independent semantic review -> captions -> executable plan.

Index and timeline validation cannot decide whether spoken content is useful.
The reviewed model JSON owns that decision; failed execution checks are sent
back to the model rather than silently replacing its plan with policy rules.
"""
from __future__ import annotations

from copy import deepcopy
import json

from .bailian import ModelOutputError
from .captions import MAX_CAPTION_CHARS, extract_words, has_estimated_timing
from .caption_format_repair import repair_caption_lengths
from .model_plan import compile_model_plan, validate_executable_cuts
from .prompts import load_prompts
from .sentence_flow import (asr_sentences, budget_sentences, sentence_groups,
                            _caption_segments, _literal_captions, _model_input,
                            _retained_tokens)
from .strategy import Strategy


def _plan_arrays(response, first, last, words):
    if not isinstance(response, dict):
        raise ValueError("模型计划必须是 JSON 对象")
    result = deepcopy(response)
    result["cuts"] = validate_executable_cuts(response.get("cuts"), first, last, words)
    for field in ("pause_compressions", "display_only_suggestions", "uncertain"):
        value = response.get(field, [])
        if not isinstance(value, list):
            raise ValueError(f"{field} 必须是数组")
        result[field] = deepcopy(value)
    return result


def build_model_project(transcription, person, chat_json, progress=None, resume=None,
                        checkpoint=None, plan_builder=None, prompts=None,
                        strategy=None, silences=None):
    prompts = prompts or load_prompts()
    if not prompts.review:
        raise ValueError("缺少语义复审提示词 review_prompt.txt，无法确认最终模型计划")
    strategy = strategy or Strategy()
    words = extract_words(transcription)
    sentences = budget_sentences(asr_sentences(transcription, words), words)
    groups = sentence_groups(sentences)
    transcripts = transcription.get("transcripts") or []
    full_text = str(transcripts[0].get("text") or "").strip() if transcripts else ""
    full_text = full_text or "".join(s["text"] for s in sentences)
    signature = [[g[0]["start_word"], g[-1]["end_word"]] for g in groups]
    state = {"signature": signature, "flow_version": 4, "mode": "model_authored",
             "edited": 0, "reviewed": 0, "captioned": 0, "draft_plans": [],
             "final_model_plans": [], "cuts": [], "pause_compressions": [],
             "display_only_suggestions": [], "captions": [], "changes": [],
             "uncertain": [], "fallback_groups": [], "edit_attempts": [],
             "review_attempts": [], "caption_attempts": [], "edit_rejections": [],
             "caption_formatting_repairs": []}
    if (isinstance(resume, dict) and resume.get("flow_version") == 4
            and resume.get("signature") == signature):
        counts = [resume.get(k) for k in ("captioned", "reviewed", "edited")]
        if (all(type(n) is int for n in counts)
                and 0 <= counts[0] <= counts[1] <= counts[2] <= len(groups)
                and isinstance(resume.get("draft_plans"), list)
                and len(resume["draft_plans"]) == counts[2]
                and isinstance(resume.get("final_model_plans"), list)
                and len(resume["final_model_plans"]) == counts[1]
                and isinstance(resume.get("captions"), list)):
            state.update(deepcopy(resume))
    total = len(groups) * 3

    def save(completed):
        if checkpoint:
            checkpoint(deepcopy(state))
        if progress:
            progress(completed, total)

    def compile_plan(cuts, pauses):
        if plan_builder:
            return plan_builder(cuts, pauses)
        return compile_model_plan(words, cuts, words[-1].end, silences or [], pauses,
                                  estimated_timing=has_estimated_timing(transcription),
                                  strategy=strategy)

    def context_for(number, phase):
        group = groups[number]
        first, last = signature[number]
        group_first = sentences.index(group[0])
        group_last = sentences.index(group[-1]) + 1
        return {"phase": phase, "full_transcript": full_text,
                "context_scope": "whole_video" if len(groups) == 1 else "chapter_with_full_context",
                "editable_word_ranges": [{"start_word": first, "end_word": last}],
                "speaker": {"organization": person.organization,
                            "title": person.title, "name": person.name},
                "target_sentences": group,
                "before": sentences[max(0, group_first - 2):group_first],
                "after": sentences[group_last:group_last + 2],
                "silence_gaps": [{"gap_id": f"S{i + 1}", "start_ms": round(a * 1000),
                                  "end_ms": round(b * 1000)}
                                 for i, (a, b) in enumerate(silences or [])],
                "words": [{"i": i, "text": words[i].text,
                           "punctuation": words[i].punctuation,
                           "start_ms": round(words[i].start * 1000),
                           "end_ms": round(words[i].end * 1000),
                           "pause_after_ms": round(max(0, words[i + 1].start - words[i].end) * 1000)
                           if i + 1 < len(words) else 0,
                           "next_word_index": i + 1 if i + 1 < len(words) else None}
                          for i in range(first, min(len(words), last + 1))]}

    def request_plan(number, phase, system, context, verify=None):
        error, previous = "", None
        attempts = state["edit_attempts" if phase == "edit" else "review_attempts"]
        label = "剪辑模型" if phase == "edit" else "语义复审模型"
        for attempt in range(2):
            try:
                response = chat_json(system, _model_input(context) + (
                    "\n上次结果无法执行，请保留语义判断并修正索引或时间边界：" + error
                    + "\n上次结果：" + json.dumps(previous, ensure_ascii=False) if error else ""))
            except ModelOutputError as exc:
                attempts.append({"group": number + 1, "attempt": attempt + 1,
                                 "response": None, "error": str(exc)})
                if checkpoint:
                    checkpoint(deepcopy(state))
                raise ModelOutputError(f"{label}调用未成功；已完成的阶段和 ASR 结果已保存，可继续：{exc}") from exc
            try:
                plan = _plan_arrays(response, *signature[number], words)
                if verify:
                    verify(plan)
                attempts.append({"group": number + 1, "attempt": attempt + 1,
                                 "response": response, "error": ""})
                return plan
            except ValueError as exc:
                error, previous = str(exc), response
                attempts.append({"group": number + 1, "attempt": attempt + 1,
                                 "response": response, "error": error})
        if checkpoint:
            checkpoint(deepcopy(state))
        raise ModelOutputError(f"{label}两次返回不可执行的计划，未替换为规则裁剪；可继续：{error}")

    for number in range(state["edited"], len(groups)):
        plan = request_plan(number, "edit", prompts.edit, context_for(number, "edit"))
        state["draft_plans"].append(plan)
        state["edited"] = number + 1
        save(number + 1)

    for number in range(state["reviewed"], len(groups)):
        context = context_for(number, "semantic_review")
        context["draft_plan"] = state["draft_plans"][number]
        context["other_chapter_drafts"] = [p for i, p in enumerate(state["draft_plans"]) if i != number]

        def verify(plan):
            compile_plan(state["cuts"] + plan["cuts"],
                         state["pause_compressions"] + plan["pause_compressions"])

        plan = request_plan(number, "semantic_review", prompts.review, context, verify)
        state["final_model_plans"].append(plan)
        state["cuts"].extend(plan["cuts"])
        state["pause_compressions"].extend(plan["pause_compressions"])
        state["display_only_suggestions"].extend(plan["display_only_suggestions"])
        state["uncertain"].extend(str(v) for v in plan["uncertain"] if str(v).strip())
        state["reviewed"] = number + 1
        save(len(groups) + number + 1)

    edit_plan = compile_plan(state["cuts"], state["pause_compressions"])
    applied = edit_plan["applied_proposals"]
    for number in range(state["captioned"], len(groups)):
        group = groups[number]
        tokens = _retained_tokens(group, words, applied)
        if tokens:
            kept_sentences = []
            for sentence in group:
                remaining = [t for t in tokens if sentence["start_word"] <= t["word_index"] < sentence["end_word"]]
                if remaining:
                    kept_sentences.append({"id": sentence["id"],
                                           "start_token": remaining[0]["token"],
                                           "end_token": remaining[-1]["token"] + 1,
                                           "text": "".join(t["text"] for t in remaining),
                                           "start_ms": remaining[0]["start_ms"],
                                           "end_ms": remaining[-1]["end_ms"]})
            context = {"phase": "caption", "full_transcript": full_text,
                       "applied_cuts": applied,
                       "display_only_suggestions": state["display_only_suggestions"],
                       "speaker": {"organization": person.organization,
                                   "title": person.title, "name": person.name},
                       "caption_preferences": {"target_min_chars": 10, "target_max_chars": 14,
                                               "max_chars": MAX_CAPTION_CHARS,
                                               "semantic_boundaries_first": True},
                       "sentences": kept_sentences, "kept_tokens": tokens}
            error, previous, pieces = "", None, None
            for attempt in range(2):
                try:
                    response = chat_json(prompts.caption, _model_input(context) + (
                        "\n上次字幕无法映射或排版，请修正：" + error + "\n上次结果："
                        + json.dumps(previous, ensure_ascii=False) if error else ""))
                except ModelOutputError as exc:
                    state["caption_attempts"].append({"group": number + 1, "attempt": attempt + 1,
                                                      "response": None, "error": str(exc)})
                    if checkpoint:
                        checkpoint(deepcopy(state))
                    raise ModelOutputError(f"字幕模型调用未成功；已确认的模型计划和 ASR 结果已保存，可继续：{exc}") from exc
                try:
                    # Model owns wording. Duration/similarity heuristics are not
                    # allowed to restore words the model intentionally cleaned.
                    pieces = _caption_segments(response, tokens, words, float("inf"),
                                               enforce_wording_policy=False)
                    overlong = [p for p in pieces if len(p["text"]) > MAX_CAPTION_CHARS]
                    if overlong and attempt == 0:
                        raise ValueError(f"字幕超过 {MAX_CAPTION_CHARS} 字，请按自然意群拆分且保持专名完整："
                                         + "；".join(p["text"] for p in overlong[:3]))
                    if overlong:
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
                                local = _literal_captions([t for t in tokens if t["word_index"] in indices], words)
                                if number + 1 not in state["fallback_groups"]:
                                    state["fallback_groups"].append(number + 1)
                                state["uncertain"].append(f"单条字幕无法按真实时间可靠排版，局部保留识别原文：{exc}")
                            state["caption_formatting_repairs"].append({"group": number + 1,
                                "source_word_indices": piece["source_word_indices"],
                                "model_text": piece["text"], "render_texts": [p["text"] for p in local],
                                "literal_fallback": any(p.get("caption_fallback") for p in local)})
                            repaired.extend(local)
                        pieces = repaired
                    state["caption_attempts"].append({"group": number + 1, "attempt": attempt + 1,
                                                     "response": response, "error": ""})
                    state["changes"].extend(v for v in response.get("changes", []) if isinstance(v, dict))
                    state["uncertain"].extend(str(v) for v in response.get("uncertain", []) if str(v).strip())
                    break
                except ValueError as exc:
                    error, previous, pieces = str(exc), response, None
                    state["caption_attempts"].append({"group": number + 1, "attempt": attempt + 1,
                                                     "response": response, "error": error})
            if pieces is None:
                pieces = _literal_captions(tokens, words)
                state["fallback_groups"].append(number + 1)
                state["uncertain"].append(f"字幕两次无法按保留词时间映射，使用识别原文：{error}")
            state["captions"].extend(pieces)
        state["captioned"] = number + 1
        save(len(groups) * 2 + number + 1)

    removed = {i for cut in applied for i in range(cut["start_word"], cut["end_word"])}
    retained_source = [i for i in range(len(words)) if i not in removed]
    retained = [t["word_index"] for t in _retained_tokens(
        [{"start_word": 0, "end_word": len(words)}], words, applied)]
    assigned = [i for c in state["captions"] for i in c.get("source_word_indices", [])]
    estimated = any(c.get("timing_estimated") for c in state["captions"])
    if not estimated and assigned != retained:
        raise ValueError("字幕未按原顺序完整覆盖实际保留的源词")
    coverage = {"original_words": len(words), "removed_words": len(removed),
                "retained_words": len(retained_source), "captionable_words": len(retained),
                "punctuation_only_word_indices": [i for i in retained_source if i not in set(retained)],
                "caption_count": len(state["captions"]),
                "retained_words_assigned_once_in_order": assigned == retained,
                "fallback_groups": state["fallback_groups"], "estimated_caption_timing": estimated}
    return {"reviewed": True, "human_reviewed": False, "flow_version": 4,
            "mode": "model_authored", "context_scope": "whole_video" if len(groups) == 1 else "chapter_with_full_context",
            "coverage_audit": coverage, "estimated_timing": has_estimated_timing(transcription),
            "asr_sentences": sentences, "cut_proposals": applied, "model_cut_proposals": state["cuts"],
            "model_edit_plan": {"draft_plans": state["draft_plans"],
                                "final_plans": state["final_model_plans"]},
            "edit_plan": edit_plan, "captions": state["captions"], "changes": state["changes"],
            "uncertain": state["uncertain"], "caption_fallback_groups": state["fallback_groups"],
            "rejected_model_proposals": [], "caption_formatting_repairs": state["caption_formatting_repairs"],
            "edit_attempts": state["edit_attempts"], "review_attempts": state["review_attempts"],
            "caption_attempts": state["caption_attempts"]}
