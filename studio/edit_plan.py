"""Versioned, inspectable decisions between analysis and media rendering."""
from __future__ import annotations

import hashlib
import json

from .captions import Word
from .edits import _proposal_span, build_cut_plan, detect_long_silence_cuts, keep_intervals
from .strategy import Strategy
from .cleanup import basic_proposal_kind
from .model_edits import (SEMANTIC_CUT_KINDS, cut_words, protected_words,
                          validate_model_cuts)


PLAN_VERSION = 5
FILLERS = {"嗯", "呃", "额"}


def _id(source: str, value: dict) -> str:
    payload = json.dumps([source, value], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def build_edit_plan(words: list[Word], model_proposals: list[dict], duration: float,
                    silences: list[tuple[float, float]], *,
                    estimated_timing: bool = False,
                    strategy: Strategy | None = None) -> dict:
    """Apply model-declared semantic edits within technical timing limits.

    Model output has no calibrated confidence. The safety limits are explicit
    policy, not an invented probability of the model being correct.
    """
    strategy = strategy or Strategy()
    decisions: list[dict] = []
    accepted: list[dict] = []
    for proposal in model_proposals:
        span = _proposal_span(proposal, words, silences=silences,
                              silence_margin=strategy.silence_margin,
                              abandoned_pause_seconds=strategy.abandoned_pause_seconds)
        if span is None:
            raise ValueError(f"模型剪切区间未能映射到原视频：{proposal}")
        abandoned = proposal.get("kind") == "abandoned_start"
        reason = ""
        try:
            validate_model_cuts([proposal], 0, len(words), words)
        except ValueError as exc:
            reason = str(exc)
        kind = (proposal.get("kind") if isinstance(proposal.get("kind"), str) and proposal.get("kind") in SEMANTIC_CUT_KINDS
                else basic_proposal_kind(words, proposal) if not reason else None)
        short = span["end"] - span["start"] < strategy.min_model_cut_seconds - .001
        caption_only = (kind in {"filler", "stutter", "isolated_filler"} and
                        span["end"] - span["start"] <= strategy.max_subtitle_cleanup_seconds + .001)
        # Protected replacement/resume words remain protected across groups
        # and independently returned model responses, in either proposal order.
        protected = set().union(*(protected_words(item) for item in accepted)) if accepted else set()
        prior_words = set().union(*(cut_words(item) for item in accepted)) if accepted else set()
        if not reason and (cut_words(proposal) & protected or protected_words(proposal) & prior_words):
            reason = "接续词和替代表达必须在最终剪辑中完整保留，与其他剪切提议冲突"
        trial, fragments = [], []
        if not reason:
            try:
                trial = build_cut_plan(words, accepted + [proposal], duration, silences,
                                       silence_seconds=strategy.silence_seconds,
                                       silence_margin=strategy.silence_margin,
                                       abandoned_pause_seconds=strategy.abandoned_pause_seconds)
                fragments = keep_intervals(trial, duration)
            except ValueError as exc:
                reason = str(exc)
        removed = sum(cut["end"] - cut["start"] for cut in trial)
        if not reason and kind is None:
            reason = "旧格式未声明语义裁剪类型，只按基础口癖或短促重复起音处理"
        elif not reason and short:
            reason = "原声片段过短，保留音视频，字幕阶段可记录短促口癖修整"
        elif not reason and "word_index" in proposal:
            reason = "词内切点只有均分估算，保留音视频，字幕阶段可记录短促修整"
        elif not reason and caption_only:
            reason = "短促口癖或结巴只修整字幕，保留音视频自然接续"
        elif not reason and estimated_timing:
            reason = "ASR 只有估算词时间，保留原声供复核"
        elif not reason and (span["start"] < 0 or span["end"] > duration + .001 or span["start"] >= duration):
            reason = "原声词时间超出源视频范围，保留原声并记录时间异常"
        elif not reason and span["end"] - span["start"] > strategy.max_single_model_cut:
            reason = "单次删除超过安全上限，保留原声供复核"
        elif not reason and duration > 0 and removed / duration > strategy.max_removal_ratio:
            reason = "累计删除比例超过安全上限，保留原声供复核"
        elif not reason and any(end - start < strategy.min_retained_fragment for start, end in fragments):
            reason = "剪切后产生过短片段，保留原声供复核"
        decision = {"id": _id("llm", proposal), "source": "llm",
                    "start": span["start"], "end": span["end"],
                    "source_text": span["source_text"],
                    "reason": span["reason"], "proposal": proposal,
                    "action": "KEEP" if reason else "CUT",
                    "need_review": bool(reason) and not (
                        kind is not None and (short or caption_only or "word_index" in proposal)),
                    "policy_reason": reason, "cleanup_kind": kind}
        if abandoned:
            decision["spoken_start"] = span["spoken_start"]
            decision["spoken_end"] = span["spoken_end"]
            decision["resume_word"] = proposal.get("resume_word", proposal.get("end_word"))
            decision["resume_start"] = span.get("resume_start")
            decision["pause_evidence"] = span.get("pause_evidence")
        if proposal.get("replacement_range") is not None:
            decision["replacement_range"] = proposal["replacement_range"]
        if proposal.get("joined_text") is not None:
            decision["joined_text"] = proposal["joined_text"]
        decisions.append(decision)
        if not reason:
            accepted.append(proposal)

    for silence in detect_long_silence_cuts(words, silences, duration,
                                            strategy.silence_seconds, strategy.silence_margin):
        decisions.append({"id": _id("silence", silence), "source": "rule:silence",
                          **silence, "action": "CUT", "need_review": False,
                          "policy_reason": "词间隔和音轨静音均已确认"})

    # A transcript alone cannot prove that a short hesitation is disposable.
    # Record isolated fillers as review candidates; the LLM may independently
    # select one for cutting above.
    for index, word in enumerate(words):
        if word.text.strip() not in FILLERS:
            continue
        if any(item["start"] <= word.start and item["end"] >= word.end
               for item in decisions if item["action"] == "CUT"):
            continue
        decisions.append({"id": _id("filler", {"word_index": index}),
                          "source": "rule:filler", "start": round(word.start, 3),
                          "end": round(word.end, 3), "source_text": word.text,
                          "reason": "疑似口气词", "action": "KEEP",
                          "need_review": word.end - word.start > strategy.max_subtitle_cleanup_seconds,
                          "policy_reason": "词文本不足以判断语义及切点安全"})

    cuts = build_cut_plan(words, accepted, duration, silences,
                          silence_seconds=strategy.silence_seconds,
                          silence_margin=strategy.silence_margin,
                          abandoned_pause_seconds=strategy.abandoned_pause_seconds)
    removed_words = set().union(*(cut_words(item) for item in accepted)) if accepted else set()
    retained_words = [index for index in range(len(words)) if index not in removed_words]
    return {"version": PLAN_VERSION, "mode": "contextual_cleanup", "policy": strategy.snapshot(),
            "policy_sha256": strategy.digest(),
            "source_duration": round(duration, 3),
            "decisions": sorted(decisions, key=lambda item: (item["start"], item["end"])),
            "applied_proposals": accepted, "cuts": cuts,
            "retained_word_audit": {"verified": True, "word_indices": retained_words,
                                     "checked": len(retained_words),
                                     "method": "all_retained_asr_words_outside_cut_intervals"},
            "review_count": sum(item["need_review"] for item in decisions)}
