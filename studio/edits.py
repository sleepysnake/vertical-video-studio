"""Build conservative word-timed video cuts and rebase captions after cutting."""
from __future__ import annotations

import re
import math
import subprocess
import sys

from .captions import Word
from .model_edits import cut_words, validate_model_cuts


SILENCE_END = re.compile(r"silence_end:\s*([\d.]+)\s*\|\s*silence_duration:\s*([\d.]+)")
SILENCE_START = re.compile(r"silence_start:\s*([\d.]+)")


def no_window() -> int:
    return subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0


def detect_silences(video: str, ffmpeg: str = "ffmpeg", duration: float = 0) -> list[tuple[float, float]]:
    """Find confirmed quiet audio regions; ASR word gaps gate the actual cuts."""
    command = [str(ffmpeg), "-hide_banner", "-nostats", "-i", str(video),
               "-map", "0:a:0", "-af", "silencedetect=noise=-35dB:d=0.55",
               "-f", "null", "-"]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace",
                            timeout=max(30, duration / 2), creationflags=no_window())
    if result.returncode:
        raise RuntimeError(f"检测音频静音失败：{result.stderr[-500:]}")
    intervals: list[tuple[float, float]] = []
    open_start: float | None = None
    for line in result.stderr.splitlines():
        beginning = SILENCE_START.search(line)
        if beginning:
            open_start = float(beginning.group(1))
        match = SILENCE_END.search(line)
        if match:
            end, length = (float(value) for value in match.groups())
            start = open_start if open_start is not None else end - length
            open_start = None
            if intervals and start - intervals[-1][1] <= 0.12:
                intervals[-1] = (intervals[-1][0], end)
            else:
                intervals.append((start, end))
    if open_start is not None and duration > open_start:
        intervals.append((open_start, duration))
    return intervals


def detect_long_silence_cuts(words: list[Word], silences: list[tuple[float, float]],
                             duration: float | None = None, threshold: float = 2.0,
                             margin: float = 0.12) -> list[dict]:
    cuts: list[dict] = []
    boundaries = [(before.end, after.start) for before, after in zip(words, words[1:])]
    if words and duration is not None:
        boundaries = [(0.0, words[0].start), *boundaries, (words[-1].end, duration)]
    for gap_start, gap_end in boundaries:
        if gap_end - gap_start < threshold:
            continue
        for quiet_start, quiet_end in silences:
            start = max(gap_start, quiet_start)
            end = min(gap_end, quiet_end)
            if end - start + 1e-6 >= threshold:
                cut_start = round(start if gap_start == 0 else start + margin, 3)
                cut_end = round(end if duration is not None and gap_end == duration else end - margin, 3)
                if cut_end > cut_start:
                    cuts.append({"start": cut_start, "end": cut_end,
                                 "source_text": "", "reason": "压缩长时间静音"})
    return cuts


def _proposal_span(item: dict, words: list[Word], *, silences=None,
                   abandoned_pause_seconds=0.55, silence_margin=0.12) -> dict | None:
    try:
        reason = str(item.get("reason", "口语冗余"))[:80]
        if "word_index" in item:
            index = item["word_index"]
            if type(index) is not int:
                return None
            if not 0 <= index < len(words):
                return None
            word = words[index]
            first, last = item["start_char"], item["end_char"]
            if type(first) is not int or type(last) is not int:
                return None
            if not 0 <= first < last <= len(word.text):
                return None
            duration = word.end - word.start
            start = word.start + duration * first / len(word.text)
            end = word.start + duration * last / len(word.text)
            source_text = word.text[first:last]
            source_range = {"word_index": index, "start_char": first, "end_char": last}
        else:
            first, last = item["start_word"], item["end_word"]
            if type(first) is not int or type(last) is not int:
                return None
            if not 0 <= first < last <= len(words):
                return None
            start, end = words[first].start, words[last - 1].end
            source_text = "".join(word.text for word in words[first:last])
            source_range = {"start_word": first, "end_word": last}
        # This maps proposals only. The edit policy below limits which
        # complete words may actually be removed from the source audio.
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            return None
        # Round inward so millisecond serialization cannot remove the adjacent
        # retained word when a provider returns sub-millisecond word times.
        span = {"start": math.ceil((start - 1e-9) * 1000) / 1000,
                "end": math.floor((end + 1e-9) * 1000) / 1000,
                "source_text": source_text, "reason": reason, **source_range}
        if span["end"] <= span["start"]:
            return None
        if item.get("kind"):
            span["kind"] = item["kind"]
        if item.get("replacement_range") is not None:
            span["replacement_range"] = dict(item["replacement_range"])
        if item.get("kind") == "abandoned_start":
            span.update({"kind": "abandoned_start", "spoken_start": round(start, 3),
                         "spoken_end": round(end, 3), "spoken_duration": round(end - start, 3)})
            # Meaning belongs to the model. The program only validates the
            # neighboring word and finds a cut point inside confirmed quiet.
            resume = item.get("resume_word", last)
            if ("word_index" not in item and type(resume) is int and
                    resume == last and 0 <= resume < len(words)):
                resume_start = words[resume].start
                span["resume_word"] = resume
                span["resume_start"] = round(resume_start, 3)
                gap = resume_start - end
                if gap + 1e-6 >= abandoned_pause_seconds:
                    candidates = []
                    for quiet_start, quiet_end in silences or []:
                        a, b = max(end, quiet_start), min(resume_start, quiet_end)
                        if (b - a + 1e-6 >= abandoned_pause_seconds and
                                resume_start - b <= max(.2, silence_margin) + 1e-6):
                            endpoint = b - silence_margin
                            if endpoint > a and endpoint > end:
                                candidates.append((endpoint, a, b))
                    if candidates:
                        endpoint, a, b = max(candidates)
                        span["end"] = math.floor((endpoint + 1e-9) * 1000) / 1000
                        span["pause_evidence"] = {"start": round(a, 3), "end": round(b, 3),
                                                  "duration": round(b - a, 3)}
        return span
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def build_cut_plan(words: list[Word], model_proposals: list[dict], duration: float,
                   silences: list[tuple[float, float]] | None = None, *,
                   silence_seconds: float = 2.0, silence_margin: float = 0.12,
                   abandoned_pause_seconds: float = 0.55) -> list[dict]:
    try:
        checked = validate_model_cuts(model_proposals, 0, len(words), words)
    except ValueError as exc:
        raise ValueError(f"模型剪切区间未能映射到原视频：{exc}") from exc
    selected = set().union(*(cut_words(item) for item in checked)) if checked else set()
    retained = [(index, word) for index, word in enumerate(words) if index not in selected]
    candidates: list[dict] = []
    for item in checked:
        span = _proposal_span(item, words, silences=silences,
                              abandoned_pause_seconds=abandoned_pause_seconds,
                              silence_margin=silence_margin) if isinstance(item, dict) else None
        if span is None:
            raise ValueError(f"模型剪切区间未能映射到原视频：{item}")
        if span["start"] >= duration or span["end"] > duration + .001:
            raise ValueError("模型剪切词时间超出源视频范围")
        candidates.append(span)
    candidates.extend(detect_long_silence_cuts(words, silences or [], duration,
                                               silence_seconds, silence_margin))
    candidates = [span for span in candidates if span["start"] < duration]
    candidates.sort(key=lambda item: (item["start"], -item["end"]))
    result: list[dict] = []
    for item in candidates:
        item = {**item, "end": min(duration, item["end"])}
        # Combining close cuts must not swallow even a tiny retained word.
        kept_between = bool(result) and any(
            word.start < item["start"] and word.end > result[-1]["end"]
            for _, word in retained)
        if not result or item["start"] > result[-1]["end"] + 0.01 or kept_between:
            result.append(item)
        else:
            result[-1]["end"] = max(result[-1]["end"], item["end"])
            result[-1]["reason"] += "；" + item["reason"]
    for span in result:
        intrusions = [index for index, word in retained
                      if span["start"] < word.end - 1e-6 and span["end"] > word.start + 1e-6]
        if intrusions:
            raise ValueError(f"剪切点侵入必须保留的原声词：{intrusions[:8]}")
        span["removed_word_indices"] = [index for index in sorted(selected)
                                        if span["start"] < words[index].end and
                                        span["end"] > words[index].start]
        span["retained_word_check"] = {"verified": True, "checked": len(retained),
                                       "intrusions": []}
    return result


def keep_intervals(cuts: list[dict], duration: float) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    cursor = 0.0
    for cut in cuts:
        start, end = float(cut["start"]), float(cut["end"])
        if not cursor <= start < end <= duration + 0.01:
            raise ValueError("视频裁剪时间区间无效")
        if start - cursor > 1e-6:
            result.append((cursor, start))
        cursor = end
    if duration - cursor > 1e-6:
        result.append((cursor, duration))
    if not result:
        raise ValueError("视频裁剪后没有可保留的画面")
    return result


def output_duration(cuts: list[dict], duration: float) -> float:
    return duration - sum(float(item["end"]) - float(item["start"]) for item in cuts)


def retime_captions(captions: list[dict], cuts: list[dict]) -> list[dict]:
    def mapped(t: float) -> float:
        removed = sum(max(0.0, min(t, float(cut["end"])) - float(cut["start"]))
                      for cut in cuts if t > float(cut["start"]))
        return round(max(0.0, t - removed), 3)

    result: list[dict] = []
    for caption in captions:
        start, end = mapped(float(caption["start"])), mapped(float(caption["end"]))
        if end - start <= 0.04:
            continue
        result.append({**caption, "start": start, "end": end,
                       "source_start": caption["start"], "source_end": caption["end"]})
    return result
