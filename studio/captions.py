"""Turn timestamped ASR words into checked short Chinese captions."""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
import json
import re
from typing import Callable

from .models import Person


PUNCT = re.compile(r"[，。！？、；：,.!?;:…“”‘’\"'（）()《》\[\]{}]")
CLAUSE_PUNCT = re.compile(r"[，。！？、；：,.!?;:…]+")
BAD_START = set("的了着过呢啊吧吗呀")
BAD_END = set("的和或与把在是并就只还将能要")
PREFERRED_CAPTION_CHARS = 14
MAX_CAPTION_CHARS = 16
STRONG_STARTS = ("我是", "我叫", "只吃", "只喝", "不是", "而是", "但是", "如果", "因为", "所以", "然后", "那么",
                 "或者", "我们", "就相当于", "与此同时", "同时", "接下来", "而且", "远远")


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str
    punctuation: str = ""


def extract_words(transcription: dict) -> list[Word]:
    words: list[Word] = []
    transcripts = transcription.get("transcripts")
    if not isinstance(transcripts, list):
        raise ValueError("转写结果缺少 transcripts")
    for transcript in transcripts[:1]:  # one mono channel; avoid duplicate stereo transcripts
        for sentence in transcript.get("sentences", []):
            source = sentence.get("words")
            if isinstance(source, list) and source:
                for item in source:
                    value = str(item.get("text", "")).strip()
                    if not value:
                        continue
                    words.append(Word(float(item["begin_time"]) / 1000,
                                      float(item["end_time"]) / 1000,
                                      value, str(item.get("punctuation", ""))))
            else:
                # Some ASR variants return sentence times without word times.
                # Approximate per-character positions so captions can still be
                # edited; this is clearly marked in the saved caption JSON.
                value = str(sentence.get("text", "")).strip()
                clean = PUNCT.sub("", value)
                if not clean:
                    continue
                start = float(sentence["begin_time"]) / 1000
                end = float(sentence["end_time"]) / 1000
                step = max(0.01, (end - start) / len(clean))
                for index, char in enumerate(clean):
                    words.append(Word(start + index * step, min(end, start + (index + 1) * step), char))
    if not words:
        raise ValueError("转写结果没有可用的语音词或句子")
    previous = -1.0
    for word in words:
        if word.start < previous - 0.04 or word.end <= word.start:
            raise ValueError("语音时间戳存在倒序或空区间")
        previous = word.end
    return words


def has_estimated_timing(transcription: dict) -> bool:
    return any(not sentence.get("words")
               for transcript in transcription.get("transcripts", [])[:1]
               for sentence in transcript.get("sentences", []))


def chunks(words: list[Word], max_chars: int = 110) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    start = 0
    count = 0
    for index, word in enumerate(words):
        count += len(word.text)
        at_sentence_end = bool(re.search(r"[。！？!?]", word.punctuation))
        at_clause_end = bool(re.search(r"[，,；;：:]", word.punctuation))
        pause = (words[index + 1].start - word.end) if index + 1 < len(words) else 0
        if ((count >= max_chars // 2 and at_sentence_end)
                or (count >= max_chars and (at_clause_end or pause >= 0.45))
                or count >= max_chars + 30):
            result.append((start, index + 1))
            start = index + 1
            count = 0
    if start < len(words):
        result.append((start, len(words)))
    return result


SYSTEM_PROMPT = """你是中文口播视频字幕校对员。只返回 JSON 对象，不写 Markdown。
任务：依据带时间戳的 ASR 词序列，理解整句话后纠正明显误识别，删去口头语、无意义重复和说话时改口的废词，再按语义分句输出字幕。
先把当前 words 中每个词判为“保留”或“从音视频剪掉”。segments 是字幕显示范围，允许跨过中间被剪掉的口癖词，以保持一句完整字幕；cuts 用左闭右开 [start_word,end_word) 精确标记确实说出口却要删除的词。被完全舍弃的整句只在 cuts 中，不在 segments 中；词内结巴可另外使用 word_index/start_char/end_char 的局部 cut。每个输入词至少属于一个 segment 或一个 cut，相邻字幕的 segment 不得互相重叠。
字幕 text 可以纠正 ASR 错别字，但不能只在文字里压缩真实说出口的内容。如果删去真实说出口的语气词、口癖、废词、改口或重复内容，必须同时用 cuts 标出对应音视频词区间。ASR 误识别的词仍然保留其音视频和索引，只修改显示文字并在 changes 单独写明原词索引和“ASR误识别”原因；不要把 ASR 错字与真实口误混写成同一项 changes。
preceding_text 和 following_text 只供理解上下文，不能当作当前 words 的内容写入字幕。每条字幕的文字必须对应自身词索引范围内的语音；不要把上一句或下一句的文字配到当前时间段。
例如 ASR 为“第二个要大量的吃输液呃绿叶蔬菜来蔬菜”，上下文讨论饮食建议时可整理成“第二个要大量的吃蔬菜”。必须逐词区分：识别错但真实发音有意义的词只改字幕；真实说出的“呃”、改口和多余重说则在字幕删除的同时给出 cuts。不能把整串话笼统归为 ASR 错字来绕过剪辑。
只做有上下文依据的精简，保留原意、否定词、数字、药名、剂量等重要信息；不能确定是否为口误或原话时保留较可信的文字并在 uncertain 中说明，但不中断自动制作。
先按自然语义断句，开场寒暄、自我介绍、原因、建议等不同意思分别成段；不要把能独立成句的两个意思合在一条字幕里。例如“各位朋友大家好我是武汉同济医院李景平医生”应分为“各位朋友大家好”和“我是武汉同济医院李景平医生”。
每条字幕优先在自然语义处断开，尽量 10 到 14 字，最多 16 字；超过 16 字必须在下一个完整短语前另起一条，不能靠缩小字号容纳长句。例如“也就是说我今天并没有吃多少只吃了一个汉堡或喝了一杯奶茶”应分为“也就是说我今天并没有吃多少”和“只吃了一个汉堡或喝了一杯奶茶”。断点应优先用 ASR 停顿、标点和完整意群，不拆开医院或机构名、人名、职称、数字与单位、否定结构及动宾短语。长度和语义必须同时满足；如果一个完整短语超过 16 字，按内部可独立理解的意群再拆。不要标点；不要以“的、了、着、过、呢、啊、吧、吗、呀”开头。
对确实说出的语气词、口癖、结巴起音和重复话语，字幕要删字，同时用 cuts 标记要从原视频同步删除的词或词内字符。示例：词 26 为“咦”，词 27 为“我吃”，词 28 为“吃得”，应输出只覆盖词 27 起的字幕“我吃得不多”；cuts 包含 {"start_word":26,"end_word":27,"reason":"语气词"} 和 {"word_index":27,"start_char":1,"end_char":2,"reason":"重复起音"}。词内字符区间也是左闭右开。不要把专名纠错或 ASR 误识别当作视频删除点。
整句或半句再次出现时，要根据前后内容判断是刻意强调、补充新信息，还是忘词、改口后重新说。刻意强调或含有独有信息的两遍都保留，分别出字幕；若前一遍是未完成的口误，而后一遍重新完整表达，应将前一遍的全部词放进一个 cut，并只给保留的后一遍出字幕。即使两遍多字少字、不完全相同，也应按表达意图判断，不能仅凭字面相似度。不要删掉独有的否定词、数字、药名、剂量或观点。
不可凭空补充原音频没有的信息。可以依据说话人工作单位等背景纠正明显误识别，例如“武汉同济院”可改成“武汉同济医院”。
输出格式：{"segments":[{"start_word":整数,"end_word":整数,"text":"字幕"}],"cuts":[{"start_word":整数,"end_word":整数,"reason":"真实说出口但需要删除的原因"},{"word_index":整数,"start_char":整数,"end_char":整数,"reason":"重复起音"}],"changes":[{"start_word":整数,"end_word":整数,"from":"原词","to":"修正词或空字符串","reason":"依据"}],"uncertain":["保留原词的疑点说明"]}。被删词可以落在一条保留字幕的时间范围内，但必须有对应 cut；明显误识别只写入 changes，不写入 cuts。
如果词义无法确定，保留较可信的原词并写入 uncertain。"""


def _validate_segments(segments: object, start: int, end: int, words: list[Word]) -> list[dict]:
    if not isinstance(segments, list) or not segments:
        raise ValueError("segments 为空")
    result: list[dict] = []
    cursor = start
    for item in segments:
        if not isinstance(item, dict):
            raise ValueError("字幕分段格式错误")
        try:
            first = int(item["start_word"])
            last = int(item["end_word"])
            value = str(item["text"]).strip()
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("字幕段缺少索引或文字") from exc
        if first != cursor:
            if abs(first - cursor) > 2 or not start <= first <= end:
                raise ValueError(f"词索引未连续覆盖：期望 {cursor}，实际 {first} 到 {last}")
            if first > cursor and result and PUNCT.search(words[first - 1].punctuation):
                # A model may leave the final word of the preceding sentence
                # between two ranges. Its time belongs to the preceding caption.
                result[-1]["end_word"] = first
                result[-1]["end"] = round(words[first - 1].end, 3)
                result[-1]["index_repaired"] = True
            else:
                # Attach a small gap at the beginning, or trim a small overlap.
                first = cursor
        if not first < last <= end:
            raise ValueError(f"字幕词索引区间无效：{first} 到 {last}")
        if not value or PUNCT.search(value) or value[0] in BAD_START:
            raise ValueError(f"字幕含标点、以虚词开头或为空：{value}")
        if len(value.replace(" ", "")) > max(6, sum(len(w.text) for w in words[first:last]) * 2 + 3):
            raise ValueError(f"字幕可能添加了未说出的内容：{value}")
        result.append({"start": round(words[first].start, 3), "end": round(words[last - 1].end, 3),
                       "text": value, "start_word": first, "end_word": last,
                       **({"index_repaired": True} if first != int(item["start_word"]) else {})})
        cursor = last
    if cursor != end:
        if end - cursor > 2:
            raise ValueError(f"词索引未覆盖末尾：{cursor} 到 {end}")
        result[-1]["end_word"] = end
        result[-1]["end"] = round(words[end - 1].end, 3)
        result[-1]["index_repaired"] = True
    return result


def _normalize_model_segments(segments: object, words: list[Word]) -> object:
    """Use model punctuation as clause hints, then remove it from display text."""
    if not isinstance(segments, list):
        return segments
    result: list[dict] = []
    for item in segments:
        if not isinstance(item, dict):
            result.append(item)
            continue
        value = str(item.get("text", "")).strip()
        pieces = [PUNCT.sub("", part).strip() for part in CLAUSE_PUNCT.split(value)]
        pieces = [part for part in pieces if part]
        cleaned = PUNCT.sub("", value).strip()
        try:
            first, last = int(item["start_word"]), int(item["end_word"])
        except (KeyError, ValueError, TypeError):
            result.append({**item, "text": cleaned})
            continue
        if len(pieces) < 2 or not 0 <= first < last <= len(words) or last - first < len(pieces):
            result.append({**item, "text": cleaned})
            continue
        source_words = words[first:last]
        source = "".join(PUNCT.sub("", word.text) for word in source_words)
        display = "".join(pieces)
        offsets = _source_offsets(source, display)
        word_boundaries = [0]
        for word in source_words:
            word_boundaries.append(word_boundaries[-1] + len(PUNCT.sub("", word.text)))
        cursor = first
        display_cursor = 0
        for number, piece in enumerate(pieces):
            if number == len(pieces) - 1:
                word_end = last
            else:
                display_cursor += len(piece)
                remaining = len(pieces) - number - 1
                candidates = range(cursor + 1, last - remaining + 1)
                word_end = min(candidates, key=lambda index: (
                    abs(word_boundaries[index - first] - offsets[display_cursor]),
                    0 if PUNCT.search(words[index - 1].punctuation) else 1))
            result.append({**item, "start_word": cursor, "end_word": word_end, "text": piece})
            cursor = word_end
    return result


def _valid_chunk_cut(item: dict, start: int, end: int, words: list[Word]) -> bool:
    """A model may only edit audio covered by the chunk it was shown."""
    try:
        if "word_index" in item:
            index = int(item["word_index"])
            return (start <= index < end and 0 <= int(item["start_char"])
                    < int(item["end_char"]) <= len(words[index].text))
        return start <= int(item["start_word"]) < int(item["end_word"]) <= end
    except (KeyError, TypeError, ValueError):
        return False


def _validate_edit_plan(segments: object, cuts: object, start: int, end: int,
                        words: list[Word], changes: object = None) -> tuple[list[dict], list[dict]]:
    """Reject missing audio edits instead of silently cleaning only caption text."""
    if not isinstance(cuts, list):
        raise ValueError("cuts 必须是数组；没有剪辑时返回 []")
    checked_cuts: list[dict] = []
    whole_word_cuts: list[dict] = []
    for cut in cuts:
        if not isinstance(cut, dict) or not _valid_chunk_cut(cut, start, end, words):
            raise ValueError("音视频剪切词索引超出当前分段或格式错误")
        reason = str(cut.get("reason", "")).strip()
        if not reason or any(label in reason for label in ("ASR", "识别", "误识", "纠错")):
            raise ValueError("ASR 文字纠错不能作为音视频剪切；请保留原词索引")
        if "word_index" not in cut:
            first, last = int(cut["start_word"]), int(cut["end_word"])
            if words[last - 1].end - words[first].start > 25:
                raise ValueError("单个剪切区间超过 25 秒，请按真实语义缩小")
            whole_word_cuts.append(cut)
        checked_cuts.append(cut)
    if not isinstance(segments, list):
        raise ValueError("segments 必须是数组")
    validated: list[dict] = []
    for item in segments:
        if not isinstance(item, dict):
            raise ValueError("字幕分段格式错误")
        try:
            first, last = int(item["start_word"]), int(item["end_word"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("字幕段缺少词索引") from exc
        if not start <= first < last <= end:
            raise ValueError("字幕词索引超出当前分段")
        validated.extend(_validate_segments([item], first, last, words))
    caption_coverage: list[bool] = [False] * (end - start)
    cut_coverage: list[bool] = [False] * (end - start)
    for item in validated:
        for index in range(int(item["start_word"]), int(item["end_word"])):
            if caption_coverage[index - start]:
                raise ValueError(f"词 {index} 同时属于两条字幕")
            caption_coverage[index - start] = True
    for item in whole_word_cuts:
        for index in range(int(item["start_word"]), int(item["end_word"])):
            if cut_coverage[index - start]:
                raise ValueError(f"词 {index} 被重复剪掉")
            cut_coverage[index - start] = True
    missing = [start + index for index in range(end - start)
               if not caption_coverage[index] and not cut_coverage[index]]
    if missing:
        raise ValueError(f"词 {missing[:5]} 未分配字幕或音视频剪切")
    if any(a["start_word"] > b["start_word"] for a, b in zip(validated, validated[1:])):
        raise ValueError("字幕词索引不是时间顺序")
    corrections = [item for item in (changes if isinstance(changes, list) else [])
                   if isinstance(item, dict)
                   and any(label in str(item.get("reason", "")) for label in ("ASR", "识别", "误识"))
                   and not any(label in str(item.get("reason", "")) for label in ("口误", "重复", "废词"))]
    for caption in validated:
        if all(cut_coverage[index - start]
               for index in range(caption["start_word"], caption["end_word"])):
            raise ValueError("字幕范围内的语音已全部剪掉")
    partial_cuts: dict[int, list[tuple[int, int]]] = {}
    for cut in checked_cuts:
        if "word_index" in cut:
            partial_cuts.setdefault(int(cut["word_index"]), []).append(
                (int(cut["start_char"]), int(cut["end_char"])))
    source_words = []
    for index in range(start, end):
        if cut_coverage[index - start]:
            continue
        value = words[index].text
        previous_start = len(value)
        for first, last in sorted(partial_cuts.get(index, []), reverse=True):
            if last > previous_start:
                raise ValueError(f"词 {index} 的词内剪切范围重叠")
            value = value[:first] + value[last:]
            previous_start = first
        source_words.append((index, PUNCT.sub("", value)))
    source = "".join(value for _, value in source_words)
    display = "".join(caption["text"] for caption in validated)
    matched = [False] * len(source)
    for block in SequenceMatcher(None, source, display, autojunk=False).get_matching_blocks():
        for offset in range(block.a, block.a + block.size):
            matched[offset] = True
    cursor = 0
    omitted: list[str] = []
    unmatched_repeats: list[str] = []
    for index, value in source_words:
        matches = any(matched[cursor:cursor + len(value)])
        for position in range(cursor, cursor + len(value)):
            if (not matched[position] and
                    ((position > 0 and source[position] == source[position - 1]) or
                     (position + 1 < len(source) and source[position] == source[position + 1]))):
                unmatched_repeats.append(f"{index}:{source[position]}")
        cursor += len(value)
        if matches or not value:
            continue
        corrected = any(value in str(change.get("from", ""))
                        and ("start_word" not in change or
                             int(change["start_word"]) <= index < int(change.get("end_word", index + 1)))
                        for change in corrections)
        if not corrected:
            omitted.append(f"{index}:{value}")
    if omitted or unmatched_repeats:
        raise ValueError("字幕省略了原词 " + "、".join(omitted[:20])
                         + ("；重复起音 " + "、".join(unmatched_repeats[:20])
                            if unmatched_repeats else "")
                         + "；请为真实口癖或改口给出对应 cuts，ASR 错字在 changes 写明")
    return validated, checked_cuts


def _source_offsets(source: str, text: str) -> list[int]:
    """Map corrected-text boundaries back to nearby ASR character offsets."""
    offsets = [0] * (len(text) + 1)
    for _tag, i1, i2, j1, j2 in SequenceMatcher(None, source, text, autojunk=False).get_opcodes():
        if j2 == j1:
            continue
        for position in range(j1, j2 + 1):
            offsets[position] = i1 + round((position - j1) * (i2 - i1) / (j2 - j1))
    offsets[-1] = len(source)
    return offsets


def _split_long_caption(caption: dict, words: list[Word], protected: tuple[str, ...] = ()) -> list[dict]:
    text = str(caption["text"])
    if len(text.replace(" ", "")) <= MAX_CAPTION_CHARS:
        return [caption]
    has_word_range = "start_word" in caption and "end_word" in caption
    first = int(caption["start_word"]) if has_word_range else 0
    last = int(caption["end_word"]) if has_word_range else 0
    source_words = words[first:last] if has_word_range and 0 <= first < last <= len(words) else []
    source = "".join(PUNCT.sub("", word.text) for word in source_words)
    offsets = _source_offsets(source, text)
    boundaries = [0]
    for word in source_words:
        boundaries.append(boundaries[-1] + len(PUNCT.sub("", word.text)))

    def nearest_word(position: int) -> int:
        return min(range(1, len(source_words)),
                   key=lambda index: abs(boundaries[index] - offsets[position]))

    protected_boundaries = {position for phrase in protected if len(phrase) > 1
                            for match in re.finditer(re.escape(phrase), text)
                            for position in range(match.start() + 1, match.end())}

    @lru_cache(None)
    def best(position: int) -> tuple[float, tuple[int, ...]]:
        if position == len(text):
            return 0.0, ()
        winner = (float("inf"), ())
        for end in range(position + 1, min(len(text), position + MAX_CAPTION_CHARS) + 1):
            if end in protected_boundaries:
                continue
            piece = text[position:end]
            length = len(piece.replace(" ", ""))
            if length > MAX_CAPTION_CHARS or not length:
                continue
            if end < len(text) and text[end] in BAD_START:
                continue
            tail_cost, tail = best(end)
            if tail_cost == float("inf"):
                continue
            cost = 10 + (length - PREFERRED_CAPTION_CHARS) ** 2 * 0.55 + tail_cost
            if length < 7:
                cost += (7 - length) * 7
            if end < len(text):
                if text[end - 1] in BAD_END:
                    cost += 22
                if any(text.startswith(prefix, end) for prefix in STRONG_STARTS):
                    cost -= 25
                if text[end - 1].isascii() and text[end - 1].isalnum() and text[end].isascii() and text[end].isalnum():
                    cost += 60
                if len(source_words) > 1:
                    word_index = nearest_word(end)
                    cost += min(8, abs(boundaries[word_index] - offsets[end]) * 2)
                    if PUNCT.search(source_words[word_index - 1].punctuation):
                        cost -= 18
            if cost < winner[0]:
                winner = cost, (end,) + tail
        return winner

    cuts = best(0)[1]
    if not cuts or cuts[-1] != len(text):
        raise ValueError("字幕无法在不拆散语义的前提下缩短")
    result: list[dict] = []
    word_cursor = first
    text_cursor = 0
    use_estimated_timing = len(source_words) < len(cuts)
    for piece_number, text_end in enumerate(cuts):
        remaining = len(cuts) - piece_number - 1
        if use_estimated_timing:
            word_end = word_cursor
        elif remaining:
            if last - word_cursor > remaining:
                suggested = first + nearest_word(text_end)
                word_end = min(last - remaining, max(word_cursor + 1, suggested))
            else:
                word_end = word_cursor
        else:
            word_end = last
        piece = dict(caption)
        piece.update({"text": text[text_cursor:text_end], "length_split": True})
        if not use_estimated_timing and word_end > word_cursor:
            piece.update({"start_word": word_cursor, "end_word": word_end,
                          "start": round(words[word_cursor].start, 3),
                          "end": round(words[word_end - 1].end, 3)})
            if "source_word_indices" in caption:
                piece["source_word_indices"] = [i for i in caption["source_word_indices"]
                                                if word_cursor <= i < word_end]
        else:
            # Rare ASR result with fewer timed words than subtitle phrases.
            span = float(caption["end"]) - float(caption["start"])
            piece.update({"start": round(float(caption["start"]) + span * text_cursor / len(text), 3),
                          "end": round(float(caption["start"]) + span * text_end / len(text), 3),
                          "timing_estimated": True})
            piece.pop("start_word", None)
            piece.pop("end_word", None)
            piece.pop("source_word_indices", None)
        result.append(piece)
        word_cursor = word_end
        text_cursor = text_end
    return result


def shorten_captions(captions: list[dict], words: list[Word],
                     protected: tuple[str, ...] = ()) -> list[dict]:
    result: list[dict] = []
    for caption in captions:
        result.extend(_split_long_caption(caption, words, protected))
    return result


def build_captions(transcription: dict, person: Person, chat_json: Callable[[str, str], dict],
                   progress: Callable[[int, int], None] | None = None,
                   resume: dict | None = None,
                   checkpoint: Callable[[dict], None] | None = None) -> dict:
    words = extract_words(transcription)
    sections = chunks(words)
    ranges = [[start, end] for start, end in sections]
    next_section = 0
    captions: list[dict] = []
    changes: list[dict] = []
    uncertain: list[str] = []
    cut_proposals: list[dict] = []
    if (isinstance(resume, dict) and resume.get("ranges") == ranges
            and isinstance(resume.get("captions"), list)
            and isinstance(resume.get("changes"), list)
            and isinstance(resume.get("uncertain"), list)):
        candidate = resume.get("next_section")
        if isinstance(candidate, int) and 0 <= candidate <= len(sections):
            next_section = candidate
            captions = list(resume["captions"])
            changes = list(resume["changes"])
            uncertain = list(resume["uncertain"])
            cut_proposals = list(resume.get("cut_proposals") or [])
    if next_section and progress:
        progress(next_section, len(sections))
    for section_number in range(next_section + 1, len(sections) + 1):
        start, end = sections[section_number - 1]
        context = {
            "speaker": {"organization": person.organization, "title": person.title, "name": person.name},
            "word_range": [start, end],
            "preceding_text": "".join(word.text for word in words[max(0, start - 30):start]),
            "following_text": "".join(word.text for word in words[end:min(len(words), end + 30)]),
            "words": [{"i": i, "text": words[i].text, "punctuation": words[i].punctuation,
                       "start_ms": round(words[i].start * 1000),
                       "end_ms": round(words[i].end * 1000)}
                      for i in range(start, end)],
        }
        error = ""
        previous_segments = None
        for attempt in range(3):
            user = json.dumps(context, ensure_ascii=False)
            if error:
                user += f"\n上次输出无效：{error}。请修正 JSON 后重试。"
                if previous_segments is not None:
                    user += "\n上次字幕：" + json.dumps(previous_segments, ensure_ascii=False)
            response = chat_json(SYSTEM_PROMPT, user)
            try:
                normalized = _normalize_model_segments(response.get("segments"), words)
                items, verified_cuts = _validate_edit_plan(
                    normalized, response.get("cuts", []), start, end, words,
                    response.get("changes", []))
                too_long = [item["text"] for item in items
                            if len(item["text"].replace(" ", "")) > MAX_CAPTION_CHARS]
                if too_long and attempt < 2:
                    previous_segments = response.get("segments")
                    error = ("这些字幕超过 16 字：" + "；".join(too_long[:3])
                             + "。请先按自然语义和词索引重新分段，保持机构名、人名和完整短语不被拆散")
                    continue
                captions.extend(shorten_captions([item for item in items if item["text"]], words,
                                                 (person.organization, person.name, person.title)))
                cut_proposals.extend(verified_cuts)
                if isinstance(response.get("changes"), list):
                    changes.extend(change for change in response["changes"] if isinstance(change, dict))
                if isinstance(response.get("uncertain"), list):
                    uncertain.extend(str(item) for item in response["uncertain"] if str(item).strip())
                break
            except ValueError as exc:
                error = str(exc)
                previous_segments = {key: response.get(key) for key in
                                     ("segments", "cuts", "changes")}
                if attempt == 2:
                    raise ValueError(f"第 {section_number} 段文本模型连续三次返回无效字幕：{error}") from exc
        if progress:
            progress(section_number, len(sections))
        if checkpoint:
            checkpoint({"ranges": ranges, "next_section": section_number,
                        "captions": captions, "changes": changes, "uncertain": uncertain,
                        "cut_proposals": cut_proposals})
    return {"reviewed": True, "human_reviewed": False, "uncertain": uncertain,
            "estimated_timing": has_estimated_timing(transcription),
            "changes": changes, "cut_proposals": cut_proposals, "captions": captions}


def validate_caption_document(data: dict, duration: float) -> None:
    if not isinstance(data, dict) or not isinstance(data.get("captions"), list) or not data["captions"]:
        raise ValueError("字幕文件缺少 captions")
    previous = 0.0
    for index, caption in enumerate(data["captions"], 1):
        try:
            start, end, value = float(caption["start"]), float(caption["end"]), str(caption["text"]).strip()
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"第 {index} 条字幕缺少时间或文字") from exc
        if not value or PUNCT.search(value) or value[0] in BAD_START or len(value.replace(" ", "")) > MAX_CAPTION_CHARS:
            raise ValueError(f"第 {index} 条字幕不符合文字规则：{value}")
        if start < previous - 0.04 or end <= start or start < 0 or end > duration + 0.25:
            raise ValueError(f"第 {index} 条字幕时间不合法：{start} 到 {end}")
        previous = end
