"""One-video pipeline with resumable ASR and explicit progress events."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import threading
import time
from uuid import uuid4
from typing import Callable

from .bailian import ApiError, BailianClient
from .brightness import analyze_brightness
from .alignment import align_captions, build_timeline
from .captions import MAX_CAPTION_CHARS, extract_words, has_estimated_timing, shorten_captions
from .edits import detect_silences, no_window, output_duration, retime_captions
from .models import VideoItem
from .prompts import PromptSet, load_prompts
from .render import probe, render_video
from .qc import inspect_output
from .sentence_flow import asr_sentences, build_sentence_project
from .model_payload import compact_model_input
from .model_plan import compile_model_plan
from .strategy import Strategy
from .caption_style import CaptionStyle
from .topic_style import TopicStyle
from .caption_edits import load_edits, to_source_times


class Cancelled(RuntimeError):
    pass


CAPTION_REVISION = 19


@dataclass(frozen=True)
class JobConfig:
    template: Path
    output_dir: Path
    notice: str
    slot: tuple[int, int, int, int] | None
    ffmpeg: str
    ffprobe: str
    font: Path | None = None
    output_tag: str = ""
    strategy: Strategy = Strategy()
    topic: str = ""
    topic_box: tuple[int, int, int, int] | None = None
    nameplate_style: str = "classic"
    brightness: str = "normal"
    quality: str = "balanced"
    brightness_target: int = 100
    topic_color: str = "#FFFFFF"
    topic_font: Path | None = None
    prompts: PromptSet = field(default_factory=load_prompts)
    caption_style: CaptionStyle = field(default_factory=CaptionStyle)
    nameplate_scale: float = 1.0
    nameplate_box: tuple[int, int, int, int] | None = None
    topic_style: TopicStyle = field(default_factory=TopicStyle)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(path)


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (ValueError, OSError):
        return None


def work_dir(item: VideoItem, output_dir: Path) -> Path:
    key = str(item.path.resolve()).casefold().encode("utf-8")
    identity = hashlib.sha256(key).hexdigest()[:12]
    return output_dir / ".vertical-video-studio" / f"{item.path.stem[:40]}-{identity}"


def captions_path(item: VideoItem, output_dir: Path) -> Path:
    return work_dir(item, output_dir) / "captions.json"


def output_path(item: VideoItem, output_dir: Path, output_tag: str = "") -> Path:
    invalid = str.maketrans('<>:"/\\|?*', '＜＞：＂／＼｜？＊')
    def part(value: str) -> str:
        safe = value.strip().translate(invalid).replace("__", "＿")
        safe = re.sub(r"[\x00-\x1f]", " ", safe).rstrip(" .")
        if not safe:
            raise ValueError("单位、头衔或姓名不能只包含点和空格")
        return safe
    if output_tag and not re.fullmatch(r"\d{8}-\d{6}-\d{6}", output_tag):
        raise ValueError("输出时间标识格式错误")
    filename = "_".join(part(value) for value in
                         (item.person.organization, item.person.title, item.person.name))
    if item.clip_id:
        filename += "_" + part(item.clip_id)
    filename += f"_{output_tag}.mp4" if output_tag else ".mp4"
    target = output_dir / filename
    if target.resolve() == item.path.resolve():
        target = output_dir / "竖屏成片" / filename
    return target


def _run_command(command: list[str], cancel: threading.Event, timeout: float) -> None:
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", errors="replace",
                               creationflags=no_window())
    started = time.monotonic()
    try:
        while True:
            if cancel.is_set():
                process.terminate()
                raise Cancelled("用户已取消")
            try:
                _, error = process.communicate(timeout=0.5)
                if process.returncode:
                    raise RuntimeError(f"FFmpeg 处理失败：{error[-1000:]}")
                return
            except subprocess.TimeoutExpired:
                if time.monotonic() - started > timeout:
                    process.terminate()
                    raise RuntimeError("FFmpeg 音频提取超时")
    finally:
        if process.poll() is None:
            process.terminate()


def extract_audio(video: Path, destination: Path, ffmpeg: str, cancel: threading.Event,
                  duration: float) -> None:
    _run_command([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
                  "-vn", "-ac", "1", "-ar", "16000", "-b:a", "48k", str(destination)],
                 cancel, max(90, duration * 3))
    if not destination.is_file() or destination.stat().st_size < 1000:
        raise RuntimeError("没有提取到有效音频，请检查原视频音轨")


class Pipeline:
    def __init__(self, client: BailianClient | None, config: JobConfig,
                 emit: Callable[[str, int, str], None], cancel: threading.Event,
                 *, render_only: bool = False):
        self.client = client
        self.config = config
        self.emit = emit
        self.cancel = cancel
        self.last_run: dict | None = None
        self.render_only = render_only

    def _check_cancel(self) -> None:
        if self.cancel.is_set():
            raise Cancelled("用户已取消")

    def run_one(self, item: VideoItem) -> Path:
        """Persist every attempt, including failures and resumed attempts."""
        started = time.monotonic()
        request_offset = len(self.client.request_records) if self.client and hasattr(self.client, "request_records") else 0
        self._active_run_id = uuid4().hex
        self._active_plan_path = None
        self._asr_cache_reused = False
        self.review_count = 0
        entry = {"started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                 "run_id": self._active_run_id, "plan_ready": False,
                 "status": "进行中", "kind": "字幕重新导出" if self.render_only else "自动制作"}
        log_path = work_dir(item, self.config.output_dir) / "runs.json"
        log = _read_json(log_path) or {"runs": []}
        if not isinstance(log.get("runs"), list):
            log = {"runs": []}
        for previous in log["runs"]:
            if isinstance(previous, dict) and previous.get("status") == "进行中":
                previous["status"] = "意外中断"
        log["runs"].append(entry)
        _write_json(log_path, log)
        try:
            target = self._run_one(item)
            entry["status"] = "完成"
            entry["output"] = str(target)
            entry["quality_status"] = getattr(self, "quality_status", "可交付")
            entry["review_count"] = getattr(self, "review_count", 0)
            return target
        except Cancelled:
            entry["status"] = "已取消"
            raise
        except Exception as exc:
            entry["status"] = "失败"
            entry["error"] = str(exc).replace(self.client.api_key, "***") if self.client and self.client.api_key else str(exc)
            raise
        finally:
            entry["ended_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            entry["elapsed_seconds"] = round(time.monotonic() - started, 2)
            entry["asr_cache_reused"] = self._asr_cache_reused
            entry["plan_ready"] = self._active_plan_path is not None
            if self._active_plan_path is not None:
                entry["plan_path"] = str(self._active_plan_path)
            self.last_run = dict(entry)
            if self.client and hasattr(self.client, "request_records"):
                requests = self.client.request_records[request_offset:]
                entry["text_api_logical_requests"] = len(requests)
                entry["text_api_http_attempts"] = sum(len(r["attempts"]) for r in requests)
                _write_json(log_path.parent / "model-calls.json", {"requests": requests,
                            "statistics": self.client.statistics})
                self.last_run = dict(entry)
            _write_json(log_path, log)

    def _run_one(self, item: VideoItem) -> Path:
        self._check_cancel()
        if not item.path.is_file():
            raise ValueError(f"视频不存在：{item.path}")
        if not self.config.template.is_file():
            raise ValueError("模板图片不存在")
        duration, width, height = probe(item.path, self.config.ffprobe)
        if width <= height:
            raise ValueError("输入视频必须是横屏")
        if duration <= 0:
            raise ValueError("无法读取视频时长")
        folder = work_dir(item, self.config.output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        if self.render_only:
            caption_data = _read_json(folder / "captions.json")
            raw = _read_json(folder / "transcription.json")
            if not caption_data or not raw or not caption_data.get("edit_plan"):
                raise ValueError("请先生成字幕，再编辑和重新导出")
            saved_source = caption_data["edit_plan"].get("source", {})
            if (saved_source.get("size") != item.path.stat().st_size or
                    saved_source.get("mtime_ns") != item.path.stat().st_mtime_ns):
                raise ValueError("原视频已变化，请重新制作后再编辑字幕")
            self.emit("edit", 70, "复用已有剪辑和字幕，准备重新导出")
            return self._render_saved(item, caption_data, raw, folder, duration)
        _write_json(folder / f"prompts-{self.config.prompts.digest()}.json",
                    {"sha256": self.config.prompts.digest(),
                     **self.config.prompts.snapshot()})
        raw_path = folder / "transcription.json"
        state_path = folder / "task.json"
        caption_file = folder / "captions.json"
        partial_file = folder / "captions.partial.json"
        fingerprint = {"path": str(item.path.resolve()), "size": item.path.stat().st_size,
                       "mtime_ns": item.path.stat().st_mtime_ns,
                       "asr_url": self.client.asr_url, "upload_url": self.client.upload_url,
                       "task_url": self.client.task_url, "asr_model": self.client.asr_model}
        state = _read_json(state_path) or {}
        if state.get("fingerprint") != fingerprint:
            state = {"fingerprint": fingerprint}
            raw_path.unlink(missing_ok=True)
            caption_file.unlink(missing_ok=True)
            partial_file.unlink(missing_ok=True)
            _write_json(state_path, state)

        raw = _read_json(raw_path)
        if raw is not None:
            try:
                asr_sentences(raw, extract_words(raw))
            except (TypeError, ValueError, KeyError):
                raw_path.unlink(missing_ok=True)
                raw = None
        if raw is None:
            task_id = state.get("task_id")
            if not task_id:
                self.emit("extract", 5, "提取音频")
                with tempfile.TemporaryDirectory(prefix="asr-audio-") as temporary:
                    audio = Path(temporary) / f"{hashlib.sha256(str(item.path).encode()).hexdigest()[:16]}.mp3"
                    extract_audio(item.path, audio, self.config.ffmpeg, self.cancel, duration)
                    self._check_cancel()
                    self.emit("upload", 13, "上传音频到百炼临时存储")
                    audio_url = self.client.upload_audio(audio)
                    self._check_cancel()
                    self.emit("submit", 20, "提交语音识别任务")
                    task_id = self.client.submit_asr(audio_url)
                    state["task_id"] = task_id
                    state["submitted_at"] = time.time()
                    _write_json(state_path, state)
            self.emit("asr", 25, f"识别中 · 任务 {task_id}")
            started = time.monotonic()
            while True:
                self._check_cancel()
                task = self.client.get_task(str(task_id))
                status = self.client.task_status(task)
                if status == "SUCCEEDED":
                    url = self.client.transcription_url(task)
                    self.emit("download", 42, "下载语音识别结果")
                    try:
                        raw = self.client.download_transcription(url, raw_path)
                    except ApiError as exc:
                        if "HTTP 403" in str(exc) or "HTTP 404" in str(exc):
                            state.pop("task_id", None)
                            _write_json(state_path, state)
                            raise ApiError("转写结果下载地址已失效；再次运行将重新提交语音任务") from exc
                        raise
                    break
                if status in {"FAILED", "CANCELED", "CANCELLED"}:
                    state.pop("task_id", None)
                    _write_json(state_path, state)
                    details = (task.get("output") or {}).get("message") or task.get("message") or status
                    raise ApiError(f"语音识别任务失败：{details}")
                if status == "UNKNOWN":
                    state.pop("task_id", None)
                    _write_json(state_path, state)
                    raise ApiError("百炼返回 UNKNOWN，任务 ID 已失效；再次运行将重新提交")
                if status not in {"PENDING", "RUNNING"}:
                    raise ApiError(f"未知语音识别状态：{status}")
                elapsed = int(time.monotonic() - started)
                self.emit("asr", 25, f"识别中 · 已等待 {elapsed} 秒 · {status}")
                if elapsed > 4 * 3600:
                    raise ApiError("语音识别等待超过 4 小时，任务 ID 已保存，可稍后继续")
                if self.cancel.wait(4):
                    raise Cancelled("用户已取消；已保存任务 ID，下次可继续查询")
        else:
            self._asr_cache_reused = True
            self.emit("download", 42, "使用已保存的识别结果")
        self.emit("recognized", 45, "语音识别完成")
        self._check_cancel()

        context = {"organization": item.person.organization, "title": item.person.title,
                   "name": item.person.name, "text_url": self.client.text_url,
                   "text_model": self.client.text_model,
                   "strategy_sha256": self.config.strategy.digest(),
                   "prompts_sha256": self.config.prompts.digest()}
        caption_data = _read_json(caption_file)
        if (caption_data is None or caption_data.get("context") != context
                or not isinstance(caption_data.get("edit_plan"), dict)
                or (not caption_data.get("human_reviewed")
                    and caption_data.get("caption_revision") != CAPTION_REVISION)):
            self.emit("edit", 48, "检测音轨静音，提供模型可选气口时间")
            silences = detect_silences(str(item.path), self.config.ffmpeg, duration)
            self.emit("llm", 50, "结合全文分析改口、重说与废弃起句")
            partial_key = {"context": context,
                           "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                           "revision": CAPTION_REVISION}
            saved_partial = _read_json(partial_file) or {}
            resume = saved_partial.get("progress") if saved_partial.get("key") == partial_key else None

            def save_text_stage(value):
                _write_json(partial_file, {"key": partial_key, "progress": value})
                _write_json(folder / "edit-attempts.json", {"run_id": self._active_run_id,
                            "attempts": value.get("edit_attempts", [])})
                _write_json(folder / "review-attempts.json", {"run_id": self._active_run_id,
                            "attempts": value.get("review_attempts", [])})
                _write_json(folder / "model-edit-plan.json", {
                    "run_id": self._active_run_id, "prompts_sha256": self.config.prompts.digest(),
                    "edited_groups": value.get("edited", 0), "reviewed_groups": value.get("reviewed", 0),
                    "draft_plans": value.get("draft_plans", []),
                    "final_plans": value.get("final_model_plans", [])})

            caption_data = build_sentence_project(
                raw, item.person,
                lambda system, user: self.client.chat_json(
                    system, compact_model_input(user),
                    on_retry=lambda attempt, reason: self.emit(
                        "llm", 50, f"文本模型响应异常，正在重试 {attempt}/2 · {reason}"),
                    phase=("edit" if system == self.config.prompts.edit else
                           "semantic_review" if system == self.config.prompts.review else "caption"), max_tokens=8192),
                lambda current, total: self.emit(
                    "llm", 50 + round(20 * current / total),
                    (f"分析剪辑初稿 {current}/{total // 3}" if current <= total // 3
                     else f"复审误删与漏剪 {current - total // 3}/{total // 3}" if current <= total * 2 // 3
                     else f"生成保留语音字幕 {current - total * 2 // 3}/{total // 3}")),
                resume=resume,
                checkpoint=save_text_stage,
                prompts=self.config.prompts,
                strategy=self.config.strategy,
                silences=silences,
                strict_model_failures=True,
                model_authored=True,
                model_plan_builder=lambda proposals, pauses: compile_model_plan(
                    extract_words(raw), proposals, duration, silences, pauses,
                    estimated_timing=has_estimated_timing(raw), strategy=self.config.strategy),
            )
            caption_data["context"] = context
            caption_data["caption_revision"] = CAPTION_REVISION
            _write_json(caption_file, caption_data)
            _write_json(folder / "caption-attempts.json", {"attempts": caption_data.get("caption_attempts", [])})
            _write_json(folder / "edit-attempts.json", {"attempts": caption_data.get("edit_attempts", [])})
            _write_json(folder / "review-attempts.json", {"run_id": self._active_run_id,
                        "attempts": caption_data.get("review_attempts", [])})
            _write_json(folder / "model-edit-plan.json", {"run_id": self._active_run_id,
                        "prompts_sha256": self.config.prompts.digest(),
                        **caption_data.get("model_edit_plan", {})})
            partial_file.unlink(missing_ok=True)
        else:
            self.emit("llm", 70, "使用已保存的字幕")
        return self._render_saved(item, caption_data, raw, folder, duration)

    def _render_saved(self, item, caption_data, raw, folder, duration):
        caption_file = folder / "captions.json"
        if caption_data.get("asr_sentences"):
            _write_json(folder / "sentences.json",
                        {"source_video": str(item.path),
                         "sentences": caption_data["asr_sentences"]})
        if not self.render_only and any(len(str(caption.get("text", "")).replace(" ", "")) > MAX_CAPTION_CHARS
               for caption in caption_data.get("captions", [])):
            caption_data["captions"] = shorten_captions(
                caption_data["captions"], extract_words(raw),
                (item.person.organization, item.person.name, item.person.title))
            _write_json(caption_file, caption_data)
        words = extract_words(raw)
        self.emit("edit", 70, "执行模型确认的裁剪并对齐字幕与语音")
        edit_plan = caption_data["edit_plan"]
        edit_plan["source"] = {"path": str(item.path.resolve()),
                               "size": item.path.stat().st_size,
                               "mtime_ns": item.path.stat().st_mtime_ns}
        edit_plan["template"] = {"path": str(self.config.template.resolve()),
                                 "slot": self.config.slot,
                                 "topic": item.topic or self.config.topic,
                                 "topic_box": self.config.topic_box,
                                 "nameplate_style": item.nameplate_style or self.config.nameplate_style,
                                 "nameplate_scale": self.config.nameplate_scale,
                                 "nameplate_box": self.config.nameplate_box,
                                 "brightness": item.brightness or self.config.brightness,
                                 "brightness_target": self.config.brightness_target,
                                 "topic_color": self.config.topic_color,
                                 "topic_font": str(self.config.topic_font) if self.config.topic_font else "",
                                 "quality": self.config.quality}
        edit_plan["template"]["caption_style"] = self.config.caption_style.snapshot()
        edit_plan["template"]["topic_style"] = self.config.topic_style.snapshot()
        edit_plan["uncertain"] = caption_data.get("uncertain") or []
        if not self.render_only:
            edit_plan["prompts_sha256"] = self.config.prompts.digest()
        self.review_count = int(edit_plan.get("review_count", 0)) + len(edit_plan["uncertain"])
        self.quality_status = "待复核" if self.review_count else "可交付"
        cuts = edit_plan["cuts"]
        rejected = caption_data.get("rejected_model_proposals") or []
        edit_plan["processing_summary"] = {
            "model_proposal_count": len(caption_data.get("model_cut_proposals") or []) + len(rejected),
            "adopted_model_proposal_count": len(edit_plan.get("applied_proposals") or []),
            "rejected_model_proposal_count": len(rejected),
            "removed_seconds": round(sum(cut["end"] - cut["start"] for cut in cuts), 3),
            "caption_count": len(caption_data["captions"]),
            "caption_literal_fallback_count": sum(bool(c.get("caption_fallback")) for c in caption_data["captions"]),
            "caption_formatting_repair_count": len(caption_data.get("caption_formatting_repairs") or []),
            "semantic_review_group_count": len((caption_data.get("model_edit_plan") or {}).get("final_plans", [])),
            "model_pause_compression_count": len(edit_plan.get("applied_pause_compressions") or []),
        }
        edit_plan["run_id"] = self._active_run_id
        _write_json(folder / "edit-plan.json", edit_plan)
        if caption_data.get("model_edit_plan"):
            _write_json(folder / "model-edit-plan.json", {"run_id": self._active_run_id,
                        "prompts_sha256": caption_data.get("context", {}).get("prompts_sha256"),
                        **caption_data["model_edit_plan"]})
        self._active_plan_path = folder / "edit-plan.json"
        if not self.render_only:
            caption_data["captions"] = align_captions(caption_data["captions"], words, cuts)
        caption_data["alignment_revision"] = 1
        caption_data["cuts"] = cuts
        _write_json(caption_file, caption_data)
        target = output_path(item, self.config.output_dir, self.config.output_tag)
        overrides = load_edits(folder, caption_data)
        timeline_captions = to_source_times(overrides, cuts, duration) if overrides else caption_data["captions"]
        if overrides:
            for caption in timeline_captions:
                indices = [index for index, word in enumerate(words)
                           if word.end > caption["start"] and word.start < caption["end"]]
                if indices:
                    caption["start_word"], caption["end_word"] = indices[0], indices[-1] + 1
        timeline = build_timeline(item.path, target, words, timeline_captions, cuts, duration)
        _write_json(folder / "caption_timeline.json", timeline)
        rendered_file = folder / "captions.rendered.json"
        rendered_captions = {**caption_data, "captions": overrides or retime_captions(caption_data["captions"], cuts),
                             "source_duration": duration, "output_duration": output_duration(cuts, duration),
                             "manual_edits_applied": bool(overrides),
                             "timeline_file": str(folder / "caption_timeline.json")}
        _write_json(rendered_file, rendered_captions)
        if overrides:
            self.emit("edit", 71, f"应用 {len(overrides)} 条已编辑字幕，音视频剪辑保持原样")
        elif caption_data.get("caption_fallback_groups"):
            self.emit("edit", 71, f"注意：{len(caption_data['caption_fallback_groups'])} 组字幕使用识别原文，详见结果记录")
        if (folder / "caption-overrides.json").is_file() and overrides is None:
            self.emit("edit", 71, "自动字幕已变化；旧手动修改保留在目录中，未套用于这次成片")
        removed_seconds = duration - rendered_captions["output_duration"]
        self.emit("edit", 71, f"剪掉 {len(cuts)} 处气口、空白或口癖，共 {removed_seconds:.1f} 秒")
        self._check_cancel()
        brightness_mode = item.brightness or self.config.brightness
        brightness_offset = None
        if brightness_mode == "auto":
            self.emit("render", 72, "检测原视频明暗")
            brightness_analysis = analyze_brightness(item.path, duration, self.config.ffmpeg,
                                                     self.config.brightness_target)
            brightness_offset = brightness_analysis["offset"]
            edit_plan["brightness_analysis"] = brightness_analysis
            _write_json(folder / "brightness.json", brightness_analysis)
            self._check_cancel()
        _write_json(folder / "edit-plan.json", edit_plan)
        self.emit("render", 72, "合成竖屏视频")
        render_video(item.path, self.config.template, rendered_file, target,
                     item.person.name, item.person.organization, item.person.title,
                     self.config.notice, self.config.slot, self.config.font,
                     ffmpeg_path=self.config.ffmpeg, ffprobe_path=self.config.ffprobe,
                     progress=lambda fraction: self.emit("render", 72 + round(27 * fraction),
                                                         f"合成视频 {fraction:.0%}"),
                     cancel_event=self.cancel, cuts=cuts,
                     topic=item.topic or self.config.topic,
                     topic_box=self.config.topic_box,
                     nameplate_style=item.nameplate_style or self.config.nameplate_style,
                     brightness=brightness_mode, brightness_offset=brightness_offset,
                     brightness_target=self.config.brightness_target,
                     topic_color=self.config.topic_color, topic_font_path=self.config.topic_font,
                     quality=self.config.quality, caption_style=self.config.caption_style,
                     nameplate_scale=self.config.nameplate_scale, nameplate_box=self.config.nameplate_box,
                     topic_style=self.config.topic_style)
        self.emit("qc", 99, "检测成片时长与音视频流")
        qc = inspect_output(target, duration, cuts, self.config.ffprobe,
                            max_removal_ratio=self.config.strategy.max_removal_ratio,
                            min_retained_fragment=self.config.strategy.min_retained_fragment,
                            policy_signals=edit_plan.get("mode") != "model_authored")
        _write_json(folder / "qc.json", qc)
        if qc["findings"]:
            self.review_count += len(qc["findings"])
            self.quality_status = "待复核"
        self.emit("done", 100, f"完成：{target.name}")
        return target
