"""Temporary upload, asynchronous ASR and text model calls."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time
from urllib.parse import quote, urlsplit, urlunsplit
from typing import Callable

import requests

from .models import DEFAULT_TASK_URL, DEFAULT_UPLOAD_URL


DEFAULT_BAILIAN_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_BAILIAN_HYBRID_DEEPSEEK_MODELS = {
    "deepseek-v4.1-flash", "deepseek-v4-pro", "deepseek-v4-flash",
}


class ApiError(RuntimeError):
    pass


class ModelOutputError(ApiError):
    """The API answered, but its text could not be used as a complete object."""


def _visible_address(address: str) -> str:
    parsed = urlsplit(address)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _bailian_api_root(address: str) -> str | None:
    """Recognize an SDK Base URL; leave unrelated custom URLs untouched."""
    parsed = urlsplit(address)
    host = (parsed.hostname or "").lower()
    if not (host.endswith(".maas.aliyuncs.com") or host in {
            "dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"}):
        return None
    path = parsed.path.rstrip("/")
    if path.endswith("/compatible-mode/v1"):
        prefix = path[: -len("/compatible-mode/v1")]
    elif path.endswith("/api/v1"):
        prefix = path[: -len("/api/v1")]
    else:
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, prefix + "/api/v1", "", ""))


def endpoints_from_base_url(address: str) -> tuple[str, str, str, str]:
    """Turn one official Bailian base URL into ASR, upload, task and chat URLs."""
    address = address.strip().rstrip("/")
    parsed = urlsplit(address)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("百炼 URL 请填写 HTTPS Base URL，不要包含密钥、参数或片段")
    host = (parsed.hostname or "").lower()
    if not (host.endswith(".maas.aliyuncs.com") or host in {
            "dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"}):
        raise ValueError("百炼 URL 请填写官方 Base URL")
    path = parsed.path
    if path.endswith("/compatible-mode/v1"):
        prefix = path[:-len("/compatible-mode/v1")]
    elif path.endswith("/api/v1"):
        prefix = path[:-len("/api/v1")]
    elif path in ("", "/"):
        prefix = ""
    else:
        raise ValueError("百炼 URL 请填写域名、/compatible-mode/v1 或 /api/v1，程序会补全接口路径")
    root = urlunsplit((parsed.scheme, parsed.netloc, prefix, "", ""))
    return (root + "/api/v1/services/audio/asr/transcription",
            root + "/api/v1/uploads",
            root + "/api/v1/tasks/{task_id}",
            root + "/compatible-mode/v1/chat/completions")


def saved_base_url(saved: dict) -> str:
    """Read the new setting or migrate an earlier separately configured endpoint."""
    if saved.get("bailian_url"):
        return str(saved["bailian_url"])
    for key, suffix in (("text_url", "/chat/completions"),
                        ("asr_url", "/services/audio/asr/transcription")):
        candidate = str(saved.get(key, "")).strip()
        if candidate.endswith(suffix):
            candidate = candidate[:-len(suffix)]
        try:
            endpoints_from_base_url(candidate)
        except ValueError:
            continue
        return candidate
    return DEFAULT_BAILIAN_URL


def _effective_address(address: str, kind: str) -> str:
    address = address.strip()
    root = _bailian_api_root(address)
    if root:
        if kind == "asr":
            return root + "/services/audio/asr/transcription"
        if kind == "upload":
            return root + "/uploads"
        if kind == "task":
            return root + "/tasks/{task_id}"
        if kind == "text" and address.rstrip("/").endswith("/compatible-mode/v1"):
            return address.rstrip("/") + "/chat/completions"
    if kind == "task":
        parsed = urlsplit(address)
        path = parsed.path
        host = (parsed.hostname or "").lower()
        if (host.endswith(".maas.aliyuncs.com") or host in {
                "dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"}) and "/compatible-mode/v1/tasks/" in path:
            return urlunsplit((parsed.scheme, parsed.netloc,
                               path.replace("/compatible-mode/v1/tasks/", "/api/v1/tasks/", 1),
                               parsed.query, parsed.fragment))
    return address


def _json_response(response: requests.Response, operation: str, address: str) -> dict:
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        # The body can contain request IDs and server diagnostics, but never a
        # bearer token sent in a request header.
        message = response.text[:500].replace("\n", " ")
        raise ApiError(f"{operation} HTTP {response.status_code} · {_visible_address(address)}：{message}") from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise ApiError(f"{operation}返回的内容不是 JSON · {_visible_address(address)}") from exc
    if not isinstance(data, dict):
        raise ApiError("接口返回的 JSON 结构异常")
    return data


def _parse_chat_content(content: object) -> dict:
    """Accept JSON text and OpenAI-compatible text-part responses."""
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        content = "".join(str(part.get("text", "")) for part in content
                          if isinstance(part, dict) and part.get("type") in (None, "text"))
    if not isinstance(content, str):
        raise ValueError("message.content 为空或类型不支持")
    value = content.strip().lstrip("\ufeff").strip()
    if value.startswith("```json") and value.endswith("```"):
        value = value[7:-3].strip()
    elif value.startswith("```") and value.endswith("```"):
        value = value[3:-3].strip()
    result = json.loads(value)
    if not isinstance(result, dict):
        raise ValueError("JSON 顶层不是对象")
    return result


def _direct_json_model_options(address: str, model: str) -> dict:
    """Apply documented thinking switches without guessing a custom API's contract."""
    model = model.lower()
    # Preserve the established Qwen3.7 adapter, including compatible gateways.
    if model.startswith("qwen3.7-"):
        return {"enable_thinking": False}
    host = (urlsplit(address).hostname or "").lower()
    official_host = host.endswith(".maas.aliyuncs.com") or host in {
        "dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com", "dashscope-us.aliyuncs.com",
    }
    if official_host and model in _BAILIAN_HYBRID_DEEPSEEK_MODELS:
        return {"enable_thinking": False}
    return {}


def _content_chars(content: object) -> int:
    """Count response text without recording any response or reasoning content."""
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(part["text"]) for part in content
                   if isinstance(part, dict) and isinstance(part.get("text"), str)
                   and part.get("type") in (None, "text"))
    if isinstance(content, dict):
        return len(json.dumps(content, ensure_ascii=False))
    return 0


def _reasoning_tokens(usage: object) -> int | None:
    if not isinstance(usage, dict):
        return None
    details = usage.get("completion_tokens_details")
    value = details.get("reasoning_tokens") if isinstance(details, dict) else None
    if value is None:
        value = usage.get("reasoning_tokens")
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _attempt_summary(record: dict) -> str:
    """Keep earlier failures visible if the final retry fails for another reason."""
    labels = {"output_truncated": "输出被截断", "timeout": "网络超时",
              "connection_error": "连接失败", "http_transient": "接口暂不可用",
              "http_error": "HTTP 错误", "invalid_response": "响应无效",
              "invalid_json": "JSON 无效", "request_error": "请求失败"}
    parts = []
    for attempt in record["attempts"]:
        reason = labels.get(attempt["error_category"], "成功")
        details = []
        if attempt["finish_reason"]:
            details.append(f"finish_reason={attempt['finish_reason']}")
        if attempt["http_status"] is not None:
            details.append(f"HTTP {attempt['http_status']}")
        if attempt["content_chars"] is not None:
            details.append(f"正文 {attempt['content_chars']} 字符")
        if attempt["reasoning_chars"] is not None:
            details.append(f"思考 {attempt['reasoning_chars']} 字符")
        if attempt["reasoning_tokens"] is not None:
            details.append(f"reasoning_tokens={attempt['reasoning_tokens']}")
        suffix = f"（{'，'.join(details)}）" if details else ""
        parts.append(f"第 {attempt['attempt']} 次：{reason}{suffix}")
    return "；".join(parts)


class BailianClient:
    @classmethod
    def from_base_url(cls, base_url: str, api_key: str, asr_model: str, text_model: str,
                      session: requests.Session | None = None) -> "BailianClient":
        asr, upload, task, text = endpoints_from_base_url(base_url)
        return cls(asr, text, api_key, asr_model, text_model, session,
                   upload_url=upload, task_url=task)

    def __init__(self, asr_url: str, text_url: str, api_key: str, asr_model: str, text_model: str,
                 session: requests.Session | None = None, *,
                 upload_url: str = DEFAULT_UPLOAD_URL, task_url: str = DEFAULT_TASK_URL):
        if not api_key.strip():
            raise ValueError("请填写 API Key")
        if not asr_model.strip() or not text_model.strip():
            raise ValueError("模型名称不能为空")
        known_sync = {"qwen-audio-3.0-asr-flash", "qwen-audio-3.1-asr-flash",
                      "qwen3-asr-flash", "fun-asr-flash"}
        if asr_model.strip() in known_sync:
            replacement = (asr_model.strip() + "-filetrans" if asr_model.startswith("qwen-audio-")
                           else "qwen3-asr-flash-filetrans" if asr_model.startswith("qwen3-")
                           else "fun-asr")
            raise ValueError(f"语音模型 {asr_model.strip()} 是同步模型；当前流程需要异步文件转写模型。可改为 {replacement}")
        self.asr_url = _effective_address(asr_url, "asr")
        self.text_url = _effective_address(text_url, "text")
        self.upload_url = _effective_address(upload_url, "upload")
        self.task_url = _effective_address(task_url, "task")
        self.api_key = api_key.strip()
        self.asr_model = asr_model.strip()
        self.text_model = text_model.strip()
        self.session = session or requests.Session()
        self._request_records: list[dict] = []

    @property
    def request_records(self) -> list[dict]:
        """Serializable text API audit snapshots, without prompts or credentials."""
        return deepcopy(self._request_records)

    @property
    def statistics(self) -> dict:
        """Count every HTTP attempt, including paid but unusable model output."""
        attempts = [attempt for record in self._request_records
                    for attempt in record["attempts"]]
        usage: dict[str, int | None] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            values = [attempt["usage"][key] for attempt in attempts
                      if attempt["usage"][key] is not None]
            # Missing provider usage is unknown, not zero consumption.
            usage[key] = sum(values) if values else None
        return {
            "logical_requests": len(self._request_records),
            "http_attempts": len(attempts),
            "retries": sum(max(0, len(record["attempts"]) - 1)
                           for record in self._request_records),
            "successful_requests": sum(record["status"] == "succeeded"
                                       for record in self._request_records),
            "failed_requests": sum(record["status"] == "failed"
                                   for record in self._request_records),
            "elapsed_seconds": round(sum(record["elapsed_seconds"]
                                         for record in self._request_records), 6),
            "usage": usage,
            "usage_reported_attempts": sum(any(value is not None for value in attempt["usage"].values())
                                           for attempt in attempts),
        }

    @property
    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _get_json(self, url: str, operation: str, *, headers: dict | None = None,
                  timeout: int | tuple[int, int] = 30) -> dict:
        """Retry safe GET requests when transport or server output is transient."""
        for attempt in range(3):
            try:
                response = self.session.get(url, headers=headers, timeout=timeout)
            except (requests.Timeout, requests.ConnectionError) as exc:
                if attempt == 2:
                    raise ApiError(f"{operation}连续三次网络请求失败：{exc.__class__.__name__}") from exc
            else:
                if response.status_code not in {429, 500, 502, 503, 504}:
                    try:
                        return _json_response(response, operation, url)
                    except ApiError:
                        if response.status_code >= 300 or attempt == 2:
                            raise
                elif attempt == 2:
                    raise ApiError(f"{operation}连续三次接口暂不可用：HTTP {response.status_code}")
            time.sleep(attempt + 1)
        raise AssertionError("GET retry loop must return or raise")

    def upload_audio(self, file_path: Path) -> str:
        response = self.session.get(
            self.upload_url, headers=self.auth_headers,
            params={"action": "getPolicy", "model": self.asr_model}, timeout=30,
        )
        policy = _json_response(response, "获取上传凭证", self.upload_url).get("data")
        if not isinstance(policy, dict):
            raise ApiError("未取得临时文件上传凭证")
        try:
            max_bytes = float(policy["max_file_size_mb"]) * 1024 * 1024
            if file_path.stat().st_size > max_bytes:
                raise ApiError(f"提取的音频超过百炼临时上传上限 {policy['max_file_size_mb']} MB")
            key = f"{policy['upload_dir'].rstrip('/')}/{file_path.name}"
            upload_host = policy["upload_host"]
            # requests' multipart mapping preserves insertion order; OSS requires
            # the binary file field after all credential fields.
            with file_path.open("rb") as stream:
                fields = {
                    "OSSAccessKeyId": (None, policy["oss_access_key_id"]),
                    "Signature": (None, policy["signature"]),
                    "policy": (None, policy["policy"]),
                    "x-oss-object-acl": (None, policy["x_oss_object_acl"]),
                    "x-oss-forbid-overwrite": (None, policy["x_oss_forbid_overwrite"]),
                    "key": (None, key),
                    "success_action_status": (None, "200"),
                    "file": (file_path.name, stream, "audio/mpeg"),
                }
                result = self.session.post(upload_host, files=fields, timeout=(30, 300))
            result.raise_for_status()
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError(f"临时文件上传凭证字段缺失或异常：{exc}") from exc
        except requests.HTTPError as exc:
            raise ApiError(f"上传音频失败：HTTP {result.status_code} {result.text[:300]}") from exc
        return f"oss://{key}"

    def submit_asr(self, audio_url: str) -> str:
        parameters: dict = {"channel_id": [0]}
        # Qwen3 uses input.file_url and output.result; the default Qwen-Audio
        # Filetrans/Fun-ASR contract uses input.file_urls and output.results.
        if self.asr_model.startswith("qwen3-asr-"):
            input_data = {"file_url": audio_url}
            parameters["enable_words"] = True
            parameters["enable_itn"] = False
        else:
            input_data = {"file_urls": [audio_url]}
            parameters["language_hints"] = ["zh", "en"]
        response = self.session.post(
            self.asr_url,
            headers={**self.auth_headers, "Content-Type": "application/json",
                     "X-DashScope-Async": "enable", "X-DashScope-OssResourceResolve": "enable"},
            json={"model": self.asr_model, "input": input_data, "parameters": parameters}, timeout=60,
        )
        data = _json_response(response, "提交语音任务", self.asr_url)
        task_id = data.get("output", {}).get("task_id")
        if not task_id:
            raise ApiError(f"提交转写后未返回 task_id：{json.dumps(data, ensure_ascii=False)[:300]}")
        return str(task_id)

    def get_task(self, task_id: str) -> dict:
        encoded_id = quote(task_id, safe="")
        url = (self.task_url.replace("{task_id}", encoded_id) if "{task_id}" in self.task_url
               else self.task_url.rstrip("/") + "/" + encoded_id)
        return self._get_json(url, "查询语音任务", headers=self.auth_headers)

    @staticmethod
    def task_status(data: dict) -> str:
        output = data.get("output") or {}
        return str(output.get("task_status") or data.get("task_status") or "UNKNOWN").upper()

    @staticmethod
    def transcription_url(data: dict) -> str:
        output = data.get("output") or {}
        results = output.get("results")
        if isinstance(results, list) and results:
            result = results[0]
            if result.get("subtask_status", "SUCCEEDED") != "SUCCEEDED":
                raise ApiError(f"音频转写子任务失败：{result.get('message') or result.get('subtask_status')}")
            url = result.get("transcription_url")
        else:
            url = (output.get("result") or {}).get("transcription_url")
        if not url:
            raise ApiError("任务成功但未找到转写结果地址")
        return str(url)

    def download_transcription(self, url: str, destination: Path) -> dict:
        # A successful HTTP response can briefly contain an incomplete ASR
        # document. Validate before caching so a retry does not reuse it.
        from .captions import extract_words
        from .sentence_flow import asr_sentences
        last_problem = ""
        for attempt in range(3):
            data = self._get_json(url, "下载转写结果", timeout=(30, 90))
            try:
                asr_sentences(data, extract_words(data))
                break
            except (TypeError, ValueError, KeyError) as exc:
                last_problem = str(exc)
                if attempt == 2:
                    raise ApiError(f"语音转写结果连续三次结构无效：{last_problem}") from exc
                time.sleep(attempt + 1)
        destination.parent.mkdir(parents=True, exist_ok=True)
        pending = destination.with_suffix(destination.suffix + ".tmp")
        pending.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        pending.replace(destination)
        return data

    def chat_json(self, system: str, user: str,
                  on_retry: Callable[[int, str], None] | None = None, *, phase: str = "",
                  max_tokens: int | None = None) -> dict:
        """Request a JSON object and retain safe accounting for all attempts."""
        if max_tokens is not None and (isinstance(max_tokens, bool)
                                       or not isinstance(max_tokens, int) or max_tokens <= 0):
            raise ValueError("文本模型 max_tokens 必须是正整数")
        record = {"request_number": len(self._request_records) + 1,
                  "operation": "text_model", "phase": str(phase).replace(self.api_key, "[redacted]"),
                  "model": self.text_model, "status": "running", "error_category": "",
                  "input_chars": len(system) + len(user),
                  "elapsed_seconds": 0.0, "attempts": []}
        self._request_records.append(record)
        started = time.monotonic()
        try:
            result = self._chat_json_request(system, user, on_retry, record, max_tokens)
            record["status"] = "succeeded"
            return result
        except Exception as exc:
            record["status"] = "failed"
            record["error_category"] = (record["attempts"][-1]["error_category"]
                                        if record["attempts"] else "request_error") or "request_error"
            if isinstance(exc, ModelOutputError) and record["attempts"]:
                raise ModelOutputError(f"{str(exc).replace(self.api_key, '[redacted]')}。"
                                       f"尝试摘要：{_attempt_summary(record)}") from exc
            raise
        finally:
            record["elapsed_seconds"] = round(time.monotonic() - started, 6)

    def _chat_json_request(self, system: str, user: str,
                           on_retry: Callable[[int, str], None] | None, record: dict,
                           max_tokens: int | None) -> dict:
        if max_tokens is None:
            budgets = (2048, 4096, 8192)
        else:
            ceiling = max(16384, max_tokens)
            budgets = (max_tokens, min(max_tokens * 2, ceiling), min(max_tokens * 4, ceiling))
        payload = {"model": self.text_model,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                   "temperature": 0, "response_format": {"type": "json_object"},
                   "max_tokens": budgets[0], "stream": False}
        # Hybrid models can spend the output budget on thinking and return no
        # JSON body. Use direct output only where the provider documents it.
        payload.update(_direct_json_model_options(self.text_url, self.text_model))
        last_problem = ""
        for attempt, budget in enumerate(budgets):
            payload["max_tokens"] = budget
            entry = {"attempt": attempt + 1, "is_retry": attempt > 0,
                     "max_tokens": budget, "http_status": None, "status": "failed",
                     "error_category": "request_error", "elapsed_seconds": 0.0,
                     "finish_reason": None, "request_id": None,
                     "input_chars": record["input_chars"], "content_chars": None,
                     "reasoning_chars": None, "reasoning_tokens": None,
                     "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}}
            record["attempts"].append(entry)
            attempt_started = time.monotonic()
            try:
                response = self.session.post(
                    self.text_url, headers={**self.auth_headers, "Content-Type": "application/json"},
                    json=payload, timeout=(30, 90),
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                entry["error_category"] = ("timeout" if isinstance(exc, requests.Timeout)
                                           else "connection_error")
                entry["elapsed_seconds"] = round(time.monotonic() - attempt_started, 6)
                last_problem = f"网络请求失败（{exc.__class__.__name__}）"
                if attempt == 2:
                    raise ModelOutputError("文本模型连续请求失败；已完成的字幕段已保存，可稍后继续") from exc
            else:
                entry["http_status"] = response.status_code
                entry["elapsed_seconds"] = round(time.monotonic() - attempt_started, 6)
                if response.status_code in {429, 500, 502, 503, 504}:
                    entry["error_category"] = "http_transient"
                    last_problem = f"接口暂不可用（HTTP {response.status_code}）"
                    if attempt == 2:
                        raise ModelOutputError(f"文本模型连续三次暂不可用：{last_problem}")
                    if on_retry:
                        on_retry(attempt + 1, last_problem)
                    time.sleep(1)
                    continue
                try:
                    data = _json_response(response, "文本模型校对", self.text_url)
                except ApiError as exc:
                    entry["error_category"] = ("http_error" if response.status_code >= 300
                                               else "invalid_response")
                    if response.status_code >= 300:
                        raise
                    last_problem = str(exc)
                    if attempt == 2:
                        raise ModelOutputError(f"文本模型连续三次返回无效响应：{last_problem}") from exc
                    if on_retry:
                        on_retry(attempt + 1, last_problem)
                    time.sleep(1)
                    continue
                choices = data.get("choices")
                choice = choices[0] if isinstance(choices, list) and choices else {}
                choice = choice if isinstance(choice, dict) else {}
                message = choice.get("message")
                message = message if isinstance(message, dict) else {}
                content = message.get("content")
                finish_reason = str(choice.get("finish_reason") or "未知")
                request_id = str(data.get("request_id") or data.get("id") or "未提供").replace(
                    self.api_key, "[redacted]")
                entry["finish_reason"] = finish_reason
                entry["request_id"] = request_id
                usage = data.get("usage")
                if isinstance(usage, dict):
                    entry["usage"] = {key: value if isinstance(value := usage.get(key), int)
                                      and not isinstance(value, bool) and value >= 0 else None
                                      for key in entry["usage"]}
                content_size = _content_chars(content)
                entry["content_chars"] = content_size
                entry["reasoning_chars"] = _content_chars(message.get("reasoning_content"))
                entry["reasoning_tokens"] = _reasoning_tokens(usage)
                # Truncation also makes JSON parsing fail. Diagnose the provider's
                # finish reason first so callers see the actual cause and retry.
                if finish_reason == "length":
                    entry["error_category"] = "output_truncated"
                    last_problem = (f"输出被截断（finish_reason=length，内容长度 {content_size}，"
                                    f"请求 ID {request_id}）")
                else:
                    try:
                        result = _parse_chat_content(content)
                    except (TypeError, ValueError) as exc:
                        entry["error_category"] = "invalid_json"
                        last_problem = (f"返回内容不是有效 JSON 对象（{exc.__class__.__name__}，"
                                        f"finish_reason={finish_reason}，内容长度 {content_size}，"
                                        f"请求 ID {request_id}）")
                    else:
                        entry["status"] = "succeeded"
                        entry["error_category"] = ""
                        return result
            if attempt < 2:
                if on_retry:
                    on_retry(attempt + 1, last_problem)
                time.sleep(1)
        raise ModelOutputError(f"文本模型连续三次未返回完整 JSON：{last_problem}。已完成的字幕段已保存，可继续所选")
