from __future__ import annotations

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import requests

from studio.bailian import BailianClient, ModelOutputError


class Response:
    status_code = 200

    def __init__(self, content: object = "{}", *, finish: str = "stop",
                 reasoning: object = None, usage: dict | None = None):
        self.data = {"id": "request-123", "choices": [{"finish_reason": finish,
                     "message": {"content": content, "reasoning_content": reasoning}}]}
        if usage is not None:
            self.data["usage"] = usage
        self.text = json.dumps(self.data)

    def raise_for_status(self):
        pass

    def json(self):
        return self.data


class Session:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, deepcopy(kwargs)))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class ReasoningAdapterTests(unittest.TestCase):
    def setUp(self):
        sleeper = patch("studio.bailian.time.sleep")
        sleeper.start()
        self.addCleanup(sleeper.stop)

    @staticmethod
    def client(session: Session, model: str, url: str =
               "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"):
        return BailianClient("https://asr.example.invalid", url, "private-key",
                             "fun-asr", model, session)

    def test_documented_bailian_hybrid_models_disable_thinking(self):
        for model in ("deepseek-v4.1-flash", "deepseek-v4-flash", "deepseek-v4-pro"):
            for host in ("dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com",
                         "llm-test.cn-beijing.maas.aliyuncs.com"):
                with self.subTest(model=model, host=host):
                    session = Session(Response())
                    self.client(session, model, f"https://{host}/compatible-mode/v1").chat_json(
                        "system", "user")
                    payload = session.calls[0][1]["json"]
                    self.assertIs(payload["enable_thinking"], False)
                    self.assertIs(payload["stream"], False)
                    self.assertEqual(session.calls[0][1]["timeout"], (30, 90))

    def test_unknown_thinking_only_and_custom_provider_do_not_receive_switch(self):
        for model, url in (
                ("deepseek-r1", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                ("deepseek-v4.1-thinking", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                ("unknown-model", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
                ("deepseek-v4.1-flash", "https://api.deepseek.com/v1/chat/completions"),
                ("deepseek-v4.1-flash", "https://dashscope.aliyuncs.com.evil.invalid/v1")):
            with self.subTest(model=model, url=url):
                session = Session(Response())
                self.client(session, model, url).chat_json("system", "user")
                self.assertNotIn("enable_thinking", session.calls[0][1]["json"])

    def test_existing_qwen37_gateway_adapter_is_preserved(self):
        session = Session(Response())
        self.client(session, "qwen3.7-flash-2026-07-15",
                    "https://custom.example.invalid/v1").chat_json("system", "user")
        self.assertIs(session.calls[0][1]["json"]["enable_thinking"], False)

    def test_audit_counts_text_and_reasoning_without_storing_them(self):
        content = '{"cuts": []}'
        session = Session(Response(content, reasoning="private model reasoning", usage={
            "prompt_tokens": 14, "completion_tokens": 30, "total_tokens": 44,
            "completion_tokens_details": {"reasoning_tokens": 19},
        }))
        client = self.client(session, "deepseek-v4.1-flash")
        client.chat_json("private system", "private input", phase="edit")
        record = client.request_records[0]
        attempt = record["attempts"][0]
        self.assertEqual(record["input_chars"], len("private systemprivate input"))
        self.assertEqual(attempt["input_chars"], record["input_chars"])
        self.assertEqual(attempt["content_chars"], len(content))
        self.assertEqual(attempt["reasoning_chars"], len("private model reasoning"))
        self.assertEqual(attempt["reasoning_tokens"], 19)
        audit = json.dumps(client.request_records)
        for private in ("private-key", "private system", "private input", "private model reasoning", content):
            self.assertNotIn(private, audit)

    def test_length_then_timeouts_retain_complete_failure_summary(self):
        session = Session(Response(None, finish="length", reasoning="reason" * 500, usage={
            "completion_tokens": 2048, "completion_tokens_details": {"reasoning_tokens": 2048}}),
            requests.ReadTimeout("private transport detail"), requests.ReadTimeout("private-key"))
        client = self.client(session, "deepseek-v4.1-flash")
        with self.assertRaises(ModelOutputError) as caught:
            client.chat_json("system", "user", phase="edit")
        message = str(caught.exception)
        self.assertIn("第 1 次：输出被截断", message)
        self.assertIn("finish_reason=length", message)
        self.assertIn("正文 0 字符", message)
        self.assertIn("思考 3000 字符", message)
        self.assertIn("reasoning_tokens=2048", message)
        self.assertIn("第 2 次：网络超时", message)
        self.assertIn("第 3 次：网络超时", message)
        self.assertNotIn("private transport detail", message)
        self.assertNotIn("private-key", message)
        record = client.request_records[0]
        self.assertEqual(record["status"], "failed")
        self.assertEqual([attempt["error_category"] for attempt in record["attempts"]],
                         ["output_truncated", "timeout", "timeout"])
        self.assertEqual([call[1]["json"]["max_tokens"] for call in session.calls],
                         [2048, 4096, 8192])

    def test_reasoning_token_count_is_optional_and_validated(self):
        for usage, expected in ((None, None), ({"reasoning_tokens": 7}, 7),
                                ({"completion_tokens_details": {"reasoning_tokens": True}}, None),
                                ({"completion_tokens_details": {"reasoning_tokens": -1}}, None)):
            with self.subTest(usage=usage):
                session = Session(Response(usage=usage))
                client = self.client(session, "deepseek-v4.1-flash")
                client.chat_json("system", "user")
                self.assertEqual(client.request_records[0]["attempts"][0]["reasoning_tokens"], expected)


if __name__ == "__main__":
    unittest.main()
