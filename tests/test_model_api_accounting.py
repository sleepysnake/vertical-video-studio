from __future__ import annotations

from copy import deepcopy
import json
import unittest
from unittest.mock import patch

import requests

from studio.bailian import ApiError, BailianClient, ModelOutputError


class Response:
    def __init__(self, data: dict, status: int = 200):
        self.data = data
        self.status_code = status
        self.text = json.dumps(data)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

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


def reply(content: str = '{}', finish_reason: str = "stop", usage: dict | None = None,
          request_id: str = "req-test") -> Response:
    data = {"id": request_id, "choices": [{"finish_reason": finish_reason,
                                          "message": {"content": content}}]}
    if usage is not None:
        data["usage"] = usage
    return Response(data)


class ModelApiAccountingTests(unittest.TestCase):
    def setUp(self):
        sleeper = patch("studio.bailian.time.sleep")
        sleeper.start()
        self.addCleanup(sleeper.stop)

    @staticmethod
    def client(session: Session, model: str = "custom-model") -> BailianClient:
        return BailianClient("http://localhost:8900/custom-asr",
                             "http://localhost:8900/custom-text", "test-secret-key",
                             "custom-asr", model, session)

    def test_success_audits_usage_without_credentials_or_prompt_content(self):
        session = Session(reply('{"cuts": []}', usage={"prompt_tokens": 101,
                          "completion_tokens": 25, "total_tokens": 126}))
        client = self.client(session)
        self.assertEqual(client.chat_json("private system", "private user", phase="edit"), {"cuts": []})
        records = client.request_records
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual((record["operation"], record["phase"], record["model"], record["status"]),
                         ("text_model", "edit", "custom-model", "succeeded"))
        attempt = record["attempts"][0]
        self.assertEqual((attempt["http_status"], attempt["finish_reason"], attempt["status"]),
                         (200, "stop", "succeeded"))
        self.assertEqual(attempt["usage"], {"prompt_tokens": 101, "completion_tokens": 25,
                                           "total_tokens": 126})
        self.assertGreaterEqual(record["elapsed_seconds"], 0)
        self.assertGreaterEqual(attempt["elapsed_seconds"], 0)
        statistics = client.statistics
        self.assertEqual((statistics["logical_requests"], statistics["http_attempts"],
                          statistics["retries"], statistics["successful_requests"],
                          statistics["failed_requests"], statistics["usage_reported_attempts"]),
                         (1, 1, 0, 1, 0, 1))
        self.assertEqual(statistics["usage"], attempt["usage"])
        encoded = json.dumps({"records": records, "statistics": statistics})
        for secret in ("test-secret-key", "Authorization", "private system", "private user",
                       "http://localhost"):
            self.assertNotIn(secret, encoded)
        # Save callers receive a snapshot; editing it cannot change later accounting.
        records[0]["attempts"][0]["usage"]["prompt_tokens"] = 9999
        self.assertEqual(client.statistics["usage"]["prompt_tokens"], 101)
        self.assertEqual(session.calls[0][0], "http://localhost:8900/custom-text")
        self.assertEqual(session.calls[0][1]["timeout"], (30, 90))
        self.assertIs(session.calls[0][1]["json"]["stream"], False)
        self.assertNotIn("enable_thinking", session.calls[0][1]["json"])

    def test_truncated_invalid_json_is_diagnosed_before_parse_and_accounts_paid_retry(self):
        session = Session(reply('{"segments":', "length", {"prompt_tokens": 70,
                          "completion_tokens": 8192, "total_tokens": 8262}, "req-truncated"),
                          reply('{"segments": []}', usage={"prompt_tokens": 70,
                                "completion_tokens": 12, "total_tokens": 82}))
        client = self.client(session, "qwen3.7-flash-2026-07-15")
        retries = []
        self.assertEqual(client.chat_json("system", "user", lambda attempt, reason:
                         retries.append((attempt, reason)), phase="caption", max_tokens=8192),
                         {"segments": []})
        self.assertEqual([call[1]["json"]["max_tokens"] for call in session.calls], [8192, 16384])
        self.assertIs(session.calls[0][1]["json"]["enable_thinking"], False)
        self.assertEqual(retries[0][0], 1)
        self.assertIn("输出被截断", retries[0][1])
        self.assertNotIn("JSONDecodeError", retries[0][1])
        attempts = client.request_records[0]["attempts"]
        self.assertEqual(attempts[0]["error_category"], "output_truncated")
        self.assertEqual(attempts[0]["finish_reason"], "length")
        self.assertIs(attempts[1]["is_retry"], True)
        self.assertEqual(client.statistics["usage"], {"prompt_tokens": 140,
                         "completion_tokens": 8204, "total_tokens": 8344})
        self.assertEqual(client.statistics["usage_reported_attempts"], 2)

    def test_transient_http_and_timeout_retry_with_unknown_usage(self):
        session = Session(Response({"error": "busy"}, 503), requests.ReadTimeout("slow"), reply())
        client = self.client(session)
        retries = []
        self.assertEqual(client.chat_json("system", "user", lambda attempt, reason:
                         retries.append((attempt, reason))), {})
        attempts = client.request_records[0]["attempts"]
        self.assertEqual([item["error_category"] for item in attempts], ["http_transient", "timeout", ""])
        self.assertEqual([item["http_status"] for item in attempts], [503, None, 200])
        self.assertEqual([item["max_tokens"] for item in attempts], [2048, 4096, 8192])
        self.assertEqual([attempt for attempt, _ in retries], [1, 2])
        statistics = client.statistics
        self.assertEqual((statistics["http_attempts"], statistics["retries"]), (3, 2))
        self.assertEqual(statistics["usage"], {"prompt_tokens": None, "completion_tokens": None,
                                              "total_tokens": None})
        self.assertEqual(statistics["usage_reported_attempts"], 0)

    def test_three_truncated_responses_fail_with_complete_attempt_history(self):
        session = Session(*(reply('{"cuts":', "length", {"completion_tokens": 2048},
                                  "req-length") for _ in range(3)))
        client = self.client(session)
        with self.assertRaisesRegex(ModelOutputError, "输出被截断.*finish_reason=length"):
            client.chat_json("system", "user", phase="edit")
        record = client.request_records[0]
        self.assertEqual((record["status"], record["error_category"]), ("failed", "output_truncated"))
        self.assertEqual(len(record["attempts"]), 3)
        self.assertEqual(client.statistics["failed_requests"], 1)
        self.assertEqual(client.statistics["usage"]["completion_tokens"], 6144)
        self.assertIsNone(client.statistics["usage"]["prompt_tokens"])

    def test_malformed_json_retry_and_permanent_http_error_are_distinct(self):
        session = Session(reply('{"cuts":', usage={"prompt_tokens": 3, "completion_tokens": 4}),
                          reply(), Response({"error": "unsupported max_tokens"}, 400))
        client = self.client(session)
        self.assertEqual(client.chat_json("system", "user", phase="edit"), {})
        with self.assertRaisesRegex(ApiError, "HTTP 400"):
            client.chat_json("system", "user", phase="caption", max_tokens=8192)
        self.assertEqual(client.request_records[0]["attempts"][0]["error_category"], "invalid_json")
        self.assertEqual(client.request_records[1]["error_category"], "http_error")
        self.assertEqual(len(client.request_records[1]["attempts"]), 1)
        self.assertEqual((client.statistics["logical_requests"], client.statistics["http_attempts"],
                          client.statistics["successful_requests"], client.statistics["failed_requests"]),
                         (2, 3, 1, 1))

    def test_invalid_usage_is_unknown_and_request_id_cannot_echo_key(self):
        session = Session(reply(usage={"prompt_tokens": True, "completion_tokens": -1,
                                      "total_tokens": "42"}, request_id="test-secret-key"))
        client = self.client(session)
        client.chat_json("system", "user", phase="test-secret-key")
        self.assertNotIn("test-secret-key", json.dumps(client.request_records))
        self.assertEqual(client.statistics["usage_reported_attempts"], 0)

    def test_invalid_budget_is_rejected_before_http_request(self):
        session = Session()
        client = self.client(session)
        for value in (0, -1, True, "8192"):
            with self.assertRaisesRegex(ValueError, "max_tokens"):
                client.chat_json("system", "user", max_tokens=value)
        self.assertFalse(session.calls)
        self.assertEqual(client.statistics["logical_requests"], 0)


if __name__ == "__main__":
    unittest.main()
