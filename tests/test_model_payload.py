from __future__ import annotations

import json
import unittest

from studio.model_payload import compact_model_input


class PayloadTests(unittest.TestCase):
    def test_long_edit_input_preserves_every_word_index_time_and_full_context(self):
        words = [{"i": i + 600, "text": "原声", "start_ms": i * 400, "end_ms": i * 400 + 300,
                  "punctuation": "", "pause_after_ms": 100, "next_word_index": i + 601}
                 for i in range(792)]
        value = json.dumps({"phase": "edit", "full_transcript": "完整上下文", "words": words}, ensure_ascii=False)
        result = compact_model_input(value)
        parsed = json.loads(result)
        self.assertEqual(parsed["full_transcript"], "完整上下文")
        self.assertEqual(parsed["word_table"]["rows"],
                         [[w["i"], w["text"], w["start_ms"], w["end_ms"]] for w in words])
        self.assertLess(len(result), len(value) * .4)

    def test_caption_table_preserves_non_contiguous_source_indices_and_repair_message(self):
        tokens = [{"token": i, "word_index": i * 2, "text": "字", "punctuation": "",
                   "start_ms": i * 500, "end_ms": i * 500 + 400} for i in range(220)]
        suffix = '\n上次字幕无效：请按自然意群拆分\n上次结果：{"segments":[]}'
        result = compact_model_input(json.dumps({"kept_tokens": tokens}) + suffix)
        parsed, end = json.JSONDecoder().raw_decode(result)
        self.assertEqual(parsed["token_table"]["rows"],
                         [[t["token"], t["word_index"], t["text"], 400] for t in tokens])
        self.assertEqual(result[end:], suffix)
