from __future__ import annotations

import json
import unittest

from r9700.claude_qualification import _parse_stream


class ClaudeQualificationTests(unittest.TestCase):
    def test_parse_stream_returns_result_and_unique_tool_names(self) -> None:
        events = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "id": "one", "name": "Read"}
                    ]
                },
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "id": "one", "name": "Read"},
                        {"type": "tool_use", "id": "two", "name": "Edit"},
                    ]
                },
            },
            {"type": "result", "subtype": "success", "result": "ok"},
        ]
        result, tools, thinking = _parse_stream(
            "\n".join(json.dumps(event) for event in events)
        )

        self.assertEqual(result["result"], "ok")
        self.assertEqual(tools, ["Read", "Edit"])
        self.assertEqual(thinking, {"blocks": 0, "characters": 0})

    def test_parse_stream_reports_unique_thinking_blocks(self) -> None:
        thinking = {"type": "thinking", "thinking": "17x = 170"}
        events = [
            {"type": "assistant", "message": {"content": [thinking]}},
            {"type": "assistant", "message": {"content": [thinking]}},
            {"type": "result", "subtype": "success", "result": "X=10"},
        ]

        result, tools, stats = _parse_stream(
            "\n".join(json.dumps(event) for event in events)
        )

        self.assertEqual(result["result"], "X=10")
        self.assertEqual(tools, [])
        self.assertEqual(stats, {"blocks": 1, "characters": 9})

    def test_parse_stream_requires_final_result(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no final result"):
            _parse_stream(json.dumps({"type": "system"}))


if __name__ == "__main__":
    unittest.main()
