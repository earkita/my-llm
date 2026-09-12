from __future__ import annotations

import json
import unittest

from r9700.claude_team_qualification import (
    _agent_calls,
    _resolved_workers,
    _stream_events,
)


class ClaudeTeamQualificationTests(unittest.TestCase):
    def test_stream_parser_extracts_background_agent_call(self) -> None:
        events = _stream_events(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {
                                "type": "tool_use",
                                "name": "Agent",
                                "input": {
                                    "subagent_type": "qwen-worker-explorer",
                                    "name": "explorer",
                                    "run_in_background": True,
                                },
                            }
                        ]
                    },
                }
            )
        )

        self.assertEqual(
            _agent_calls(events),
            [
                {
                    "subagent_type": "qwen-worker-explorer",
                    "name": "explorer",
                    "run_in_background": True,
                }
            ],
        )

    def test_resolved_worker_parser_records_alias_and_marker(self) -> None:
        events = [
            {
                "type": "user",
                "tool_use_result": {
                    "agentId": "worker-1",
                    "agentType": "qwen-worker-verifier",
                    "resolvedModel": "qwen3.8-27b-workers-thinking",
                    "content": [{"type": "text", "text": "VERIFIER_OK"}],
                    "totalDurationMs": 3210,
                    "totalTokens": 456,
                    "totalToolUseCount": 0,
                },
            }
        ]

        self.assertEqual(
            _resolved_workers(events),
            {
                "qwen-worker-verifier": {
                    "agent_id": "worker-1",
                    "resolved_model": "qwen3.8-27b-workers-thinking",
                    "text": "VERIFIER_OK",
                    "duration_ms": 3210,
                    "tokens": 456,
                    "tool_uses": 0,
                }
            },
        )

    def test_stream_parser_rejects_non_json(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "invalid stream JSON"):
            _stream_events("not-json")


if __name__ == "__main__":
    unittest.main()
