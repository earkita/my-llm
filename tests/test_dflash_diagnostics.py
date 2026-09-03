from __future__ import annotations

import copy
import unittest

from scripts.dflash_diagnostics import (
    DiagnosticError,
    compare_payloads,
    metric_delta,
    parse_prometheus,
    render_markdown,
)


def _capture(label: str, token_ids: list[int]) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "dflash_capture",
        "label": label,
        "request": {
            "prompt": [11, 12, 13],
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
            "max_tokens": 4,
            "ignore_eos": True,
            "n": 1,
        },
        "response": {
            "choices": [
                {
                    "logprobs": {
                        "token_logprobs": [-0.1, -0.2, -0.3, -0.4]
                    }
                }
            ]
        },
        "result": {
            "output_token_ids": token_ids,
            "output_token_ids_sha256": label * 4,
            "speculative_decoding": None,
        },
        "prometheus": {"delta": {"available": False}},
    }


class DFlashDiagnosticTests(unittest.TestCase):
    def test_parse_prometheus_sums_workers_and_positions(self) -> None:
        payload = parse_prometheus(
            """
# HELP ignored ignored
vllm:spec_decode_num_drafts_total{engine="0"} 4
vllm:spec_decode_num_drafts_total{engine="1"} 3
vllm:spec_decode_num_draft_tokens_total{engine="0"} 28
vllm:spec_decode_num_accepted_tokens_total{engine="0"} 7
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="0"} 4
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="1",position="0"} 2
vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="1"} 1
vllm:spec_decode_num_accepted_tokens_per_pos_total{position="2",engine="0"} 1
"""
        )
        self.assertTrue(payload["available"])
        self.assertEqual(payload["num_drafts"], 7)
        self.assertEqual(payload["num_draft_tokens"], 28)
        self.assertEqual(payload["num_accepted_tokens"], 7)
        self.assertEqual(
            payload["accepted_tokens_per_position"], {"0": 6, "1": 1, "2": 1}
        )

    def test_metric_delta_calculates_acceptance(self) -> None:
        before = {
            "available": True,
            "num_drafts": 10,
            "num_draft_tokens": 70,
            "num_accepted_tokens": 14,
            "accepted_tokens_per_position": {"0": 8, "1": 4},
        }
        after = {
            "available": True,
            "num_drafts": 14,
            "num_draft_tokens": 98,
            "num_accepted_tokens": 21,
            "accepted_tokens_per_position": {"0": 12, "1": 6},
        }
        delta = metric_delta(before, after)
        self.assertEqual(delta["num_drafts"], 4)
        self.assertEqual(delta["num_draft_tokens"], 28)
        self.assertEqual(delta["num_accepted_tokens"], 7)
        self.assertEqual(delta["draft_acceptance_rate"], 0.25)
        self.assertEqual(delta["mean_acceptance_length"], 2.75)

    def test_compare_reports_first_exact_token_divergence(self) -> None:
        baseline = _capture("target", [20, 21, 22, 23])
        candidate = _capture("dflash-k1", [20, 21, 99, 23])
        candidate["result"]["speculative_decoding"] = {
            "draft_acceptance_rate": 0.025,
            "mean_acceptance_length": 1.025,
        }

        result = compare_payloads([baseline, candidate])

        self.assertFalse(result["all_greedy_equivalent"])
        row = result["comparisons"][0]
        self.assertEqual(row["first_divergent_output_token"], 2)
        self.assertEqual(row["baseline_token_id_at_divergence"], 22)
        self.assertEqual(row["candidate_token_id_at_divergence"], 99)
        markdown = render_markdown(result)
        self.assertIn("`dflash-k1`", markdown)
        self.assertIn("2.50%", markdown)

    def test_compare_rejects_different_generation_settings(self) -> None:
        baseline = _capture("target", [20, 21])
        candidate = copy.deepcopy(_capture("dflash-k7", [20, 21]))
        candidate["request"]["seed"] = 9
        with self.assertRaisesRegex(DiagnosticError, "differs in: seed"):
            compare_payloads([baseline, candidate])


if __name__ == "__main__":
    unittest.main()
