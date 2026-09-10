import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "watch-claude-throughput.py"
)
SPEC = importlib.util.spec_from_file_location("watch_claude_throughput", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MONITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MONITOR)


def metrics(**overrides: float) -> dict[str, float]:
    values = {key: 0.0 for key in MONITOR.METRICS}
    values.update(overrides)
    return values


class CompletionAccumulatorTests(unittest.TestCase):
    def test_accumulates_updates_from_request_start_until_completion(self) -> None:
        accumulator = MONITOR.CompletionAccumulator(metrics())

        completion, unavailable, active = accumulator.observe(
            metrics(
                running_requests=1,
                prompt_tokens=100,
                prefix_cache_queries=100,
                prefix_cache_hits=80,
                generation_tokens=1,
                prefill_seconds=2,
                ttft_seconds=2,
            )
        )
        self.assertIsNone(completion)
        self.assertEqual(unavailable, 0)
        self.assertTrue(active)

        completion, unavailable, active = accumulator.observe(
            metrics(
                prompt_tokens=100,
                prefix_cache_queries=100,
                prefix_cache_hits=80,
                generation_tokens=21,
                completed_requests=1,
                prefill_seconds=2,
                decode_seconds=1,
                ttft_seconds=2,
                e2e_seconds=3,
            )
        )

        self.assertEqual(unavailable, 0)
        self.assertFalse(active)
        self.assertIsNotNone(completion)
        assert completion is not None
        self.assertEqual(completion["prompt_tokens"], 100)
        self.assertEqual(completion["output_tokens"], 21)
        self.assertEqual(completion["cached_tokens"], 80)
        self.assertEqual(completion["cache_query_tokens"], 100)
        self.assertEqual(completion["cache_hit_percent"], 80)
        self.assertEqual(completion["prefill_tokens_per_second"], 50)
        self.assertEqual(completion["decode_tokens_per_second"], 20)
        self.assertEqual(completion["mean_ttft_seconds"], 2)
        self.assertEqual(completion["mean_e2e_seconds"], 3)

    def test_marks_completion_partial_when_attached_mid_request(self) -> None:
        accumulator = MONITOR.CompletionAccumulator(
            metrics(running_requests=1, prompt_tokens=50, prefill_seconds=1)
        )

        completion, unavailable, active = accumulator.observe(
            metrics(
                prompt_tokens=100,
                generation_tokens=21,
                completed_requests=1,
                prefill_seconds=2,
                decode_seconds=1,
                ttft_seconds=2,
                e2e_seconds=3,
            )
        )

        self.assertIsNone(completion)
        self.assertEqual(unavailable, 1)
        self.assertFalse(active)

    def test_reports_live_request_cache_delta(self) -> None:
        accumulator = MONITOR.CompletionAccumulator(
            metrics(prefix_cache_queries=1000, prefix_cache_hits=250)
        )

        accumulator.observe(
            metrics(
                running_requests=1,
                prefix_cache_queries=1200,
                prefix_cache_hits=330,
            )
        )

        self.assertEqual(
            accumulator.request_cache(
                metrics(
                    running_requests=1,
                    prefix_cache_queries=1200,
                    prefix_cache_hits=330,
                )
            ),
            {
                "cached_tokens": 80,
                "cache_query_tokens": 200,
                "cache_hit_percent": 40,
            },
        )


if __name__ == "__main__":
    unittest.main()
