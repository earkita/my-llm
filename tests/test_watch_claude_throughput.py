import importlib.util
from collections import deque
from pathlib import Path
import tempfile
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
                spec_draft_tokens=28,
                spec_accepted_tokens=21,
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
        self.assertEqual(completion["uncached_prefill_tokens_per_second"], 10)
        self.assertEqual(completion["decode_tokens_per_second"], 20)
        self.assertEqual(completion["mean_ttft_seconds"], 2)
        self.assertEqual(completion["mean_e2e_seconds"], 3)
        self.assertEqual(completion["spec_draft_tokens"], 28)
        self.assertEqual(completion["spec_accepted_tokens"], 21)
        self.assertEqual(completion["spec_acceptance_percent"], 75)

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


class LiveDecodeRateTests(unittest.TestCase):
    def test_reports_rolling_decode_and_resets_between_requests(self) -> None:
        tracker = MONITOR.LiveDecodeRate(
            100,
            0.0,
            active=False,
            window=10.0,
        )

        self.assertEqual(tracker.observe(1.0, 100, active=True), 0)
        self.assertEqual(tracker.observe(2.0, 120, active=True), 20)
        self.assertEqual(tracker.observe(3.0, 145, active=True), 22.5)
        self.assertEqual(tracker.observe(4.0, 150, active=False), 0)
        self.assertEqual(tracker.observe(5.0, 150, active=True), 0)
        self.assertEqual(tracker.observe(6.0, 160, active=True), 10)

    def test_limits_live_decode_to_the_requested_window(self) -> None:
        tracker = MONITOR.LiveDecodeRate(
            0,
            0.0,
            active=True,
            window=2.0,
        )

        self.assertEqual(tracker.observe(1.0, 10, active=True), 10)
        self.assertEqual(tracker.observe(2.0, 30, active=True), 15)
        self.assertEqual(tracker.observe(3.0, 60, active=True), 25)

    def test_waiting_request_does_not_carry_over_previous_decode(self) -> None:
        tracker = MONITOR.LiveDecodeRate(
            100,
            0.0,
            active=True,
            window=10.0,
        )

        self.assertEqual(tracker.observe(1.0, 120, active=True), 20)
        self.assertEqual(tracker.observe(2.0, 125, active=False), 0)
        self.assertEqual(tracker.observe(3.0, 125, active=True), 0)
        self.assertEqual(tracker.observe(4.0, 135, active=True), 10)

    def test_completion_resets_decode_when_next_request_is_already_running(self) -> None:
        tracker = MONITOR.LiveDecodeRate(
            100,
            0.0,
            active=True,
            window=10.0,
        )

        self.assertEqual(tracker.observe(1.0, 120, active=True), 20)
        self.assertEqual(
            tracker.observe(
                2.0,
                125,
                active=True,
                request_completed=True,
            ),
            0,
        )
        self.assertEqual(tracker.observe(3.0, 125, active=True), 0)
        self.assertEqual(tracker.observe(4.0, 135, active=True), 10)


class MultiEngineMetricTests(unittest.TestCase):
    def test_parser_keeps_worker_engine_counters_separate(self) -> None:
        lines = []
        all_metrics = {**MONITOR.METRICS, **MONITOR.OPTIONAL_METRICS}
        for engine, multiplier in (("0", 1), ("1", 2)):
            for index, name in enumerate(all_metrics.values(), start=1):
                lines.append(
                    f'{name}{{engine="{engine}",model_name="qwen"}} '
                    f"{index * multiplier}"
                )

        parsed = MONITOR._parse_metrics("\n".join(lines))

        self.assertEqual(set(parsed), {"0", "1"})
        self.assertEqual(parsed["0"]["prompt_tokens"], 4)
        self.assertEqual(parsed["1"]["prompt_tokens"], 8)
        self.assertEqual(parsed["0"]["spec_draft_tokens"], 13)
        self.assertEqual(parsed["1"]["spec_accepted_tokens"], 28)

    def test_cache_capacity_parser_maps_each_data_parallel_engine(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.log"
            path.write_text(
                "(EngineCore_DP1 pid=2) GPU KV cache size: 250,106 tokens\n"
                "(EngineCore_DP0 pid=1) GPU KV cache size: 249,999 tokens\n"
            )

            self.assertEqual(
                MONITOR._cache_capacities(path),
                {"0": 249999, "1": 250106},
            )


class DashboardTests(unittest.TestCase):
    def test_dashboard_renders_one_model_row_without_ansi_when_disabled(self) -> None:
        endpoint = MONITOR.Endpoint(
            label="MAIN",
            url="http://127.0.0.1:8000",
            profile="qwen",
            runtime="qwen-runtime",
            log_path=Path("runtime.log"),
            capacities={"0": 616228},
            gpu_ids=(6, 4, 5, 0),
        )
        payload = {
            "timestamp": "2026-09-12T18:00:00+02:00",
            "source": "MAIN",
            "engine": "0",
            "phase": "decode",
            "running_requests": 1,
            "waiting_requests": 0,
            "kv_cache_percent": 12.5,
            "kv_growth_tokens_per_second": 900.0,
            "live_decode_tokens_per_second": 31.5,
            "request_cache": {
                "cached_tokens": 800,
                "cache_query_tokens": 1000,
            },
            "request_speculative": {
                "spec_accepted_tokens": 7,
                "spec_draft_tokens": 10,
            },
            "gpu": {
                "gpu_ids": [6, 4, 5, 0],
                "gfx_percent_mean": 99.0,
                "gfx_percent_min": 98.0,
                "gfx_percent_max": 100.0,
                "umc_percent_mean": 40.0,
                "clock_mhz_mean": 2300.0,
                "power_watts_total": 700.0,
            },
        }

        view = MONITOR._dashboard(
            [payload],
            [endpoint],
            deque(),
            interval=1.0,
            width=160,
            color=False,
        )

        self.assertIn("QWEN MULTI — LIVE", view)
        self.assertIn("MAIN", view)
        self.assertIn("DECODE", view)
        self.assertIn("31.5", view)
        self.assertIn("800/1000 80%", view)
        self.assertIn("7/10 70%", view)
        self.assertNotIn("\033[", view)

if __name__ == "__main__":
    unittest.main()
