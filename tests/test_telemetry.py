from __future__ import annotations

import unittest

from r9700.telemetry import label_samples


class LabelSamplesTests(unittest.TestCase):
    def test_labels_benchmark_prefill_and_decode(self) -> None:
        samples = [
            {"perf_counter": 10.5, "phase": "outside_measured_requests"},
            {"perf_counter": 12.5, "phase": "outside_measured_requests"},
        ]
        rows = [{"started_perf": 10.0, "ended_perf": 13.0, "ttft_seconds": 2.0}]

        label_samples(samples, rows)

        self.assertEqual([sample["phase"] for sample in samples], ["prefill", "decode"])

    def test_labels_non_streaming_request_without_ttft(self) -> None:
        samples = [
            {"perf_counter": 9.0, "phase": "outside_measured_requests"},
            {"perf_counter": 10.5, "phase": "outside_measured_requests"},
            {"perf_counter": 13.5, "phase": "outside_measured_requests"},
        ]
        rows = [{"started_perf": 10.0, "ended_perf": 13.0}]

        label_samples(samples, rows)

        self.assertEqual(
            [sample["phase"] for sample in samples],
            ["outside_measured_requests", "request", "outside_measured_requests"],
        )


if __name__ == "__main__":
    unittest.main()
