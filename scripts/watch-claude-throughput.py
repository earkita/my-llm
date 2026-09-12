#!/usr/bin/env python3
"""Passively show live vLLM throughput for local Qwen traffic."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from r9700.live_throughput import (  # noqa: E402,F401
    CompletionAccumulator,
    Endpoint,
    LiveDecodeRate,
    METRICS,
    OPTIONAL_METRICS,
    _cache_capacities,
    _cache_capacity,
    _completion,
    _dashboard,
    _delta,
    _metrics,
    _metrics_by_engine,
    _parse_metrics,
    _prefix_cache_delta,
    main,
)


if __name__ == "__main__":
    raise SystemExit(main())
