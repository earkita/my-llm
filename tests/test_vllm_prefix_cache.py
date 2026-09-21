from __future__ import annotations

from pathlib import Path
import unittest

from r9700.backends.vllm import command as build_command
from r9700.config import ConfigurationError, load_profile, load_runtime, validate_runtime


PROFILE = "profiles/dev/glm53-flash-new-dflash.json"
MODE = "dflash-full-piecewise-256k-cache-xxhash-3p2g"


class VllmPrefixCacheTests(unittest.TestCase):
    def test_xxhash_cache_mode_is_production_matched(self) -> None:
        profile = load_profile(PROFILE)
        runtime = load_runtime(PROFILE, MODE)

        self.assertEqual(runtime["limits"]["kv_cache_memory_bytes"], 3_200_000_000)
        self.assertEqual(runtime["cache"]["prefix_caching_hash_algo"], "xxhash")
        self.assertTrue(runtime["cache"]["prefix_cache"])
        self.assertEqual(runtime["cache"]["prefix_cache_retention_interval"], 1280)

        command = build_command(
            profile["model"], runtime, Path("/models/glm"), "127.0.0.1", 8000
        )
        self.assertEqual(
            command[command.index("--kv-cache-memory-bytes") + 1], "3200000000"
        )
        self.assertEqual(
            command[command.index("--prefix-caching-hash-algo") + 1], "xxhash"
        )

    def test_hash_algorithm_requires_prefix_cache(self) -> None:
        profile = load_profile(PROFILE)
        runtime = dict(profile["runtime"])
        runtime.pop("experimental_modes", None)
        runtime["cache"] = dict(runtime["cache"])
        runtime["cache"]["prefix_cache"] = False
        runtime["cache"]["prefix_caching_hash_algo"] = "xxhash"

        with self.assertRaisesRegex(ConfigurationError, "requires prefix_cache"):
            validate_runtime(runtime)


if __name__ == "__main__":
    unittest.main()
