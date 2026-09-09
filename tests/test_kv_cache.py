from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from r9700.config import ConfigurationError
from r9700.kv_cache import (
    BOUNDED_FS_MODULE,
    BOUNDED_FS_TIER,
    cache_usage,
    clear,
    prepare,
    reclaim_for_write,
)


class KVCacheTests(unittest.TestCase):
    def test_reclaim_evicts_oldest_unprotected_block(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            oldest = root / "oldest.bin"
            protected = root / "protected.bin"
            newest = root / "newest.bin"
            for index, path in enumerate((oldest, protected, newest), start=1):
                path.write_bytes(b"x" * 4)
                os.utime(path, ns=(index, index))

            current, reclaimed = reclaim_for_write(
                root,
                current_bytes=12,
                incoming_bytes=8,
                max_bytes=12,
                min_free_bytes=0,
                protected_paths={str(oldest)},
            )

            self.assertEqual((current, reclaimed), (4, 8))
            self.assertTrue(oldest.exists())
            self.assertFalse(protected.exists())
            self.assertFalse(newest.exists())

    def test_reclaim_rejects_a_write_larger_than_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(OSError):
                reclaim_for_write(
                    Path(temporary),
                    current_bytes=0,
                    incoming_bytes=9,
                    max_bytes=8,
                    min_free_bytes=0,
                )

    def test_prepare_and_clear_require_owned_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            tier = {
                "root_dir": str(root),
                "max_bytes": 1024,
                "min_free_bytes": 0,
            }
            selected = ({"name": "test-profile"}, tier, root)
            with patch("r9700.kv_cache._selected_cache", return_value=selected):
                prepared = prepare("test-profile", "offload")
                self.assertTrue(prepared["prepared"])
                payload = root / "namespace" / "block.bin"
                payload.parent.mkdir()
                payload.write_bytes(b"payload")

                with patch(
                    "r9700.service.managed_state",
                    side_effect=ConfigurationError("not running"),
                ):
                    result = clear("test-profile", "offload")

                self.assertEqual(result["removed_files"], 1)
                self.assertFalse(payload.exists())
                self.assertTrue((root / ".r9700-kv-cache.json").exists())
                self.assertEqual(cache_usage(root)[1], 1)

    def test_prepare_refuses_a_non_empty_unowned_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            root.mkdir()
            (root / "unrelated.txt").write_text("keep")
            tier = {
                "root_dir": str(root),
                "max_bytes": 1024,
                "min_free_bytes": 0,
            }
            selected = ({"name": "test-profile"}, tier, root)
            with (
                patch("r9700.kv_cache._selected_cache", return_value=selected),
                self.assertRaisesRegex(ConfigurationError, "non-empty unowned"),
            ):
                prepare("test-profile", "offload")
            self.assertEqual((root / "unrelated.txt").read_text(), "keep")

    def test_clear_refuses_an_active_runtime_using_the_same_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            tier = {
                "type": BOUNDED_FS_TIER,
                "module_path": BOUNDED_FS_MODULE,
                "root_dir": str(root),
                "max_bytes": 1024,
                "min_free_bytes": 0,
            }
            selected = ({"name": "test-profile"}, tier, root)
            active_runtime = {
                "kv_transfer_config": {
                    "kv_connector_extra_config": {"secondary_tiers": [tier]}
                }
            }
            with patch("r9700.kv_cache._selected_cache", return_value=selected):
                prepare("test-profile", "offload")
                with (
                    patch(
                        "r9700.service.managed_state",
                        return_value={
                            "profile": "test-profile",
                            "runtime_mode": "offload",
                        },
                    ),
                    patch("r9700.kv_cache.load_runtime", return_value=active_runtime),
                    self.assertRaisesRegex(ConfigurationError, "runtime mode is active"),
                ):
                    clear("test-profile", "offload")
