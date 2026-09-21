import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles" / "dev" / "glm53-flash-new-dflash.json"
CONFIG_DIR = Path("tuning/moe/glm53-flash-new")
CONFIG_NAME = (
    "E=288,N=256,device_name=AMD_Radeon_R9700,dtype=int4_w4a16.json"
)


class Glm53MoeTuningTests(unittest.TestCase):
    def test_diagnostic_mode_selects_repo_tuning_directory(self) -> None:
        profile = json.loads(PROFILE.read_text())
        mode = profile["runtime"]["experimental_modes"][
            "dflash-full-piecewise-256k-moe-tuned"
        ]
        environment = mode["runtime_overrides"]["environment"]

        self.assertEqual(
            environment["VLLM_TUNED_CONFIG_FOLDER"], CONFIG_DIR.as_posix()
        )
        self.assertEqual(
            mode["runtime_overrides"]["limits"],
            profile["runtime"]["experimental_modes"][
                "dflash-full-piecewise-256k"
            ]["runtime_overrides"]["limits"],
        )

    def test_tuning_file_covers_decode_and_chunked_prefill_buckets(self) -> None:
        config = json.loads((ROOT / CONFIG_DIR / CONFIG_NAME).read_text())

        self.assertEqual(config["triton_version"], "3.8.0")
        self.assertEqual(
            sorted(int(key) for key in config if key != "triton_version"),
            [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536, 2048, 3072, 4096],
        )
        for key, tile in config.items():
            if key == "triton_version":
                continue
            self.assertEqual(tile["SPLIT_K"], 1)
            self.assertEqual(tile["GROUP_SIZE_M"], 1)
            self.assertIn(tile["BLOCK_SIZE_K"], (32, 64))


if __name__ == "__main__":
    unittest.main()
