from __future__ import annotations

import sys

from .triton_backend import prefer_explicit_rocm_driver


def main() -> None:
    prefer_explicit_rocm_driver()
    from vllm.entrypoints.cli.main import main as vllm_main

    sys.argv[0] = "vllm"
    vllm_main()


if __name__ == "__main__":
    main()
