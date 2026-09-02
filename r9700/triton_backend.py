from __future__ import annotations

import os


def prefer_explicit_rocm_driver() -> None:
    """Keep Triton's probe aligned with an explicit AMD backend selection."""
    if os.environ.get("TRITON_DEFAULT_BACKEND") != "amd":
        return

    import torch

    if torch.version.hip is None:
        return

    from triton.backends import backends

    amd = backends.get("amd")
    nvidia = backends.get("nvidia")
    if amd is None or not amd.driver.is_active() or nvidia is None:
        return
    nvidia.driver.is_active = staticmethod(lambda: False)
