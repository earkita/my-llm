"""Apply the selected Triton backend in vLLM parent and worker interpreters."""

from r9700.triton_backend import prefer_explicit_rocm_driver


prefer_explicit_rocm_driver()
