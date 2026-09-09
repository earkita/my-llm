from __future__ import annotations

import os
import threading
from collections.abc import Collection
from pathlib import Path

from typing_extensions import override
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.tiering.base import JobId, JobResult, TransferJob
from vllm.v1.kv_offload.tiering.fs.io import batch_store_block
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager

from r9700.kv_cache import block_cache_usage, reclaim_for_write


class BoundedFileSystemTierManager(FileSystemTierManager):
    """vLLM filesystem tier with a process-local hard capacity guard.

    A dedicated cache root must have one active writer. Stores are serialized
    with eviction so complete ``*.bin`` blocks never grow past ``max_bytes``
    and the underlying filesystem retains ``min_free_bytes``. Blocks involved
    in queued or active transfers are protected from eviction.
    """

    def __init__(
        self,
        *args,
        root_dir: str,
        max_bytes: int,
        min_free_bytes: int = 0,
        **kwargs,
    ) -> None:
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes <= 0
        ):
            raise ValueError("max_bytes must be a positive integer")
        if (
            isinstance(min_free_bytes, bool)
            or not isinstance(min_free_bytes, int)
            or min_free_bytes < 0
        ):
            raise ValueError("min_free_bytes must be a non-negative integer")
        super().__init__(*args, root_dir=root_dir, **kwargs)
        self._capacity_root = Path(root_dir).expanduser().resolve()
        self._max_bytes = max_bytes
        self._min_free_bytes = min_free_bytes
        self._capacity_lock = threading.Lock()
        self._protected_by_job: dict[JobId, set[str]] = {}
        self._cache_bytes, _ = block_cache_usage(self._capacity_root)

    def _protect(self, job_id: JobId, paths: set[str]) -> None:
        with self._capacity_lock:
            self._protected_by_job[job_id] = paths

    def _release(self, job_id: JobId) -> None:
        with self._capacity_lock:
            self._protected_by_job.pop(job_id, None)

    def _protected_paths(self) -> set[str]:
        return {
            path
            for job_paths in self._protected_by_job.values()
            for path in job_paths
        }

    @override
    def submit_store(self, job_metadata: TransferJob) -> None:
        keys = list(job_metadata.keys)
        paths = [self.file_mapper.get_file_name(key) for key in keys]
        absolute_paths = {os.path.abspath(path) for path in paths}
        self._protect(job_metadata.job_id, absolute_paths)
        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = keys

        offsets = [
            int(block_id) * self._block_size for block_id in job_metadata.block_ids
        ]

        def bounded_store() -> None:
            with self._capacity_lock:
                missing = [path for path in paths if not os.path.exists(path)]
                incoming_bytes = len(missing) * self._block_size
                self._cache_bytes, _ = reclaim_for_write(
                    self._capacity_root,
                    current_bytes=self._cache_bytes,
                    incoming_bytes=incoming_bytes,
                    max_bytes=self._max_bytes,
                    min_free_bytes=self._min_free_bytes,
                    protected_paths=self._protected_paths(),
                )
                try:
                    batch_store_block(
                        paths,
                        self._primary_kv_view,
                        offsets,
                        self._block_size,
                        self._use_o_direct,
                    )
                finally:
                    self._cache_bytes += sum(
                        os.path.getsize(path)
                        for path in missing
                        if os.path.exists(path)
                    )

        self._pool.enqueue_store(job_metadata.job_id, 1, [bounded_store])

    @override
    def submit_load(self, job_metadata: TransferJob) -> None:
        paths = {
            os.path.abspath(self.file_mapper.get_file_name(key))
            for key in job_metadata.keys
        }
        self._protect(job_metadata.job_id, paths)
        try:
            super().submit_load(job_metadata)
        except Exception:
            self._release(job_metadata.job_id)
            raise

    @override
    def get_finished_jobs(self):
        results: list[JobResult] = list(super().get_finished_jobs())
        for result in results:
            self._release(result.job_id)
        return results

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext):
        # Disk eviction is oldest-write-first. Updating mtimes for every cache
        # hit would put synchronous metadata I/O on the scheduler hot path.
        return
