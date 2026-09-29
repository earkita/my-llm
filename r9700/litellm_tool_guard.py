from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


READ_ONLY_TOOLS = frozenset({"Read", "Glob", "Grep", "WebSearch", "WebFetch"})
AGENT_TOOLS = frozenset({"Agent"})
READ_ONLY_LIMIT = 4
AGENT_LIMIT = 4
MAX_BUFFER_BYTES = 64 * 1024
MAX_BATCH_SECONDS = 10.0
LOGGER = logging.getLogger("r9700.litellm_tool_guard")


class ToolClass(str, Enum):
    READ_ONLY = "read_only"
    AGENT = "agent"
    MUTABLE = "mutable"


@dataclass(frozen=True)
class GuardResult:
    stop: bool
    reason: str | None = None


@dataclass
class _ToolBuilder:
    index: int
    tool_id: str = ""
    name: str = ""
    argument_parts: list[str] = field(default_factory=list)
    complete: bool = False

    @property
    def raw_arguments(self) -> str:
        return "".join(self.argument_parts)


@dataclass(frozen=True)
class GuardedToolCall:
    index: int
    tool_id: str
    emitted_id: str
    name: str
    arguments: dict[str, Any]
    signature: str
    tool_class: ToolClass


def classify_tool(name: str) -> ToolClass:
    if name in READ_ONLY_TOOLS:
        return ToolClass.READ_ONLY
    if name in AGENT_TOOLS:
        return ToolClass.AGENT
    return ToolClass.MUTABLE


def canonical_signature(name: str, arguments: dict[str, Any]) -> str:
    canonical = json.dumps(
        arguments,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"{name}\0{canonical}"


class AdaptiveToolBatch:
    def __init__(
        self,
        *,
        read_only_limit: int = READ_ONLY_LIMIT,
        agent_limit: int = AGENT_LIMIT,
        max_buffer_bytes: int = MAX_BUFFER_BYTES,
        max_batch_seconds: float = MAX_BATCH_SECONDS,
        started_at: float | None = None,
    ) -> None:
        self.read_only_limit = read_only_limit
        self.agent_limit = agent_limit
        self.max_buffer_bytes = max_buffer_bytes
        self.max_batch_seconds = max_batch_seconds
        self.started_at = time.monotonic() if started_at is None else started_at
        self.buffer_bytes = 0
        self.builders: dict[int, _ToolBuilder] = {}
        self.retained: list[GuardedToolCall] = []
        self.seen_signatures: set[str] = set()
        self.seen_ids: set[str] = set()
        self.batch_class: ToolClass | None = None
        self.stop_reason: str | None = None

    @property
    def retained_indexes(self) -> frozenset[int]:
        return frozenset(call.index for call in self.retained)

    @property
    def emitted_ids(self) -> dict[int, str]:
        return {call.index: call.emitted_id for call in self.retained}

    def seconds_remaining(self, now: float | None = None) -> float:
        current = time.monotonic() if now is None else now
        return self.max_batch_seconds - (current - self.started_at)

    def _stop(self, reason: str) -> GuardResult:
        self.stop_reason = reason
        return GuardResult(stop=True, reason=reason)

    def _check_limits(self, added_bytes: int, now: float | None) -> GuardResult | None:
        self.buffer_bytes += added_bytes
        if self.buffer_bytes > self.max_buffer_bytes:
            return self._stop("buffer_limit")
        if self.seconds_remaining(now) <= 0:
            return self._stop("time_limit")
        return None

    def feed(
        self,
        *,
        index: int,
        tool_id: str | None,
        name: str | None,
        arguments_delta: str | None,
        now: float | None = None,
    ) -> GuardResult:
        added_bytes = sum(
            len(value.encode("utf-8"))
            for value in (tool_id, name, arguments_delta)
            if value
        )
        limited = self._check_limits(added_bytes, now)
        if limited is not None:
            return limited

        builder = self.builders.setdefault(index, _ToolBuilder(index=index))
        if builder.complete:
            if any((tool_id, name, arguments_delta)):
                return self._stop("data_after_complete_call")
            return GuardResult(stop=False)
        if tool_id:
            if builder.tool_id and builder.tool_id != tool_id:
                return self._stop("tool_id_changed")
            builder.tool_id = tool_id
        if name:
            if builder.name and builder.name != name:
                return self._stop("tool_name_changed")
            builder.name = name
        if arguments_delta:
            builder.argument_parts.append(arguments_delta)

        if not builder.name or not builder.raw_arguments:
            return GuardResult(stop=False)
        try:
            arguments = json.loads(builder.raw_arguments)
        except json.JSONDecodeError:
            return GuardResult(stop=False)
        if not isinstance(arguments, dict):
            return self._stop("arguments_not_object")
        return self._finalize(builder, arguments)

    def _finalize(
        self, builder: _ToolBuilder, arguments: dict[str, Any]
    ) -> GuardResult:
        builder.complete = True
        tool_class = classify_tool(builder.name)
        signature = canonical_signature(builder.name, arguments)
        if signature in self.seen_signatures:
            return self._stop("duplicate_signature")
        if self.batch_class is not None and tool_class != self.batch_class:
            return self._stop("mixed_tool_classes")

        emitted_id = builder.tool_id
        if not emitted_id or emitted_id in self.seen_ids:
            emitted_id = f"guard-{uuid.uuid4().hex}"
        call = GuardedToolCall(
            index=builder.index,
            tool_id=builder.tool_id,
            emitted_id=emitted_id,
            name=builder.name,
            arguments=arguments,
            signature=signature,
            tool_class=tool_class,
        )
        self.retained.append(call)
        self.seen_signatures.add(signature)
        self.seen_ids.add(emitted_id)
        self.batch_class = tool_class

        if tool_class == ToolClass.MUTABLE:
            return self._stop("mutable_limit")
        limit = (
            self.read_only_limit
            if tool_class == ToolClass.READ_ONLY
            else self.agent_limit
        )
        if len(self.retained) >= limit:
            return self._stop(f"{tool_class.value}_limit")
        return GuardResult(stop=False)

    def finish(self) -> GuardResult:
        for builder in self.builders.values():
            if not builder.complete:
                return self._stop("incomplete_arguments")
        return GuardResult(stop=False)


def _choice_delta(chunk: Any) -> Any | None:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return None
    return getattr(choices[0], "delta", None)


def _tool_deltas(chunk: Any) -> list[Any]:
    delta = _choice_delta(chunk)
    if delta is None:
        return []
    return list(getattr(delta, "tool_calls", None) or [])


def _finish_reason(chunk: Any) -> str | None:
    choices = getattr(chunk, "choices", None)
    if not choices:
        return None
    return getattr(choices[0], "finish_reason", None)


def _function_value(tool_delta: Any, key: str) -> str | None:
    function = getattr(tool_delta, "function", None)
    if function is None:
        return None
    value = getattr(function, key, None)
    return value if isinstance(value, str) else None


def _copy_chunk(chunk: Any) -> Any:
    copier = getattr(chunk, "model_copy", None)
    if callable(copier):
        return copier(deep=True)
    import copy

    return copy.deepcopy(chunk)


def _filtered_tool_chunk(
    chunk: Any,
    retained_indexes: frozenset[int],
    emitted_ids: dict[int, str],
) -> Any | None:
    tool_deltas = _tool_deltas(chunk)
    if not tool_deltas:
        return chunk
    filtered = []
    for tool_delta in tool_deltas:
        index = int(getattr(tool_delta, "index", 0))
        if index not in retained_indexes:
            continue
        copied_delta = _copy_chunk(tool_delta)
        current_id = getattr(copied_delta, "id", None)
        emitted_id = emitted_ids.get(index)
        if emitted_id and current_id:
            copied_delta.id = emitted_id
        filtered.append(copied_delta)
    if not filtered:
        return None
    copied_chunk = _copy_chunk(chunk)
    _choice_delta(copied_chunk).tool_calls = filtered
    return copied_chunk


def _synthetic_finish_chunk(
    template: Any, *, finish_reason: str = "tool_calls"
) -> Any:
    payload: dict[str, Any]
    dumper = getattr(template, "model_dump", None)
    if callable(dumper):
        original = dumper(mode="json", exclude_none=True)
        payload = {
            key: original[key]
            for key in ("id", "created", "model", "object", "system_fingerprint")
            if key in original
        }
    else:
        payload = {
            key: getattr(template, key)
            for key in ("id", "created", "model", "object", "system_fingerprint")
            if getattr(template, key, None) is not None
        }
    payload["choices"] = [
        {"index": 0, "delta": {}, "finish_reason": finish_reason}
    ]
    return type(template)(**payload)


def is_mimo_tool_request(request_data: dict[str, Any]) -> bool:
    model_values = [
        request_data.get("model"),
        (request_data.get("litellm_params") or {}).get("model"),
        (request_data.get("metadata") or {}).get("model_group"),
    ]
    return bool(request_data.get("tools")) and any(
        isinstance(value, str) and "mimo-v2.6-flash" in value.lower()
        for value in model_values
    )


def _log_guard(batch: AdaptiveToolBatch, request_id: str, *, streamed: bool) -> None:
    LOGGER.info(
        "mimo tool guard request=%s streamed=%s class=%s raw_calls=%d "
        "emitted_calls=%d buffered_bytes=%d reason=%s",
        request_id,
        streamed,
        batch.batch_class.value if batch.batch_class is not None else "none",
        len(batch.builders),
        len(batch.retained),
        batch.buffer_bytes,
        batch.stop_reason or "natural_finish",
    )


def adaptive_tool_response(response: Any, request_data: dict[str, Any]) -> Any:
    if not is_mimo_tool_request(request_data):
        return response
    choices = getattr(response, "choices", None)
    if not choices:
        return response
    message = getattr(choices[0], "message", None)
    tool_calls = list(getattr(message, "tool_calls", None) or [])
    if not tool_calls:
        return response

    batch = AdaptiveToolBatch()
    for index, call in enumerate(tool_calls):
        function = getattr(call, "function", None)
        arguments = getattr(function, "arguments", None)
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments, ensure_ascii=False)
        result = batch.feed(
            index=int(getattr(call, "index", index) or index),
            tool_id=getattr(call, "id", None),
            name=getattr(function, "name", None),
            arguments_delta=arguments if isinstance(arguments, str) else None,
        )
        if result.stop:
            break
    batch.finish()
    retained_by_index = {call.index: call for call in batch.retained}
    filtered = []
    for index, call in enumerate(tool_calls):
        call_index = int(getattr(call, "index", index) or index)
        retained = retained_by_index.get(call_index)
        if retained is None:
            continue
        copied = _copy_chunk(call)
        copied.id = retained.emitted_id
        filtered.append(copied)
    copied_response = _copy_chunk(response)
    copied_response.choices[0].message.tool_calls = filtered
    if not filtered:
        copied_response.choices[0].finish_reason = "stop"
    _log_guard(
        batch,
        str(getattr(response, "id", "unknown")),
        streamed=False,
    )
    return copied_response


async def adaptive_tool_stream(
    response: AsyncIterator[Any],
    request_data: dict[str, Any],
    *,
    now: Any = time.monotonic,
) -> AsyncGenerator[Any, None]:
    if not is_mimo_tool_request(request_data):
        async for chunk in response:
            yield chunk
        return

    iterator = response.__aiter__()
    batch: AdaptiveToolBatch | None = None
    buffered: list[Any] = []
    last_tool_chunk: Any | None = None
    closed = False
    upstream_finished = False

    async def close_upstream() -> None:
        nonlocal closed
        if closed:
            return
        closed = True
        closer = getattr(response, "aclose", None)
        if callable(closer):
            await closer()

    async def emit_retained() -> AsyncGenerator[Any, None]:
        if batch is None:
            return
        for held in buffered:
            filtered = _filtered_tool_chunk(
                held, batch.retained_indexes, batch.emitted_ids
            )
            if filtered is not None and _finish_reason(filtered) is None:
                yield filtered

    stopped_early = False
    natural_finish: Any | None = None
    try:
        while True:
            try:
                if batch is None:
                    chunk = await anext(iterator)
                else:
                    remaining = batch.seconds_remaining(now())
                    if remaining <= 0:
                        batch._stop("time_limit")
                        stopped_early = True
                        break
                    chunk = await asyncio.wait_for(anext(iterator), timeout=remaining)
            except StopAsyncIteration:
                upstream_finished = True
                if batch is not None:
                    finish_result = batch.finish()
                    if not finish_result.stop:
                        batch._stop("missing_finish")
                    stopped_early = True
                break
            except TimeoutError:
                if batch is not None:
                    batch._stop("time_limit")
                stopped_early = True
                break

            tool_deltas = _tool_deltas(chunk)
            if batch is None and not tool_deltas:
                yield chunk
                continue
            if batch is None:
                batch = AdaptiveToolBatch(started_at=now())

            buffered.append(chunk)
            if tool_deltas:
                last_tool_chunk = chunk
                for tool_delta in tool_deltas:
                    result = batch.feed(
                        index=int(getattr(tool_delta, "index", 0)),
                        tool_id=getattr(tool_delta, "id", None),
                        name=_function_value(tool_delta, "name"),
                        arguments_delta=_function_value(tool_delta, "arguments"),
                        now=now(),
                    )
                    if result.stop:
                        stopped_early = True
                        break
                if stopped_early:
                    break
            if _finish_reason(chunk) is not None:
                natural_finish = chunk
                finish_result = batch.finish()
                if finish_result.stop:
                    stopped_early = True
                    natural_finish = None
                break

        if batch is None:
            return
        if stopped_early and not upstream_finished:
            await close_upstream()
        request_id = str(
            getattr(last_tool_chunk or natural_finish, "id", "unknown")
        )
        _log_guard(batch, request_id, streamed=True)
        async for retained in emit_retained():
            yield retained
        if stopped_early:
            if last_tool_chunk is not None:
                finish_reason = "tool_calls" if batch.retained else "stop"
                yield _synthetic_finish_chunk(
                    last_tool_chunk, finish_reason=finish_reason
                )
            return
        if natural_finish is not None:
            yield natural_finish
        async for trailing in iterator:
            yield trailing
        upstream_finished = True
    finally:
        # Propagate client cancellation/early generator close to LiteLLM and
        # therefore to the in-flight vLLM request as well.
        if not upstream_finished:
            await close_upstream()
