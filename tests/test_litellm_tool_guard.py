from __future__ import annotations

import asyncio
import copy
import json
import unittest
from types import SimpleNamespace
from typing import Any

from r9700.litellm_tool_guard import (
    AdaptiveToolBatch,
    ToolClass,
    adaptive_tool_response,
    adaptive_tool_stream,
    canonical_signature,
    classify_tool,
    is_mimo_tool_request,
)


class FakeChunk:
    def __init__(
        self,
        *,
        id: str = "chatcmpl-test",
        created: int = 1,
        model: str = "mimo-v2.6-flash",
        object: str = "chat.completion.chunk",
        choices: list[dict[str, Any]] | None = None,
        system_fingerprint: str | None = None,
    ) -> None:
        self.id = id
        self.created = created
        self.model = model
        self.object = object
        self.system_fingerprint = system_fingerprint
        self.choices = [self._choice(value) for value in choices or []]

    @staticmethod
    def _choice(value: dict[str, Any]) -> SimpleNamespace:
        delta_value = value.get("delta", {})
        tool_calls = []
        for call in delta_value.get("tool_calls", []):
            function = call.get("function", {})
            tool_calls.append(
                SimpleNamespace(
                    id=call.get("id"),
                    index=call.get("index", 0),
                    type=call.get("type", "function"),
                    function=SimpleNamespace(
                        name=function.get("name"),
                        arguments=function.get("arguments"),
                    ),
                )
            )
        delta = SimpleNamespace(
            content=delta_value.get("content"),
            reasoning_content=delta_value.get("reasoning_content"),
            tool_calls=tool_calls or None,
        )
        return SimpleNamespace(
            index=value.get("index", 0),
            delta=delta,
            finish_reason=value.get("finish_reason"),
        )

    def model_copy(self, *, deep: bool) -> "FakeChunk":
        return copy.deepcopy(self) if deep else copy.copy(self)

    def model_dump(self, *, mode: str, exclude_none: bool) -> dict[str, Any]:
        del mode
        choices = []
        for choice in self.choices:
            delta: dict[str, Any] = {}
            if choice.delta.content is not None:
                delta["content"] = choice.delta.content
            if choice.delta.reasoning_content is not None:
                delta["reasoning_content"] = choice.delta.reasoning_content
            if choice.delta.tool_calls:
                delta["tool_calls"] = [
                    {
                        "id": call.id,
                        "index": call.index,
                        "type": call.type,
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in choice.delta.tool_calls
                ]
            value = {
                "index": choice.index,
                "delta": delta,
                "finish_reason": choice.finish_reason,
            }
            choices.append(
                {key: item for key, item in value.items() if not exclude_none or item is not None}
            )
        value = {
            "id": self.id,
            "created": self.created,
            "model": self.model,
            "object": self.object,
            "system_fingerprint": self.system_fingerprint,
            "choices": choices,
        }
        return {
            key: item for key, item in value.items() if not exclude_none or item is not None
        }


class FakeResponse:
    def __init__(self, chunks: list[Any]) -> None:
        self.chunks = iter(chunks)
        self.aclose_calls = 0

    def __aiter__(self) -> "FakeResponse":
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self.chunks)
        except StopIteration as error:
            raise StopAsyncIteration from error

    async def aclose(self) -> None:
        self.aclose_calls += 1


def tool_chunk(
    index: int,
    arguments: str,
    *,
    name: str | None = None,
    tool_id: str | None = None,
) -> FakeChunk:
    return FakeChunk(
        choices=[
            {
                "index": 0,
                "delta": {
                    "tool_calls": [
                        {
                            "id": tool_id,
                            "index": index,
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ]
                },
            }
        ]
    )


def finish_chunk() -> FakeChunk:
    return FakeChunk(
        choices=[{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
    )


def text_chunk(text: str) -> FakeChunk:
    return FakeChunk(choices=[{"index": 0, "delta": {"content": text}}])


def request_data() -> dict[str, Any]:
    return {
        "model": "mimo-v2.6-flash",
        "tools": [{"type": "function", "function": {"name": "Read"}}],
    }


def sse_event(event: str, payload: dict[str, Any]) -> bytes:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {data}\n\n".encode()


def sse_message_start() -> bytes:
    return sse_event(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg-test",
                "type": "message",
                "role": "assistant",
                "model": "mimo-v2.6-flash-mopd",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 0},
            },
        },
    )


def sse_tool(index: int, name: str, tool_id: str, arguments: str) -> list[bytes]:
    midpoint = max(1, len(arguments) // 2)
    return [
        sse_event(
            "content_block_start",
            {
                "type": "content_block_start",
                "index": index,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": name,
                    "input": {},
                },
            },
        ),
        sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": arguments[:midpoint],
                },
            },
        ),
        sse_event(
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": arguments[midpoint:],
                },
            },
        ),
        sse_event(
            "content_block_stop",
            {"type": "content_block_stop", "index": index},
        ),
    ]


def sse_finish() -> list[bytes]:
    return [
        sse_event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
        ),
        sse_event("message_stop", {"type": "message_stop"}),
    ]


def sse_events(chunks: list[Any]) -> list[dict[str, Any]]:
    raw = b"".join(chunk for chunk in chunks if isinstance(chunk, bytes))
    events = []
    for frame in raw.split(b"\n\n"):
        for line in frame.splitlines():
            if line.startswith(b"data: "):
                events.append(json.loads(line.removeprefix(b"data: ")))
    return events


async def collect(response: FakeResponse) -> list[FakeChunk]:
    return [chunk async for chunk in adaptive_tool_stream(response, request_data())]


def calls(chunks: list[FakeChunk]) -> list[SimpleNamespace]:
    return [
        call
        for chunk in chunks
        for choice in chunk.choices
        for call in (choice.delta.tool_calls or [])
    ]


class AdaptiveToolBatchTests(unittest.TestCase):
    def test_matches_provider_model_case_insensitively(self) -> None:
        self.assertTrue(
            is_mimo_tool_request(
                {
                    "model": "hosted_vllm/XiaomiMiMo/MiMo-V2.6-Flash-MOPD",
                    "tools": [{}],
                }
            )
        )

    def test_classifies_only_explicit_safe_tools_as_read_only(self) -> None:
        self.assertEqual(classify_tool("Read"), ToolClass.READ_ONLY)
        self.assertEqual(classify_tool("Agent"), ToolClass.AGENT)
        for name in ("Bash", "Edit", "Write", "unknown"):
            self.assertEqual(classify_tool(name), ToolClass.MUTABLE)

    def test_signature_is_stable_across_key_order(self) -> None:
        self.assertEqual(
            canonical_signature("Read", {"offset": 1, "file_path": "a"}),
            canonical_signature("Read", {"file_path": "a", "offset": 1}),
        )

    def test_fragmented_arguments_complete_once(self) -> None:
        batch = AdaptiveToolBatch(started_at=0.0)
        first = batch.feed(
            index=0,
            tool_id="one",
            name="Read",
            arguments_delta='{"file_',
            now=1.0,
        )
        second = batch.feed(
            index=0,
            tool_id=None,
            name=None,
            arguments_delta='path":"README.md"}',
            now=1.1,
        )
        self.assertFalse(first.stop)
        self.assertFalse(second.stop)
        self.assertEqual(len(batch.retained), 1)

    def test_duplicate_signature_stops_without_retaining_duplicate(self) -> None:
        batch = AdaptiveToolBatch(started_at=0.0)
        for index in (0, 1):
            result = batch.feed(
                index=index,
                tool_id=str(index),
                name="Read",
                arguments_delta='{"file_path":"README.md"}',
                now=1.0,
            )
        self.assertTrue(result.stop)
        self.assertEqual(result.reason, "duplicate_signature")
        self.assertEqual(len(batch.retained), 1)

    def test_mutable_call_stops_after_first_complete_call(self) -> None:
        batch = AdaptiveToolBatch(started_at=0.0)
        result = batch.feed(
            index=0,
            tool_id="one",
            name="Bash",
            arguments_delta='{"command":"true"}',
            now=1.0,
        )
        self.assertTrue(result.stop)
        self.assertEqual(result.reason, "mutable_limit")
        self.assertEqual(len(batch.retained), 1)

    def test_mixed_batch_keeps_only_first_class(self) -> None:
        batch = AdaptiveToolBatch(started_at=0.0)
        batch.feed(
            index=0,
            tool_id="one",
            name="Read",
            arguments_delta='{"file_path":"a"}',
            now=1.0,
        )
        result = batch.feed(
            index=1,
            tool_id="two",
            name="Agent",
            arguments_delta='{"prompt":"work"}',
            now=1.1,
        )
        self.assertTrue(result.stop)
        self.assertEqual(result.reason, "mixed_tool_classes")
        self.assertEqual([call.name for call in batch.retained], ["Read"])

    def test_fourth_read_stops_at_limit(self) -> None:
        batch = AdaptiveToolBatch(started_at=0.0)
        result = None
        for index in range(4):
            result = batch.feed(
                index=index,
                tool_id=str(index),
                name="Read",
                arguments_delta=json.dumps({"file_path": str(index)}),
                now=1.0,
            )
        assert result is not None
        self.assertTrue(result.stop)
        self.assertEqual(result.reason, "read_only_limit")
        self.assertEqual(len(batch.retained), 4)

    def test_fourth_agent_stops_at_limit(self) -> None:
        batch = AdaptiveToolBatch(started_at=0.0)
        result = None
        for index in range(4):
            result = batch.feed(
                index=index,
                tool_id=str(index),
                name="Agent",
                arguments_delta=json.dumps({"prompt": str(index)}),
                now=1.0,
            )
        assert result is not None
        self.assertTrue(result.stop)
        self.assertEqual(result.reason, "agent_limit")
        self.assertEqual(len(batch.retained), 4)

    def test_buffer_limit_fails_closed(self) -> None:
        batch = AdaptiveToolBatch(max_buffer_bytes=8, started_at=0.0)
        result = batch.feed(
            index=0,
            tool_id="one",
            name="Read",
            arguments_delta='{"file_path":"large"}',
            now=1.0,
        )
        self.assertTrue(result.stop)
        self.assertEqual(result.reason, "buffer_limit")
        self.assertEqual(batch.retained, [])

    def test_time_limit_fails_closed(self) -> None:
        batch = AdaptiveToolBatch(max_batch_seconds=1.0, started_at=0.0)
        result = batch.feed(
            index=0,
            tool_id="one",
            name="Read",
            arguments_delta='{"file_path":"a"}',
            now=1.1,
        )
        self.assertTrue(result.stop)
        self.assertEqual(result.reason, "time_limit")
        self.assertEqual(batch.retained, [])

    def test_invalid_arguments_are_never_retained(self) -> None:
        batch = AdaptiveToolBatch(started_at=0.0)
        batch.feed(
            index=0,
            tool_id="one",
            name="Read",
            arguments_delta="{broken",
            now=1.0,
        )
        result = batch.finish()
        self.assertTrue(result.stop)
        self.assertEqual(result.reason, "incomplete_arguments")
        self.assertEqual(batch.retained, [])


class AdaptiveToolStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_only_stream_is_unchanged(self) -> None:
        source = FakeResponse([text_chunk("a"), text_chunk("b")])
        output = await collect(source)
        self.assertEqual([chunk.choices[0].delta.content for chunk in output], ["a", "b"])
        self.assertEqual(source.aclose_calls, 0)

    async def test_two_unique_reads_are_emitted_on_natural_finish(self) -> None:
        source = FakeResponse(
            [
                tool_chunk(0, '{"file_path":"a"}', name="Read", tool_id="a"),
                tool_chunk(1, '{"file_path":"b"}', name="Read", tool_id="b"),
                finish_chunk(),
            ]
        )
        output = await collect(source)
        self.assertEqual([call.function.name for call in calls(output)], ["Read", "Read"])
        self.assertEqual(output[-1].choices[0].finish_reason, "tool_calls")
        self.assertEqual(source.aclose_calls, 0)

    async def test_duplicate_read_is_suppressed_and_upstream_closed(self) -> None:
        source = FakeResponse(
            [
                tool_chunk(0, '{"file_path":"a"}', name="Read", tool_id="a"),
                tool_chunk(1, '{"file_path":"a"}', name="Read", tool_id="b"),
                finish_chunk(),
            ]
        )
        output = await collect(source)
        self.assertEqual(len(calls(output)), 1)
        self.assertEqual(source.aclose_calls, 1)
        self.assertEqual(output[-1].choices[0].finish_reason, "tool_calls")

    async def test_mutable_call_is_emitted_once_and_terminates(self) -> None:
        source = FakeResponse(
            [
                tool_chunk(0, '{"command":"true"}', name="Bash", tool_id="a"),
                tool_chunk(1, '{"command":"false"}', name="Bash", tool_id="b"),
                finish_chunk(),
            ]
        )
        output = await collect(source)
        self.assertEqual([call.function.arguments for call in calls(output)], ['{"command":"true"}'])
        self.assertEqual(source.aclose_calls, 1)

    async def test_duplicate_ids_are_rewritten(self) -> None:
        source = FakeResponse(
            [
                tool_chunk(0, '{"file_path":"a"}', name="Read", tool_id="same"),
                tool_chunk(1, '{"file_path":"b"}', name="Read", tool_id="same"),
                finish_chunk(),
            ]
        )
        output = await collect(source)
        emitted_ids = [call.id for call in calls(output)]
        self.assertEqual(len(emitted_ids), 2)
        self.assertEqual(len(set(emitted_ids)), 2)

    async def test_four_agents_are_retained_and_terminate_upstream(self) -> None:
        source = FakeResponse(
            [
                tool_chunk(
                    index,
                    json.dumps({"prompt": str(index)}),
                    name="Agent",
                    tool_id=str(index),
                )
                for index in range(4)
            ]
            + [finish_chunk()]
        )
        output = await collect(source)
        self.assertEqual([call.function.name for call in calls(output)], ["Agent"] * 4)
        self.assertEqual(source.aclose_calls, 1)

    async def test_mixed_stream_keeps_only_first_call(self) -> None:
        source = FakeResponse(
            [
                tool_chunk(0, '{"file_path":"a"}', name="Read", tool_id="a"),
                tool_chunk(1, '{"file_path":"a","old":"x","new":"y"}', name="Edit", tool_id="b"),
                finish_chunk(),
            ]
        )
        output = await collect(source)
        self.assertEqual([call.function.name for call in calls(output)], ["Read"])
        self.assertEqual(source.aclose_calls, 1)

    async def test_malformed_only_call_finishes_without_executing_it(self) -> None:
        source = FakeResponse(
            [
                tool_chunk(0, "{broken", name="Read", tool_id="a"),
                finish_chunk(),
            ]
        )
        output = await collect(source)
        self.assertEqual(calls(output), [])
        self.assertEqual(output[-1].choices[0].finish_reason, "stop")
        self.assertEqual(source.aclose_calls, 1)

    async def test_missing_upstream_finish_gets_one_synthetic_finish(self) -> None:
        source = FakeResponse(
            [tool_chunk(0, '{"file_path":"a"}', name="Read", tool_id="a")]
        )
        output = await collect(source)
        self.assertEqual(len(calls(output)), 1)
        self.assertEqual(
            [chunk.choices[0].finish_reason for chunk in output].count("tool_calls"),
            1,
        )
        self.assertEqual(source.aclose_calls, 0)

    async def test_client_close_closes_upstream_once(self) -> None:
        source = FakeResponse(
            [text_chunk("before"), text_chunk("unused")]
        )
        guarded = adaptive_tool_stream(source, request_data())
        first = await anext(guarded)
        self.assertEqual(first.choices[0].delta.content, "before")
        await guarded.aclose()
        self.assertEqual(source.aclose_calls, 1)


class AnthropicSSEToolStreamTests(unittest.IsolatedAsyncioTestCase):
    async def collect_raw(self, chunks: list[Any]) -> tuple[FakeResponse, list[Any]]:
        source = FakeResponse(chunks)
        output = [
            chunk
            async for chunk in adaptive_tool_stream(source, request_data())
        ]
        return source, output

    async def test_text_only_sse_is_byte_for_byte_unchanged(self) -> None:
        wire = b"".join(
            [
                sse_message_start(),
                sse_event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    },
                ),
                sse_event(
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "text_delta", "text": "zażółć"},
                    },
                ),
                sse_event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": 0},
                ),
                *sse_finish(),
            ]
        )
        chunks = [wire[:17], wire[17:91], wire[91:]]

        source, output = await self.collect_raw(chunks)

        self.assertEqual(b"".join(output), wire)
        self.assertEqual(source.aclose_calls, 0)

    async def test_fragmented_mutable_call_is_retained_once(self) -> None:
        wire = b"".join(
            [
                sse_message_start(),
                *sse_tool(0, "Bash", "tool-a", '{"command":"true"}'),
                *sse_tool(1, "Bash", "tool-b", '{"command":"false"}'),
                *sse_finish(),
            ]
        )
        chunks = [wire[offset : offset + 23] for offset in range(0, len(wire), 23)]

        source, output = await self.collect_raw(chunks)
        events = sse_events(output)

        starts = [
            event
            for event in events
            if event.get("type") == "content_block_start"
            and event.get("content_block", {}).get("type") == "tool_use"
        ]
        self.assertEqual([event["content_block"]["id"] for event in starts], ["tool-a"])
        self.assertEqual(events[-2]["delta"]["stop_reason"], "tool_use")
        self.assertEqual(events[-1]["type"], "message_stop")
        self.assertEqual(source.aclose_calls, 1)

    async def test_fifth_read_is_not_emitted(self) -> None:
        chunks: list[bytes] = [sse_message_start()]
        for index in range(5):
            chunks.extend(
                sse_tool(
                    index,
                    "Read",
                    f"tool-{index}",
                    json.dumps({"file_path": str(index)}),
                )
            )
        chunks.extend(sse_finish())

        source, output = await self.collect_raw(chunks)
        events = sse_events(output)
        starts = [
            event
            for event in events
            if event.get("type") == "content_block_start"
            and event.get("content_block", {}).get("type") == "tool_use"
        ]

        self.assertEqual(len(starts), 4)
        self.assertEqual(source.aclose_calls, 1)

    async def test_duplicate_read_is_suppressed(self) -> None:
        chunks = [
            sse_message_start(),
            *sse_tool(0, "Read", "tool-a", '{"file_path":"README.md"}'),
            *sse_tool(1, "Read", "tool-b", '{"file_path":"README.md"}'),
            *sse_finish(),
        ]

        source, output = await self.collect_raw(chunks)
        events = sse_events(output)
        starts = [
            event
            for event in events
            if event.get("type") == "content_block_start"
            and event.get("content_block", {}).get("type") == "tool_use"
        ]

        self.assertEqual([event["content_block"]["id"] for event in starts], ["tool-a"])
        self.assertEqual(source.aclose_calls, 1)

    async def test_malformed_arguments_fail_closed(self) -> None:
        chunks = [
            sse_message_start(),
            *sse_tool(0, "Read", "tool-a", "{broken"),
            *sse_finish(),
        ]

        source, output = await self.collect_raw(chunks)
        events = sse_events(output)

        self.assertFalse(
            any(
                event.get("type") == "content_block_start"
                and event.get("content_block", {}).get("type") == "tool_use"
                for event in events
            )
        )
        self.assertEqual(events[-2]["delta"]["stop_reason"], "end_turn")
        self.assertEqual(source.aclose_calls, 1)

    async def test_client_close_propagates_to_raw_upstream(self) -> None:
        source = FakeResponse([sse_message_start(), *sse_finish()])
        guarded = adaptive_tool_stream(source, request_data())

        first = await anext(guarded)
        self.assertEqual(first, sse_message_start())
        await guarded.aclose()

        self.assertEqual(source.aclose_calls, 1)


class AdaptiveToolResponseTests(unittest.TestCase):
    @staticmethod
    def response(tool_calls: list[SimpleNamespace]) -> SimpleNamespace:
        value = SimpleNamespace(
            id="response",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(tool_calls=tool_calls),
                    finish_reason="tool_calls",
                )
            ],
        )
        value.model_copy = lambda deep: copy.deepcopy(value)
        return value

    @staticmethod
    def call(
        index: int, tool_id: str, name: str, arguments: dict[str, Any]
    ) -> SimpleNamespace:
        return SimpleNamespace(
            index=index,
            id=tool_id,
            function=SimpleNamespace(
                name=name,
                arguments=json.dumps(arguments),
            ),
        )

    def test_non_streaming_duplicate_is_filtered(self) -> None:
        response = self.response(
            [
                self.call(0, "a", "Read", {"file_path": "a"}),
                self.call(1, "b", "Read", {"file_path": "a"}),
            ]
        )
        filtered = adaptive_tool_response(response, request_data())
        self.assertEqual(len(filtered.choices[0].message.tool_calls), 1)

    def test_non_streaming_mutable_batch_is_reduced_to_one(self) -> None:
        response = self.response(
            [
                self.call(0, "a", "Bash", {"command": "true"}),
                self.call(1, "b", "Bash", {"command": "false"}),
            ]
        )
        filtered = adaptive_tool_response(response, request_data())
        self.assertEqual(len(filtered.choices[0].message.tool_calls), 1)
        self.assertEqual(response.choices[0].message.tool_calls[1].id, "b")

    def test_non_streaming_malformed_call_is_removed(self) -> None:
        response = self.response(
            [
                SimpleNamespace(
                    index=0,
                    id="a",
                    function=SimpleNamespace(name="Read", arguments="{broken"),
                )
            ]
        )
        filtered = adaptive_tool_response(response, request_data())
        self.assertEqual(filtered.choices[0].message.tool_calls, [])
        self.assertEqual(filtered.choices[0].finish_reason, "stop")


if __name__ == "__main__":
    unittest.main()
