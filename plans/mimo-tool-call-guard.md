# MiMo adaptive tool-call guard

## Goal

Restore useful parallel tool use for `mimo-v2.6-flash` without allowing one
model completion to execute a duplicated batch, consume the full output
budget, or mix dependent mutations concurrently.

The former production fence remains the rollback baseline:

- `parallel_tool_calls: false`;
- stop generation after the first `</tool_call>`.

It is safe, but it turns every tool into another model round trip and prevents
parallel repository exploration and agent fan-out. The adaptive guard replaced
that fence after passing the live OpenAI and Anthropic qualification cases.

## Placement

Implement the guard in the existing LiteLLM callback path, not in the MiMo
parser:

- pure policy and stream state machine: `r9700/litellm_tool_guard.py`;
- integration: `LocalRequestNormalizationHook.async_post_call_streaming_iterator_hook`
  in `config/litellm_hooks.py`;
- configuration stays bound to the `mimo-v2.6-flash` alias;
- `0006-fix-qwen3-streaming-final-parameter.patch` remains responsible only
  for syntactically complete streamed arguments.

LiteLLM runs the iterator hook before serializing SSE to Claude Code. The
wrapper can therefore buffer tool-call deltas, compare completed calls before
exposing them, close the upstream iterator on a violation, and emit one clean
terminal event. Closing the upstream iterator must be verified to cancel the
corresponding vLLM request; hiding chunks while generation continues is not an
acceptable implementation.

If the installed LiteLLM iterator cannot cancel and terminate a transformed
Anthropic stream correctly, the fallback is a recipe-bound vLLM serving patch.
Do not put duplicate suppression inside the tool parser unless the parser can
also abort generation.

## Policy

Canonical tool signature:

```text
tool name + NUL + canonical JSON arguments
```

Canonical JSON uses sorted keys and compact separators. Invalid JSON is never
executed.

Tool classes and limits per assistant response:

| class | initial members | emitted limit | behavior |
| --- | --- | ---: | --- |
| read-only | `Read`, `Glob`, `Grep`, `WebSearch`, `WebFetch` | 4 unique | may run in parallel |
| agent fan-out | `Agent` | 4 unique | may run in parallel only when every call is an agent call |
| mutable or ambiguous | `Bash`, `Edit`, `Write`, `NotebookEdit`, unknown tools | 1 | force a single call |

Additional invariants:

1. Exact duplicate signatures are emitted once.
2. Duplicate `tool_use` IDs are replaced with unique IDs before emission.
3. A mixed batch containing a mutable or unknown tool is reduced to its first
   call. The model can request the remaining work after the result.
4. The fifth completed read-only or agent call terminates upstream generation;
   it is not exposed to the client.
5. The first repeated signature terminates upstream generation immediately.
6. A batch is capped independently by 64 KiB of buffered tool data and a
   10-second generation deadline. Exceeding either limit fails closed to the
   first valid call.
7. Text-only responses and responses without tools pass through unchanged.
8. No tool arguments are written to telemetry. Record only counts, tool
   classes, termination reason, token usage and request ID.

`Bash` deliberately remains ambiguous. Command-string heuristics are too easy
to evade and cannot reliably distinguish a read from a mutation such as
`fuser -k`, redirection, generated scripts, or nested shells.

## Streaming algorithm

1. Pass through ordinary text and thinking until the first tool-call delta.
2. Buffer tool-call deltas instead of forwarding them immediately.
3. Reassemble each call by tool index and validate its JSON arguments.
4. Apply the policy to the completed batch.
5. For a normal upstream finish, emit the retained calls followed by the
   original terminal event.
6. On duplicate, count, byte or time limit:
   - close the upstream async iterator;
   - confirm the vLLM request leaves `num_requests_running`;
   - emit retained calls and a synthesized `tool_calls`/`tool_use` terminal
     event with correct partial usage;
   - never emit the violating call.
7. Preserve backpressure and propagate client disconnect cancellation through
   the wrapper.

Buffering begins only at the first tool delta. It slightly delays tool
execution, but prevents Claude Code from executing a block before the guard
knows whether the response is a duplicated or unsafe mixed batch.

## Tests

### Offline unit tests

- arguments split across arbitrary chunk boundaries;
- four unique reads retained in order;
- fifth read closes upstream and is suppressed;
- repeated signature closes upstream and is suppressed;
- duplicate IDs are rewritten uniquely;
- one `Bash` retained and subsequent calls suppressed;
- mixed read/write batch reduced to one call;
- malformed JSON fails closed;
- text-only and reasoning responses remain byte-for-byte equivalent;
- iterator cancellation and client cancellation both call `aclose()` once;
- exactly one terminal event is emitted.

### Live API qualification

Run through both Anthropic and OpenAI LiteLLM endpoints with streaming enabled:

1. adversarial repeated `Bash`: one visible call and bounded output;
2. four independent reads: four unique calls;
3. five independent reads: four calls, upstream canceled;
4. four unique agents: four calls with unique IDs;
5. repeated agent batch: first unique batch only;
6. mixed `Read` plus `Edit`: one call;
7. malformed final parameter: valid JSON through patch `0006`;
8. two-turn tool loop: result is accepted and the model completes normally;
9. plain response and image request remain unchanged.

Repeat the real Claude Code compaction reproduction at 32K and at the observed
320K-class context. Group transcript entries by request ID and require:

- no duplicate canonical signature in a response;
- no more than the class limit;
- no tool response reaches `max_tokens`;
- the backend request disappears after an early guard termination.

Compare current single-call fence and adaptive guard for repository discovery,
four independent reads, validation commands and four-agent fan-out. Record
TTFT, total wall time, model turns, prefill tokens, cache hits, output tokens
and executed tool count.

## Rollout

1. Build the pure state machine and offline tests while production remains on
   the one-call fence.
2. Run the adaptive guard on a development MiMo alias and the adversarial live
   suite. Completed with `mimo-v2.6-flash-adaptive` before promotion.
3. Enable parallel reads only; keep `Bash`, mutations and agents sequential.
4. After a clean Claude Code soak, enable bounded unique `Agent` fan-out.
5. Remove the YAML `</tool_call>` stop and `parallel_tool_calls: false` only
   after all live gates pass. Completed for the production alias.
6. Roll back instantly by restoring those two YAML settings and restarting
   LiteLLM; the vLLM runtime does not need a restart.

Production promotion requires zero duplicate executions, zero malformed tool
arguments, confirmed upstream cancellation, and lower wall time than the
single-call fence for both four-read and four-agent workloads.

## External evidence

- Anthropic documents parallel calls as the efficient default for independent
  work and sequential execution as correct for dependent operations:
  <https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use>
- Claude Code currently executes duplicated parallel blocks without a built-in
  deduplication or fan-out cap: <https://github.com/anthropics/claude-code/issues/64080>
- A request for scoped sequential mutations while retaining parallel reads
  reports substantially fewer collateral cancellations in sequential mode:
  <https://github.com/anthropics/claude-code/issues/64237>
- vLLM's `parallel_tool_calls=false` path filters output to the first call; it
  does not provide the adaptive policy or generation cancellation required
  here: <https://docs.vllm.ai/en/latest/api/vllm/entrypoints/serve/utils/tool_calls_utils/>
