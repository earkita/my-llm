# Qwen3.8 Flash-Next UNCENSORED MXFP4/FP8

## Wynik

Konwersja została ukończona 11 września 2026 r. i zweryfikowana na czterech
fizycznych kartach R9700 (`gfx1201`). Źródło i wzorzec nie zostały zmienione.

- źródło: `/mnt/ai/models/qwen/Qwen3.8-Flash-Next-UNCENSORED-FP8`;
- wzorzec formatu: `/mnt/ai/models/qwen/Qwen3.8-Flash-Next-MXFP4-FP8`;
- wynik: `/mnt/ai/models/qwen/Qwen3.8-Flash-Next-UNCENSORED-MXFP4-FP8`;
- profil produkcyjny: `profiles/production/qwen38-flash-uncensored.json`;
- recepta runtime: `vllm_qwen38flash_pr53896`.

Wzorzec posłużył wyłącznie do odczytu konfiguracji kwantyzacji, nazw, typów,
kształtów i rozmieszczenia tensorów w shardach. Konwerter nie odczytuje jego
payloadów safetensors, nie kopiuje z niego wag ani skal. Wszystkie wartości w
wyniku są kopiowane lub wyliczane ze źródła UNCENSORED.

## Odtworzenie

Publiczny wrapper wiąże konwersję z Pythonem i ROCm zainstalowanej recepty,
ustawia `gfx1201` oraz implementację Triton AMD Quark. Użyte wersje to AMD
Quark `0.12.post1`, compressed-tensors `0.17.0`, safetensors `0.8.0` i PyTorch
`2.13.0+rocm7.14.0`.

```bash
scripts/run-qwen38-mxfp4-conversion.sh dry-run \
  --source /mnt/ai/models/qwen/Qwen3.8-Flash-Next-UNCENSORED-FP8 \
  --reference /mnt/ai/models/qwen/Qwen3.8-Flash-Next-MXFP4-FP8 \
  --output /mnt/ai/models/qwen/Qwen3.8-Flash-Next-UNCENSORED-MXFP4-FP8

scripts/run-qwen38-mxfp4-conversion.sh convert \
  --source /mnt/ai/models/qwen/Qwen3.8-Flash-Next-UNCENSORED-FP8 \
  --reference /mnt/ai/models/qwen/Qwen3.8-Flash-Next-MXFP4-FP8 \
  --output /mnt/ai/models/qwen/Qwen3.8-Flash-Next-UNCENSORED-MXFP4-FP8 \
  --device cuda:0 \
  --expert-chunk 8

scripts/run-qwen38-mxfp4-conversion.sh validate \
  --source /mnt/ai/models/qwen/Qwen3.8-Flash-Next-UNCENSORED-FP8 \
  --reference /mnt/ai/models/qwen/Qwen3.8-Flash-Next-MXFP4-FP8 \
  --output /mnt/ai/models/qwen/Qwen3.8-Flash-Next-UNCENSORED-MXFP4-FP8
```

`convert` celowo odmawia pracy, gdy katalog wynikowy już istnieje. Zapisuje do
ukrytego katalogu `.Qwen3.8-Flash-Next-UNCENSORED-MXFP4-FP8.partial` i dopiero
po ukończeniu zmienia jego nazwę atomowo. Po przerwaniu można kontynuować tę
samą konwersję opcją `--resume`; gotowego checkpointu nie nadpisuje.

Model jest przetwarzany shard po shardzie. Eksperci są obrabiani po osiem,
więc pełny checkpoint ani jego pełna reprezentacja BF16 nie trafia jednocześnie
do RAM lub VRAM.

## Inwentaryzacja i schemat

| Checkpoint | Shardy safetensors | Tensory | Typy tensorów | Payload tensorów |
| --- | ---: | ---: | --- | ---: |
| źródło | 131 | 152 089 | BF16 76 694; F8_E4M3 75 392; I64 3 | 185 502 232 570 B |
| wzorzec | 132, w tym osobny KV-scales | 5 168 | BF16 2 822; F8_E4M3 1 827; U8 480; F32 36; I64 3 | 125 789 603 234 B |
| wynik | 131 | 5 132 | BF16 2 822; F8_E4M3 1 827; U8 480; I64 3 | 125 789 603 090 B |

Indeks wzorca deklaruje `125 790 261 210` B, czyli o `657 976` B więcej niż
wynika z nagłówków tensorów. Indeks wyniku został wyliczony od nowa i jego
`metadata.total_size` jest równy rzeczywistym `125 789 603 090` B. Konfiguracje
modelu źródła i wzorca są semantycznie identyczne po pominięciu
`quantization_config`.

Odtworzony format compressed-tensors ma nazwę `mxfp4-pack-quantized`:

- zwykłe warstwy `Linear` mają statyczne, symetryczne wagi MXFP4 E2M1,
  grupę 32, skale E8M0 i aktywacje A16;
- eksperci i shared expert MTP zachowują FP8 E4M3: statyczne bloki wag
  `128x128` z BF16 `weight_scale_inv` i dynamiczne aktywacje FP8 w grupach 128;
- projekcje self-attention `q/k/v/o` oraz linear-attention
  `in_proj_qkv/in_proj_z/out_proj` mają ten sam schemat FP8;
- wyłączone są `lm_head`, embeddingi, vision, `fc_embedding`, `fc_hidden`,
  bramki MoE, indexer, PLE, `conv1d`, `in_proj_a/b/ba`, hyper-connection,
  input-mix, block-inject i normy.

Główne routed experts źródła są najpierw odtwarzane jako
`real_weight = fp8_weight * bf16_weight_scale_inv` dla każdego bloku
`128x128`, a dopiero potem kwantyzowane przez AMD Quark RTN do MXFP4 z grupą
32 i zaokrąglaniem skali `EVEN`. Nie pominięto więc skal wejściowego FP8.
Główne projekcje attention pochodzące z BF16 są kwantyzowane do E4M3 z
BF16 `weight_scale_inv`; nie korzystają ze skal wzorca.

Przykładowo pojedynczy ekspert warstwy 0 ma w źródle `gate_proj` i `up_proj`
F8 `[640, 2560]`, skalę BF16 `[5, 20]`, a `down_proj` F8 `[2560, 640]`.
Wynik łączy 512 ekspertów w:

- `gate_up_proj_packed` U8 `[512, 1280, 1280]` i skalę U8
  `[512, 1280, 80]`;
- `down_proj_packed` U8 `[512, 2560, 320]` i skalę U8
  `[512, 2560, 20]`.

Shared expert `gate_proj` przechodzi z BF16 `[640, 2560]` do packed U8
`[640, 1280]` ze skalą `[640, 80]`. `linear_attn.in_proj_qkv` przechodzi z
BF16 `[10240, 2560]` do F8 o tym samym kształcie ze skalą BF16 `[80, 20]`;
analogicznie `self_attn.q_proj` warstwy 3 ma F8 `[12288, 2560]` i skalę
`[96, 20]`.

Wzorzec zawiera dodatkowe 36 skal aktywacji KV w F32 (144 B) dla co czwartej
warstwy attention. Są wynikiem kalibracji aktywacji i nie dają się poprawnie
wyprowadzić z samych wag źródła. Nie zostały skopiowane. Tak jak deklaruje
`kv_cache_scheme=null`, profil używa `cache.dtype=auto` (BF16), a nie FP8 KV.
Tokenizer, `chat_template.jinja`, generation config i pozostałe pliki
pomocnicze pochodzą dokładnie ze źródła UNCENSORED.

## Walidacja checkpointu

Plan zawiera 5 132 tensorów: 4 326 kopii ze źródła, 163 pary waga/skala FP8,
96 par scalonych wag/skali MXFP4 oraz 144 par niescalonych wag/skali MXFP4.
Walidacja zakończyła się bez brakujących, nadmiarowych lub nieskończonych
tensorów, bez niezgodności typu i kształtu oraz bez błędów inwentaryzacji.
Sprawdzono skończoność 58 968 416 765 elementów i SHA-256 wszystkich 4 326
tensorów kopiowanych bez transformacji. Sprawdzono też 13 plików pomocniczych;
nie było braków ani różnic hashy. Hash indeksu wyniku to
`44957661ab0f3d00a636fc2cd5c7083bc3f844ee3026ab3e024793cffe3f6e3f`.

Surowe raporty lokalne znajdują się w:

- `logs/conversion/qwen38-uncensored-mxfp4-fp8/checkpoint-validation.json`;
- `logs/conversion/qwen38-uncensored-mxfp4-fp8/config-comparison.json`;
- `logs/conversion/qwen38-uncensored-mxfp4-fp8/plan-summary.json`;
- `logs/conversion/qwen38-uncensored-mxfp4-fp8/tensor-plan.tsv`.

## Walidacja R9700/gfx1201

Checkpoint został załadowany przez zarządzany pełny stack vLLM + LiteLLM na
GPU 0-3, z TP4, EP4, MTP2, `max_model_len=262144` i
`gpu_memory_utilization=0.95`. GPU 4-7 pozostają wolne dla planowanego układu
`4+1+1+1+1`. Nie zatrzymano ani nie zastąpiono obcej działającej usługi.

Runtime vLLM `0.28.0+pr53896.89d0bb71` wymaga zapisanych w recepcie patchy
0010-0016: obsługi mieszanych grup FP8/MXFP4, CPU-offload tabeli PLE,
mapowania obu rodzajów skal FP8 Qwen4, rozdzielenia backendów MTP i głównego
MoE, replikacji niepodzielnego shared expert pod TP4 oraz zachowania A16 dla
weight-only MXFP4. Główne MoE MXFP4 korzysta z emulacji OCP, a FP8 MTP z
Tritona.

Testy API przeszły:

- zwykłe generowanie: pytanie `17 * 19` zwróciło `323`;
- reasoning: odpowiedź była poprawna, a osobne pole reasoning zawierało 189
  tokenów;
- tool calling: model zakończył przez `tool_calls`, wybrał `get_weather` i
  przekazał `{"city":"Warsaw"}`;
- test proxy LiteLLM przeszedł.

Dokładny test `256` tokenów promptu + `64` tokeny wyjścia, concurrency 1,
jedna rozgrzewka i dwa pomiary wykonano poleceniem:

```bash
skills/measure-r9700-model/scripts/test-and-benchmark.sh \
  --profile qwen38-flash-uncensored \
  --prompt-tokens 256 \
  --output-tokens 64 \
  --concurrency 1 \
  --repetitions 2 \
  --warmup 1
```

Uzyskano średnio 532,65 tok/s obserwowanego prefill i 15,43 tok/s decode
(minimum 13,51 tok/s), średni TTFT 0,481 s i p95 E2E 5,091 s. Test potwierdził
tożsamość zarządzanego procesu, health, nazwę serwowanego modelu, deklarowany
limit kontekstu i dokładne liczniki użycia. Raporty to
`logs/validation/api-qwen38-flash-20260911T183258.json` oraz
`logs/benchmarks/qwen38-flash-c1-256x64-20260911T183258.json`.

Limit 262 144 tokenów został sprawdzony w konfiguracji i tożsamości runtime,
ale nie został jeszcze wykonany end-to-end dla tego checkpointu. Krótki test
nie jest kwalifikacją obciążenia wielowątkowego ani długiego thermal soak.
Dowodem kompatybilności jest opisany test na czterech rzeczywistych
R9700/gfx1201; zgodność z MI350 nie jest tu traktowana jako potwierdzenie
działania na RDNA4.

## Claude Code

Claude Code `2.1.197` został sprawdzony przez LiteLLM na tej samej działającej
instancji 4 × R9700. Role główne używały thinking, a hook proxy zmienił effort
`high` na obsługiwane przez Qwen `xhigh`. Gate przeszedł 21/21 warunków:
odpowiedź literalną, zadanie rachunkowe z reasoning, non-thinking test aliasu
fast oraz dziewięcioturową naprawę kodu z wywołaniami `Bash`, `Bash`, `Read`,
`Read`, `Bash`, `Edit`, `Bash`, `Bash`. Łącznie zaobserwowano dziewięć bloków
thinking, a alias fast zwrócił zero. Testy po zmianie przeszły, plik testowy
nie został zmieniony i nie wystąpiły odmowy uprawnień.

Odtwarzalny gate dla aktywnego profilu:

```bash
./run test claude-code \
  --profile qwen38-flash-uncensored \
  --output logs/validation/claude-code-qwen38-flash-uncensored-thinking.json \
  --timeout 600
```

Raport tej kwalifikacji zapisano jako
`logs/validation/claude-code-qwen38-flash-uncensored-thinking.json`.
