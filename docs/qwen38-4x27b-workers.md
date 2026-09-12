# Qwen3.8-27B: pula czterech workerów R9700

Profil `qwen38-4x27b` uruchamia cztery niezależne repliki TP1 modelu
`amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16` jako lokalną pulę data-parallel.
Pula słucha na `127.0.0.1:8100`; główna instancja Qwen Flash pozostaje na
`127.0.0.1:8000`, a LiteLLM na `:4000`.

## Tożsamość modelu i recepta

- checkpoint: `amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16`, rewizja
  `0f7ee2559e8dbc25879e1fe1677b2b10708b91a9`;
- lokalizacja:
  `/mnt/ai/models/qwen/Qwen3.8-27B-Quark-AWQ-INT4-W4A16`;
- `model.safetensors`: 19,512,909,752 B, SHA-256
  `f1699476b8c79a9a3121469b8bdfd84c0ee55417d2af02260b0d8982257c6346`;
- Quark W4A16: symetryczny INT4, `group_size=128`, `pack_method=reorder`,
  aktywacje BF16, AWQ kalibrowany na 128 próbkach po 512 tokenów;
- 64 warstwy językowe: 48 GatedDeltaNet i 16 full-attention; Vision, `lm_head`
  i natywna warstwa MTP pozostają BF16 zgodnie z listą wyłączeń checkpointu;
- runtime: recepta `vllm_qwen38flash_pr53896`, ROCm 7.14, PyTorch 2.13,
  własne łatki `0017` i `0018`: pierwsza adaptuje liniową część vLLM PR 48606
  do API przypiętego checkoutu, a druga obsługuje skompresowane projekcje
  context-KV draftera bez założenia, że istnieje gęsty tensor `weight`;
- na `gfx1201` selektor wybiera `RDNAHybridW4A16LinearKernel` z grupą 128.

Drafter jest przypięty niezależnie do
`syvai/Qwen3.8-27B-DFlash2-W4A16`, rewizja
`4d30ec736ffc6b8688dc2ae2b502d9b48bdec279`. Jego pojedynczy plik wag ma
1,280,633,960 B i SHA-256
`ec26996e6a0745ab5edb857117220ce1e219ad524f71e6e149b703804947d8e7`.
Konfiguracja ma SHA-256
`61d6276fe8d76295232cb02d26cbb0d29c25565911f50441e779c88c9220c556`.
DFlash2 ma pięć warstw, blok o rozmiarze 8 i symetryczne
compressed-tensors W4A16, `group_size=128`; projekcje pomocnicze wskazane przez
checkpoint pozostają poza kwantyzacją.

Źródła upstream: [karta modelu AMD](https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16),
[obsługa Quark W4A16 w vLLM, PR 48606](https://github.com/vllm-project/vllm/pull/48606)
oraz [obsługa architektury Qwen3.5, PR 46110](https://github.com/vllm-project/vllm/pull/46110).
Oba PR-y były nadal otwarte 11 września 2026; dlatego profil przypina dokładny
obraz kodu i nie zakłada zgodności dowolnego wydania vLLM.

### Jak uruchamiają to inni na jednej R9700

Publiczna [recepta gfx1201 dla Qwen3.8-27B](https://github.com/zzpanic/qwen3.6-vllm-gfx1201-launchers)
używa obrazu `vllm-radiance:0.9.3`, checkpointu AutoRound W4A16, FP8 KV,
`gpu_memory_utilization=0.97`, `max_model_len=204800`, dwóch sekwencji,
budżetu batched-token 4854 i dostrojonej tabeli kafli W4A16. Jej domyślny
osobny drafter DFlash2 INT4 K4 jest pobierany z
[`syvai/Qwen3.8-27B-DFlash2-W4A16`](https://huggingface.co/syvai/Qwen3.8-27B-DFlash2-W4A16).
Wariant bez dodatkowego draftera uruchamia się tam jako
`SPEC=mtp MAXLEN=131072` i używa MTP K4. Pełne polecenia i pomiary są w
[`INT4.md`](https://github.com/zzpanic/qwen3.6-vllm-gfx1201-launchers/blob/master/INT4.md).

To nie jest bezpośrednio przenośny przepis dla pobranego checkpointu AMD:
publiczny target to `devan-carlin/Qwen3.8-27B-int4-AutoRound`, inny backend
kwantyzacji i inny układ MTP. Profil przejmuje dobór DFlash2 K4,
probabilistycznego samplera, `TRITON_ATTN` oraz standardowego FP8 KV, ale nie
przejmuje wyników AutoRound ani nie traktuje ich jako kwalifikacji checkpointu
AMD. Checkpoint AMD i drafter nie dostarczają skal kalibracyjnych KV, a
przypięty runtime nie udostępnia dynamicznego `calculate_kv_scales`; target i
draft używają więc per-tensor E4M3 ze skalą 1.0. Wymaga to osobnej kwalifikacji
jakości na lokalnych R9700.

## Topologia i parametry

Każda replika zajmuje jedną kartę. Stabilny przydział workerów to BDF-y
`07:00.0`, `0a:00.0`, `23:00.0`, `e6:00.0`; na obecnym hoście rozwiązują się
do ordinali HIP `5,6,7,1`. Menedżer odmawia startu, jeśli którykolwiek BDF
nachodzi na aktywny profil główny.

Profil ustawia DP4/TP1/PP1, kontekst 131,072, concurrency 1 na replikę,
`max_num_batched_tokens=2048`, `gpu_memory_utilization=0.96`, FP8 E4M3 KV w
układzie `LBHNC`, natywny prefix
cache z retencją 1616 oraz zewnętrzny DFlash2 K4. Draft TP wynosi 1, metoda
próbkowania jest probabilistyczna, attention draftera używa `TRITON_ATTN`, a
sam drafter działa eager. AITER pozostaje wyłączony również dla targetu.
Vision jest wyłączone, ponieważ pula ma obsługiwać workery tekstowe/kodowe.

## Instalacja i obsługa

```bash
./run model verify qwen38-4x27b
./run install --profile qwen38-4x27b --dry-run
./run install --profile qwen38-4x27b
./run doctor --profile qwen38-4x27b
./run worker-pool start qwen38-4x27b --dry-run
./run worker-pool start qwen38-4x27b
./run worker-pool status
./run worker-pool logs --follow
```

Profil buduje dla vLLM następujący kontrakt draftera (ścieżka jest rozwiązywana
z przypiętego artefaktu, nie wpisywana niezależnie do launchera):

```json
{"method":"dflash","num_speculative_tokens":4,"draft_tensor_parallel_size":1,"attention_backend":"TRITON_ATTN","draft_sample_method":"probabilistic","kv_cache_dtype":"fp8","enforce_eager":true,"model":"/mnt/ai/models/qwen/Qwen3.8-27B-DFlash2-W4A16"}
```

Łagodne zatrzymanie dotyczy wyłącznie puli i nie wysyła SIGKILL:

```bash
./run worker-pool stop
```

Profile `launcher` celowo odrzucają `qwen38-4x27b`, aby zwykłe przełączenie
głównego stosu nie zastąpiło ani nie osierociło puli dodatkowej.

## LiteLLM i Claude Code

Po uruchomieniu puli i kontrolowanym restarcie LiteLLM dostępne są aliasy:

- `qwen3.8-27b-workers-thinking` — ustawienia reasoning z karty AMD;
- `qwen3.8-27b-workers-fast` — non-thinking z temperaturą 0.7 i `top_p=0.8`.

Oba wskazują `http://127.0.0.1:8100/v1`. Żądania Claude Code z effort
`high` lub `max` są normalizowane do obsługiwanego przez Qwen `xhigh`.
Szablon znajduje się w
`templates/.claude/qwen38-4x27b/qwen38-4x27b.settings.local.json`.
Zmiana pliku LiteLLM nie restartuje działającego proxy; restart trzeba wykonać
jawnie po sprawdzeniu puli.

Kontekst 131,072 nie jest wartością przepisana z testu MI350. Bieżący start
FP8 na R9700/gfx1201 przydzielił każdej replice 11.53 GiB KV, czyli 250,106
tokenów i raportowaną concurrency 1.91x. Po wyrównaniu w dół do pełnych stron
po 1,616 tokenów praktyczny sufit pojedynczego kontekstu wynosi 248,864.
Względem limitu profilu fizyczny zapas to 119,034 tokeny. Cztery niezależne
repliki mają razem 1,000,424 sloty KV, ale cache nie jest współdzielony i nie
pozwala złożyć jednego dłuższego kontekstu.

Poniższe wyniki dotyczą wcześniejszego runtime'u MTP K1, nie nowego DFlash2.
Każda replika rzeczywiście zajmowała jedną kartę. Benchmark po trzy żądania na
rangę dał 30.91, 31.14, 32.14 i 32.58 tok/s decode, średnio 31.69 tok/s na
R9700. Cztery równoległe repliki osiągnęły 111.18 tok/s zagregowanego wyjścia,
przy indeksie równości Jaina 0.99947. Mapowanie rang to odpowiednio HIP
`5,6,7,1`, czyli BDF-y `07:00.0`, `0a:00.0`, `23:00.0`, `e6:00.0`.

## Walidacja

Pełny skan z 11 września 2026 objął jeden shard, 2,191 tensorów, 19,512,618,464
bajtów payloadu i 3,621,110,512 elementów zmiennoprzecinkowych. Wszystkie były
skończone. Walidator potwierdził 496 kompletnych zestawów Quark INT4
`weight/weight_zero_point/weight_scale`, poprawne kształty grup 128 i brak
kwantyzacji elementów z listy wyłączeń. Maszynowy raport znajduje się w
`logs/validation/qwen38-4x27b-checkpoint.json`.

Osobny pełny skan DFlash2 z 12 września 2026 objął 153 tensory, 1,280,617,536
bajtów payloadu i 207,770,880 elementów zmiennoprzecinkowych. Wszystkie były
skończone. Walidator potwierdził 36 kompletnych skompresowanych modułów W4A16,
ich I32 `weight_packed`, F16 skale, zapisane kształty, grupę 128, symetrię i
brak zero-pointów. Raport:
`logs/validation/qwen38-27b-dflash2-w4a16-checkpoint.json`.

Powtarzalne odtworzenie raportu:

```bash
scripts/run-qwen38-checkpoint-validation.sh \
  --model-dir /mnt/ai/models/qwen/Qwen3.8-27B-DFlash2-W4A16 \
  --output logs/validation/qwen38-27b-dflash2-w4a16-checkpoint.json
```

Bieżąca pula została uruchomiona po łagodnym restarcie wyłącznie worker-poola;
główny Qwen Flash i LiteLLM pozostały aktywne. Wszystkie cztery rangi przeszły
generowanie, reasoning, tryb bez thinking, wymuszone tool calling, prefix
cache oraz obecność zaakceptowanych draftów DFlash2. Każda ranga uzyskała hit
16,160 tokenów i 271/390 zaakceptowanych draftów, czyli 69.49%. Raporty:
`logs/validation/qwen38-4x27b-dflash2-fp8-features-rank{0,1,2,3}.json`.

Porównywalny benchmark 256 prompt / 64 output, 4 równoległe rangi, 3
powtórzenia dał 984.85 tok/s prefill obserwowanego przez klienta, 59.11 tok/s
decode średnio na kartę, minimum 56.31 tok/s, 185.55 tok/s zagregowanego
wyjścia i fairness Jaina 0.99840. Raport:
`logs/validation/qwen38-4x27b-dflash2-fp8-benchmark`.

Osobny czysty prefill 32,768 + 2 osiągnął 438.03 tok/s i TTFT 74.81 s. Dokładna
granica `131008 + 64 = 131072` przeszła bez OOM: 144.01 tok/s prefill, TTFT
909.73 s, 2.43 tok/s decode i E2E 935.61 s. Odpowiednie raporty to
`logs/validation/qwen38-4x27b-dflash2-fp8-prefill-32k` oraz
`logs/validation/qwen38-4x27b-dflash2-fp8-context-131072`.

Podczas czystego prefill aktywna karta raportowała 100% GFX, 6--10% UMC,
zegar GFX 2.24--2.30 GHz i lokalny zegar VRAM do 1,258 MHz. Model i KV są w
VRAM jednej karty TP1, bez CPU offloadu i bez ruchu TP, więc PCIe nie jest
ograniczeniem. Spadek od 984.85 tok/s dla 256 tokenów przez 438.03 tok/s dla
32K do 144.01 tok/s dla 131K wynika przede wszystkim z long-context attention
i ścieżek kerneli W4A16/GDN. Profil używa konserwatywnego
`max_num_batched_tokens=2048`.

Kontrolowane porównanie z 12 września 2026 użyło na każdym wariancie tego
samego czystego requestu `32768 + 2`, concurrency 1, rangi 0, FP8 KV i DFlash2
K4. Każdy wariant przeszedł przed pomiarem gate reasoning, non-thinking, tools,
prefix cache i speculative decode:

| batched tokens | prefill 32K | zmiana | peak aktywacji | KV na kartę | pełne strony KV |
|---:|---:|---:|---:|---:|---:|
| 2,048 | 438.03 tok/s | baza | 0.25 GiB | 250,106 | 248,864 |
| 4,096 | 431.13 tok/s | -1.58% | 0.66 GiB | 239,464 | 239,168 |
| 8,192 | 410.13 tok/s | -6.37% | 1.50 GiB | 219,777 | 219,776 |

Większe chunki nie poprawiły prefill na tym kernelu i odebrały KV cache.
Profil produkcyjny pozostaje więc przy 2,048. Raporty wariantów:
`logs/validation/qwen38-4x27b-dflash2-fp8-bt4096-prefill-32k` i
`logs/validation/qwen38-4x27b-dflash2-fp8-bt8192-prefill-32k`.

Pierwsza próba DFlash2 przy historycznym limicie VRAM 0.92 zakończyła się
kontrolowanym błędem przed gotowością API: dla kontekstu 131,072 wymagane było
11.07 GiB KV, a dostępne 10.24 GiB odpowiadało maksymalnie 119,952 tokenom.
Ponieważ karty workerów są dedykowane, profil podnosi limit do 0.96 i zachowuje
pełne 131,072 zamiast obniżać kontrakt kontekstu. Publiczna recepta R9700 używa
jeszcze wyższej wartości 0.97.

Historycznie przy 0.96 vLLM przydzielił 11.53 GiB BF16 KV, czyli 136,602 tokeny i
concurrency 1.04x. Następna kontrola wykazała, że DFlash2 ustawia hybrydowy
`scheduler_block_size=816`; retencja 800 używana przez MTP nie była jego
wielokrotnością. Profil zachowuje prefix cache i wyrównuje retencję do jednej
pełnej strony. Po przejściu na FP8 runtime zmierzył stronę hybrydową 1616
tokenów, dlatego bieżąca retencja wynosi 1616. Start zmierzył 11.53 GiB i
250,106 tokenów KV na kartę; żądanie graniczne 131,072 potwierdziło alokację
bez OOM.

Aktualny stan kwalifikacji runtime należy odczytać z sekcji Qwen3.8-27B w
`docs/verification.md`; wyniku MI350 nie wolno traktować jako dowodu dla
R9700/gfx1201.

Powtarzalna kwalifikacja funkcji API:

```bash
for rank in 0 1 2 3; do
  .venv/bin/python scripts/qualify-qwen38-workers.py \
    --url http://127.0.0.1:8100 \
    --model qwen3.8-27b-worker \
    --rank "${rank}" \
    --output "logs/validation/qwen38-4x27b-dflash2-fp8-features-rank${rank}.json"
done

./run benchmark --profile qwen38-4x27b \
  --url http://127.0.0.1:8100 \
  --prompt-tokens 256 --output-tokens 64 \
  --concurrency 4 --repetitions 3 --warmup 1 \
  --pin-data-parallel --timeout 300 \
  --output logs/validation/qwen38-4x27b-dflash2-fp8-benchmark

./run benchmark --profile qwen38-4x27b \
  --url http://127.0.0.1:8100 \
  --prompt-tokens 131008 --output-tokens 64 \
  --concurrency 1 --repetitions 1 --warmup 0 \
  --data-parallel-rank 0 --timeout 1800 \
  --output logs/validation/qwen38-4x27b-dflash2-fp8-context-131072
```
