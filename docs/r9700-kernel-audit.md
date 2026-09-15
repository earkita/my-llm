# Audyt kerneli Qwen na R9700/gfx1201

## Zakres i werdykt

Audyt wykonano 14 września 2026 r. dla aktywnego profilu `qwen-multi`:

- główny model: Qwen3.8 Flash-Next MXFP4/FP8, TP4/EP4 na czterech R9700;
- pula workerów: cztery niezależne Qwen3.8-27B Quark AWQ W4A16, po jednej
  R9700 na replikę, z DFlash2 K4;
- recepta obu usług: `vllm_qwen38flash_pr53896`, ROCm 7.14, PyTorch
  `2.13.0+rocm7.14.0`, vLLM `0.28.0+pr53896.89d0bb71`.

Po uzgodnieniu zakres ograniczono do konfiguracji, logów aktywnego runtime,
załadowanych bibliotek, źródeł i już wygenerowanych artefaktów Tritona. Nie
restartowano usług i nie zmieniano konfiguracji produkcyjnej.

Najważniejsze ustalenia:

1. Runtime jest rzeczywiście ROCm/HIP, a nie NVIDIA CUDA. Aktywne procesy
   ładują `libamdhip64.so.7`, `libtorch_hip.so`, `librocblas.so.5`,
   `libhipblas.so.3`, `libhipblaslt.so.1` i `librccl.so.1`. W ich mapach nie ma
   `libcuda`, `libcudart`, cuBLAS, CUTLASS ani FlashInfer.
2. Artefakty vLLM `_C.abi3.so` i `_rocm_C.abi3.so` zawierają wyłącznie bundle
   GPU `hipv4-amdgcn-amd-amdhsa--gfx1201`. Metadane sprawdzonych kerneli
   Tritona mają `backend=hip`, `arch=gfx1201`, `warp_size=32`.
3. FP8 W8A8 głównego modelu jest natywne: skompilowany
   `_w8a8_triton_block_scaled_mm` zawiera instrukcje
   `v_wmma_f32_16x16x16_fp8_fp8`. Wejścia są FP8, akumulacja FP32, a wynik
   jest zapisywany w typie modelu, BF16.
4. MXFP4 głównego modelu **nie jest natywne**. vLLM wybiera
   `EmulationMxfp4LinearKernel`; przed każdym `F.linear` dekwantyzuje pełną
   wagę MXFP4 do typu aktywacji i wykonuje GEMM w wysokiej precyzji. Główne
   MoE dekwantyzuje na bieżąco całe lokalne `w1` i `w2` do BF16, a następnie
   uruchamia BF16 `TritonExperts`. To najpoważniejszy wykryty fallback.
5. W4A16 workerów używa własnej ścieżki RDNA. Prefill wykonuje jeden scalony
   kernel Tritona, który rozpakowuje INT4 w rejestrach i robi BF16 WMMA z
   akumulacją FP32. Decode używa skinny HIP GEMM `wvSplitK_int4_hf_sml_` z
   akumulacją FP32. Nie powstaje pełny pośredni tensor zdekwantyzowanych wag.
6. PLE głównego modelu jest świadomie umieszczone na CPU. Każdy lookup kopiuje
   indeksy GPU -> CPU, wykonuje `F.embedding` na CPU i kopiuje wybrane wiersze
   CPU -> GPU. Jest to rzeczywisty CPU/device-copy fallback, zastosowany dlatego,
   że pełna tabela ma około 51 GiB przed shardingiem i nie mieści się w VRAM.

`torch.cuda`, `device="cuda"`, `torch::kCUDA` i nazwa `cudaStream_t` występują
w kodzie ROCm jako zgodny interfejs PyTorch/HIP. Nie zostały potraktowane jako
dowód użycia NVIDIA CUDA.

## Ścieżki głównego modelu

| operacja | oczekiwane | faktycznie wybrany kernel/backend | typ danych | natywne gfx1201 | fallback/cast | obawa wydajnościowa |
| --- | --- | --- | --- | --- | --- | --- |
| FP8 dense GEMM, projekcje attention | FP8 WMMA | `TritonFp8BlockScaledMMKernel` -> `_w8a8_triton_block_scaled_mm` | FP8 E4M3 x FP8 E4M3, FP32 accumulate, BF16 output | tak, Triton HIP; potwierdzone `v_wmma_f32_16x16x16_fp8_fp8` | brak dekwantyzacji do BF16 przed GEMM | brak dostrojonych JSON-ów dla kilku kształtów R9700; vLLM używa konfiguracji domyślnej |
| MXFP4 dense GEMM | scalony MXFP4/A16 GEMM | `EmulationMxfp4LinearKernel`: Quark `dequant_mxfp4` + `F.linear` | packed MXFP4 -> BF16; BF16 GEMM, zwykle FP32 accumulate | nie dla części MXFP4 | pełna waga jest dekwantyzowana przy każdym wywołaniu; aktywacja może przejść sztuczne Q/DQ | krytyczny koszt prefill i szczególnie decode |
| główne MoE MXFP4 | natywne/scalone expert GEMM | `OCP_MXQuantizationEmulationTritonExperts` -> dekwantyzacja `w1/w2` -> BF16 `fused_moe_kernel` | packed U8 MXFP4 -> BF16, BF16 WMMA/FP32 accumulate | GEMM BF16 jest natywny, kwantyzacja nie | dekwantyzacja całych lokalnych wag ekspertów na każdy forward | największy kandydat na wąskie gardło głównego modelu |
| MoE MTP FP8 | FP8 expert GEMM | backend `TRITON Fp8 MoE` | W8A8 FP8, wynik BF16 | tak według selektora; dokładna instancja wymaga trace | brak wykazanego FP8 -> BF16 przed GEMM | domyślna konfiguracja MoE dla R9700, bez pliku tuningowego |
| niekwantyzowane BF16 lineary/head/gates | BF16 WMMA | obecnie zwykłe `F.linear`; hipBLASLt jest wyłączony przez `ROCBLAS_USE_HIPBLASLT=0`, więc oczekiwana ścieżka to hipBLAS/rocBLAS-Tensile | BF16, typ akumulacji zależny od wybranego rozwiązania rocBLAS | biblioteka ROCm, ale dokładny kernel niepotwierdzony bez trace | brak castu wykazanego w kodzie | lokalny, bezpieczny Triton A16W16 istnieje, lecz jego bramka `VLLM_QWEN4_EXP_RDNA4_TRITON_BF16_GEMM=1` nie jest ustawiona |
| GDN prefill | zoptymalizowany ROCm | `Triton/FLA` | BF16/FP32 state zgodnie z warstwą | tak, generowane przez Triton HIP | brak CPU fallbacku | nie jest AITER/CK; brak śladu czasu kernela |
| GDN decode | scalony kernel | fallback do ścieżki Triton, ponieważ `_C.fused_gdn_decode_post_conv_mtp` nie jest zbudowany | BF16 model, BF16/FP32 state | tak, lecz niescalone | jawny fallback z kernela fused | dodatkowe uruchomienia kerneli na każdy token |
| QSA sparse attention | natywne sparse attention | `_qsa_sparse_paged_gqa_splitk_kernel`, `_qsa_merge_splitk_kernel`, `_qsa_mqa_paged_kernel` i kerneli pomocnicze | BF16 KV (`cache.dtype=auto`) | tak, metadane Tritona `gfx1201` | brak NVIDIA FlashInfer/CUTLASS | JIT podczas pierwszych nowych kształtów; dokładny udział w czasie wymaga trace |
| standardowe FlashAttention | zoptymalizowane ROCm FA | log: `Using FlashAttention version None`; model korzysta głównie z własnych QSA/GDN | BF16 | brak potwierdzonego standardowego FA | nie należy utożsamiać załadowanego `libaotriton_v2` z dispatch | potencjalny fallback tylko dla warstw, które próbują standardowego FA |
| KV cache | natywne GPU cache ops | BLHNC, BF16, Triton/vLLM cache ops | BF16 | tak, artefakty cache są `gfx1201` | brak kwantyzacji KV | większy ruch i pojemność niż FP8, ale bez błędu skalowania |
| PLE embedding | tabela w VRAM | `Qwen4ExpPLEFp8EmbeddingMethod` z tabelą na CPU | tabela FP8; indeks GPU -> CPU; wynik CPU -> GPU | nie | jawne dwie kopie między urządzeniami na lookup | opóźnienie PCIe i synchronizacja na krok; kompromis wymagany przez pojemność VRAM |
| MTP wielokrokowe | scalona aktualizacja attention metadata | MTP2 odbudowuje metadata między krokami | zależnie od QSA/GDN | częściowo | logowany fallback z fused multi-step | koszt sterowania i dodatkowych launchy w decode |

Skompilowana ścieżka A16W16 Triton (`_gemm_a16_w16_kernel`) ma warianty
`gfx1201` z `v_wmma_f32_16x16x16_bf16`, ale nie jest aktywowana dla tego
procesu: wymaganej zmiennej środowiskowej nie ma i w logu nie występuje
komunikat `Routed Qwen4Exp BF16 GEMMs through AITER Triton`.

## Ścieżki workerów 27B

| operacja | oczekiwane | faktycznie wybrany kernel/backend | typ danych | natywne gfx1201 | fallback/cast | obawa wydajnościowa |
| --- | --- | --- | --- | --- | --- | --- |
| W4A16 prefill (`M > 5`) | scalony W4A16 GEMM | `RDNAHybridW4A16LinearKernel` -> `_triton_w4a16_skinny_fmt_kernel` | packed INT4 + BF16 scales/activations, FP32 accumulate, BF16 output | tak; Triton HIP `gfx1201`, BF16 WMMA | rozpakowanie i skalowanie są w tym samym kernelu, bez pełnego bufora BF16 wag | instrukcje WMMA są BF16, nie INT4 matrix; koszt unpack pozostaje |
| W4A16 decode (`M <= 5`) | skinny quantized GEMM | `_rocm_C::wvSplitK_int4_g` -> `wvSplitK_int4_hf_sml_<...>` (lub wariant medium przy przekroczeniu LDS) | packed INT4 + BF16, suma FP32, BF16 output | tak; `_rocm_C` zawiera wyłącznie bundle `gfx1201` | dekwantyzacja jest scalona w HIP kernelu | właściwa ścieżka dla małego M; zwykle ograniczona odczytem wag i unpackiem |
| BF16 warstwy wyłączone z AWQ | BF16 WMMA | PyTorch linear przez ROCm BLAS; hipBLASLt wyłączony | BF16 | ROCm, dokładny kernel bez trace nieznany | brak wykazanego CPU fallbacku | MTP/head pozostają BF16 i mogą być zauważalne w decode |
| GDN prefill/decode | natywne ROCm | Triton/FLA prefill; Triton decode fallback | BF16/FP32 state | tak, Triton HIP | brak zbudowanego fused GDN decode | więcej launchy na token |
| full attention target i drafter | zoptymalizowane paged attention | `TRITON_ATTN`; `kernel_unified_attention` | FP8 E4M3 KV, obliczenia attention w wyższym typie | tak, cache artefaktu ma `arch=gfx1201` | własny ROCm paged-attention odrzucony i zastąpiony Tritonem | fallback jest jawny, ale nie ma dowodu, że AITER/custom byłby szybszy na tym kształcie |
| zapis KV | scalony cache op | `reshape_and_cache_kernel_flash` | BF16 wejście -> FP8 E4M3 cache | tak, Triton HIP `gfx1201` | cast/kwantyzacja w cache kernelu | checkpoint nie ma skal q/k/v; używane są nieskalibrowane wartości 1.0, głównie problem jakości |
| DFlash2 K4 | scalony W4A16 + Triton attention | `_prepare_dflash_inputs_kernel`, hybrydowe W4A16 i `TRITON_ATTN` | W4A16, FP8 KV | tak | brak wykazanego CPU fallbacku | skuteczność zależy od acceptance; dodatkowe draft/verify i attention rosną z kontekstem |

## Dokładne nazwy znalezione w aktywnych logach i cache

JIT monitor aktywnego głównego procesu zarejestrował:

```text
fused_moe_kernel
_qsa_sparse_paged_gqa_splitk_kernel
_qsa_merge_splitk_kernel
_compress_qsa_groups_kernel
_store_qsa_rows_kernel
_qsa_mqa_paged_kernel
_expand_qsa_indices_kernel
_triton_mrope_forward
layer_norm_fwd_kernel
_compute_local_logits_stats_kernel
_rejection_kernel
_resample_kernel
```

JIT monitor aktywnego workera zarejestrował:

```text
_triton_w4a16_skinny_fmt_kernel
_prepare_dflash_inputs_kernel
_compute_local_logits_stats_kernel
_rejection_kernel
_resample_kernel
```

Ponadto selektory runtime i odpowiadające im artefakty wskazują:

```text
_w8a8_triton_block_scaled_mm
kernel_unified_attention
reshape_and_cache_kernel_flash
wvSplitK_int4_hf_sml_<...>
wvSplitK_int4_hf_<...>
```

Sprawdzone metadane `_w8a8_triton_block_scaled_mm`,
`_triton_w4a16_skinny_fmt_kernel`, `kernel_unified_attention`,
`reshape_and_cache_kernel_flash`, `fused_moe_kernel` oraz kerneli QSA mają
`backend=hip`, `arch=gfx1201`, `warp_size=32`. W asemblerze FP8 występuje
`v_wmma_f32_16x16x16_fp8_fp8`; w prefill W4A16 i BF16 MoE występuje
`v_wmma_f32_16x16x16_bf16`.

Nie są to nazwy z osi czasu profilera. Są to nazwy faktycznie skompilowane
przez JIT monitor aktywnego runtime oraz artefakty zgodne z wybranym backendem.
Nie pozwalają przypisać procentu czasu do poszczególnych kerneli.

## Status profilera

Próba dołączenia `rocprofv3 1.3.5` do workera PID `3398933` została odrzucona:
proces nie ma wątku `rocp-bg-attach`, ponieważ nie został uruchomiony z
`ROCP_TOOL_ATTACH=1`. Serwer nie ma też skonfigurowanego vLLM torch profiler,
więc endpoint `/start_profile` nie jest zarejestrowany. Po ograniczeniu przez
użytkownika audytu do konfiguracji nie wykonywano restartu.

Pełny ślad prefill/decode wymaga jawnego, łagodnego restartu tego samego profilu
z `ROCP_TOOL_ATTACH=1` albo uruchomienia serwera pod `rocprofv3`. Dopiero taki
ślad może uczciwie podać kolejność, liczbę wywołań, czas i dokładne
instancjonowane nazwy HIP kerneli. Samo `torch.cuda` nie wystarcza.

## Bazowa wydajność bez zmian

Przed zmianą zakresu wykonano po jednym żądaniu 8192 + 256, concurrency 1,
bez warmup. Prefix cache pozostał włączony zgodnie z profilem produkcyjnym;
generator użył deterministycznego wariantu 0. To pomiar klienta end-to-end,
nie czysty czas kernela:

| runtime | prefill | TTFT | decode | E2E |
| --- | ---: | ---: | ---: | ---: |
| główny Flash MXFP4/FP8 TP4/EP4 | 1991.02 tok/s | 4.114 s | 17.28 tok/s | 18.868 s |
| worker 27B W4A16, DP rank 0 | 954.63 tok/s | 8.581 s | 27.23 tok/s | 17.945 s |

Raporty maszynowe:

- `logs/benchmarks/qwen38-flash-kernel-audit-8k-256-before.json`;
- `logs/benchmarks/qwen38-worker-kernel-audit-8k-256-before.json`.

Istniejący, zgodny tożsamością pomiar workera 32768 + 2 osiągnął 438.03 tok/s
prefill i TTFT 74.81 s. Towarzysząca mu telemetria opisana w dokumentacji
workera pokazała 100% GFX, 6--10% UMC i 2.24--2.30 GHz; SMI nie dostarczyło
bezpośredniego licznika efektywnej przepustowości pamięci. Wynik znajduje się w
`logs/validation/qwen38-4x27b-dflash2-fp8-prefill-32k`.

Nie wykonano nowego 32k dla głównego modelu ani pomiaru „after”, ponieważ po
prośbie użytkownika audyt jest tylko konfiguracyjny i żadna konfiguracja nie
została zmieniona.

## Fallbacki i zalecenia

Wykryte fallbacki, od najważniejszego:

1. **MXFP4 dense i główne MoE są emulowane.** Nie istnieje obecnie bezpieczna
   pojedyncza flaga, która zmieni ten checkpoint w natywny MXFP4 GEMM na
   R9700. Włączenie przypadkowego kernela byłoby błędne: potrzebna jest zgodność
   z OCP E2M1, skalą E8M0, grupą 32 i układem ekspertów. Realne rozwiązanie to
   dopasowany scalony kernel RDNA4 albo inny checkpoint/format z natywnym
   W4A16/FP8.
2. **PLE działa na CPU i wykonuje kopie PCIe.** Jest to celowy kompromis
   pojemnościowy. Usunięcie go wymaga innego podziału kart lub formatu/tabeli,
   nie tylko przełączenia flagi.
3. **GDN decode nie ma fused kernela**, a MTP odbudowuje metadata pomiędzy
   krokami. Oba zwiększają liczbę launchy i opóźnienie decode.
4. **Paged attention workera przechodzi z ROCm custom do Tritona.** Triton jest
   natywnym backendem HIP/gfx1201, więc słowo „fallback” nie oznacza CPU ani
   CUDA. Nie ma danych A/B uzasadniających wymuszenie AITER.
5. **Konfiguracje tuningowe FP8/MoE dla kształtów R9700 są nieobecne.** To
   najmniejszy bezpieczny obszar przyszłego strojenia: wygenerować konfiguracje
   dla dokładnych kształtów i zaakceptować je dopiero po powtarzalnym A/B.
6. `ROCBLAS_USE_HIPBLASLT=0` i wyłączony AITER są świadome. Lokalna łatka
   dokumentuje, że hipBLASLt potrafił wybrać dla tych kształtów nielegalny
   kernel ISA1201. Nie należy odwracać tych flag bez gate'u poprawności.
7. Recepta zawiera bezpieczną ścieżkę BF16 Triton dla Qwen4Exp, ale profil nie
   ustawia `VLLM_QWEN4_EXP_RDNA4_TRITON_BF16_GEMM=1`. Jest to kandydat do
   osobnego, kontrolowanego A/B; nie naprawi głównego kosztu emulacji MXFP4.

Pozostałe wąskie gardło prefill to przede wszystkim emulacja/dekwantyzacja
MXFP4 głównego modelu, długokontekstowe QSA/GDN oraz mały budżet chunków 2048.
Pozostałe wąskie gardło decode to ponowna dekwantyzacja wag, CPU PLE, brak
fused GDN/MTP metadata, koszty TP4/EP4 po PCIe oraz rosnąca z kontekstem praca
attention. U workerów prefill ogranicza W4 unpack + BF16 WMMA i attention, a
decode pozostaje głównie memory/launch-bound mimo poprawnej skinny W4A16
ścieżki.

## Zmiany

Nie zmieniono profili, kerneli ani działających usług. Dodano wyłącznie ten
raport oraz zachowano artefakty dwóch bazowych benchmarków i testów API.
