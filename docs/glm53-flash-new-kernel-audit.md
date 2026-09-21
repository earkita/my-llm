# Audyt i strojenie kerneli GLM-5.3-Flash W4A16 na R9700

Audyt wykonano 16 września 2026 r. dla `glm53-flash-new`: vLLM recipe
`vllm_glm53flashrocm10_v0.31`, TP8 na ośmiu R9700, BF16 activation, W4A16
weights, FP8 KV, DFlash2 K4 i `ROCM_AITER_MLA_SPARSE`.

## Werdykt

- Wszystkie osiem agentów profilera zgłosiło `gfx1201`. Triton ma
  `backend=hip`, `arch=gfx1201`, `warp_size=32`; kerneli rocBLAS mają w nazwie
  `ISA1201`.
- Procesy ładują `libamdhip64`, `libtorch_hip`, `librocblas`, `libhipblas`,
  `libhipblaslt` i `librccl`. Nie załadowano rzeczywistego sterownika CUDA ani
  cuBLAS. Pliki `libcuda_stub.so` i `libcudart_stub.so` pochodzą z TileLang i
  są stubami linkera, nie backendem NVIDIA.
- Decode sparse MLA korzysta z natywnego FP8 WMMA w
  `_rdna4_fp8_paged_mqa_logits_kernel`: FP8 x FP8, akumulacja FP32; wcześniej
  sprawdzony code object zawiera `v_wmma_f32_16x16x16_fp8_fp8`.
- Prefill sparse MLA używa `_sparse_attn_prefill_ragged_kernel`. FP8 KV jest
  skalowane w tym samym kernelu przez FP32 i konwertowane do BF16 przed dot.
  Nie ma pełnego bufora pośredniego ani CPU fallbacku, ale dot nie wykorzystuje
  natywnego FP8 WMMA. To największy wykryty cast do dalszego sprawdzenia.
- W4A16 MoE używa `fused_moe_kernel_gptq_awq`. INT4 jest rozpakowywane i
  skalowane wewnątrz kernela do BF16, a dot akumuluje do FP32. Nie ma osobnej
  dekwantyzacji całych wag.
- Trace nie wykazał istotnych kopii host/device: dwa H2D, łącznie 27,44 us na
  ranku 0. Nie znaleziono CPU compute fallbacku.

## Ścieżki runtime

| operacja | faktyczny kernel/backend | dtype | gfx1201 | fallback/cast | uwaga |
| --- | --- | --- | --- | --- | --- |
| W4A16 expert GEMM | Triton `fused_moe_kernel_gptq_awq` | packed INT4 + BF16 activation, FP32 scale/accumulate, BF16 output | tak | unpack/dequant w kernelu | brakowało dokładnego pliku tuningu E288/N256 |
| sparse MLA prefill | Triton `_sparse_attn_prefill_ragged_kernel` | FP8 KV -> FP32 scale -> BF16 dot, FP32 softmax/acc | tak | scalony cast FP8->BF16 | 9,07% zsumowanego czasu GPU trace |
| sparse MLA decode | Triton `_rdna4_fp8_paged_mqa_logits_kernel` | FP8 x FP8, FP32 accumulate/output | tak, FP8 WMMA | brak pełnego castu | natywna ścieżka RDNA4 |
| indexer MQA | Triton `_fp8_mqa_logits_kernel_NUM_HEADS_32_HEAD_SIZE_128_BLOCK_KV_128` | FP8, FP32 output | tak | brak CPU | 0,27% trace |
| dense BF16 GEMM | rocBLAS/Tensile `Cijk_*ISA1201*` | BF16, typowo FP32 accumulate | tak | brak | `ROCBLAS_USE_HIPBLASLT=0`; nie mylić załadowanej biblioteki z dispatch |
| skinny BF16 GEMM | AITER `wvSplitK_hf_sml_<__hip_bfloat16,...>` | BF16, FP32 accumulate | tak | brak | wiele krótkich wywołań w decode |
| MHC/model residual | AITER/TileLang `hc_prenorm_gemm_block_m_tilelang_kernel`, `mhc_post_tilelang_kernel`, `mhc_pre_big_fuse_with_norm_tilelang_kernel` | BF16/FP32 | tak | brak CPU | drugi koszt compute po attention/MoE |
| tensor parallel | RCCL `ncclDevKernel_Generic_4(...)` | collective buffers modelu | tak | brak | dużo czasu oczekiwania/komunikacji; czas kernela nakłada się między rankami |
| KV/cache | `kernel_unified_attention`, natywne cache ops | FP8 KV, wyższa precyzja akumulacji | tak | oczekiwane skalowanie | brak FlashInfer/CUTLASS |

## Trace rocprofv3

Profil produkcyjnie zgodnego trybu objął 8K+256 i 32K+256. Sumowanie czasu
GPU po ośmiu rankach nie jest czasem ściennym, ale dobrze pokazuje udział
dispatchy w tym samym oknie:

| kernel | calls | suma GPU | udział |
| --- | ---: | ---: | ---: |
| `ncclDevKernel_Generic_4(ncclDevKernelArgsStorage<4096ul>)` | 99 533 | 200,817 s | 64,30% |
| `_sparse_attn_prefill_ragged_kernel` | 10 329 | 28,335 s | 9,07% |
| `fused_moe_kernel_gptq_awq` | 78 876 | 25,480 s | 8,16% |
| `hc_prenorm_gemm_block_m_tilelang_kernel` | 8 550 | 6,619 s | 2,12% |
| `mhc_post_tilelang_kernel` | 14 712 | 4,573 s | 1,46% |
| `wvSplitK_hf_sml_<__hip_bfloat16,32,4,16,8,2,5>(...)` | 149 679 | 3,979 s | 1,27% |
| `mhc_pre_big_fuse_with_norm_tilelang_kernel` | 84 509 | 3,697 s | 1,18% |
| `kernel_unified_attention` | 4 695 | 1,031 s | 0,33% |
| `_fp8_mqa_logits_kernel_NUM_HEADS_32_HEAD_SIZE_128_BLOCK_KV_128` | 1 045 | 0,854 s | 0,27% |
| `_rdna4_fp8_paged_mqa_logits_kernel` | 9 207 | 0,284 s | 0,09% |

Pełne CSV per PID są w
`logs/profiles/glm53-kernel-tuning/launch-trace-2/`. Tryb profilera używa
bibliotek `rocprofiler-sdk` z tego samego recipe co vLLM; mieszanie z
systemowym ROCm powodowało konflikt LLVM. Tracing obniżał wydajność, dlatego
jego wyniki tok/s nie są baseline'em.

## Strojenie W4A16 MoE

Dodano dokładny plik użytkownika vLLM:

`tuning/moe/glm53-flash-new/E=288,N=256,device_name=AMD_Radeon_R9700,dtype=int4_w4a16.json`

Mikrobenchmark na jednej R9700, z rzeczywistym BF16 + packed INT4 group-128,
wybrał kafle dla bucketów 1--4096. Dla małych batchy `16x16x64`, 1 warp i
`waves_per_eu=4` skracało sam MoE o około 20--23%; dla 4096 najlepsze
`128x64x32` poprawiało mikrobenchmark o około 12%.

Pełny A/B po osobnych restartach, warmup 1 i trzech powtórzeniach:

| workload | domyślny TTFT | strojony TTFT | domyślny decode | strojony decode | zmiana decode |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8K+256 C1 | 6,339 s | 6,326 s | 93,90 tok/s | 95,20 tok/s | +1,39% |
| 32K+256 C1 | 20,736 s | 20,610 s | 93,33 tok/s | 94,88 tok/s | +1,66% |

Prefill wyniósł odpowiednio 1292,3 vs 1294,9 tok/s dla 8K oraz 1580,3 vs
1589,9 tok/s dla 32K. API correctness gate przeszedł. Runtime jednoznacznie
zalogował `Using configuration from tuning/moe/glm53-flash-new/...`.

Artefakty A/B:

- `logs/benchmarks/glm53-kernel-tuning/fresh-default-8k/benchmark.json`;
- `logs/benchmarks/glm53-kernel-tuning/moe-tuned-8k/benchmark.json`;
- `logs/benchmarks/glm53-kernel-tuning/fresh-default-32k/benchmark.json`;
- `logs/benchmarks/glm53-kernel-tuning/moe-tuned-32k/benchmark.json`.

Zysk jest spójny, ale nie osiąga progu promocji 5%. Konfiguracja pozostaje w
trybie diagnostycznym `dflash-full-piecewise-256k-moe-tuned`; profil
produkcyjny nie został zmieniony.

## Następne wąskie gardło

Po usunięciu części kosztu MoE następnymi podejrzanymi dla decode są
TP8/RCCL, liczne małe launch'e MHC/GDN i efektywność DFlash. Nie można jednak
uznać samego 64,3% czasu kernela RCCL za 64,3% czasu ściennego: collective
może oczekiwać na inne ranki i nakładać się z ich pracą. Dla prefill największym
compute kernelem jest sparse attention, którego FP8 KV jest przed dot
konwertowane do BF16 wewnątrz kernela. Następny eksperyment powinien więc
porównać natywny FP8 dot w prefill sparse MLA, ale dopiero z testem zgodności
numerycznej i 32K/256K retrieval; obecnej poprawnej ścieżki nie należy
zastępować samą flagą.

## Prefix-cache A/B: SHA-256 2,2 GB vs xxHash 3,2 GB

Test `prefix_repetition` wykonał trzy sekwencyjne requesty z jednym wspólnym
prefiksem, bez warmupu i z identycznym seedem po obu restartach. Pierwszy TTFT
jest zimny, a wynik warm jest średnią dwóch kolejnych requestów. Obie strony
używały FP8 KV, retencji 1280, DFlash2 K4 i tych samych grafów.

| wejście | wariant | cold TTFT | warm TTFT | przyspieszenie cache |
| --- | --- | ---: | ---: | ---: |
| 32 768 | SHA-256, 2,2 GB/GPU | 25,958 s | 1,821 s | 14,26x |
| 32 768 | xxHash, 3,2 GB/GPU | 25,970 s | 1,824 s | 14,24x |
| 262 112 | SHA-256, 2,2 GB/GPU | 214,154 s | 2,785 s | 76,90x |
| 262 112 | xxHash, 3,2 GB/GPU | 213,874 s | 2,782 s | 76,89x |

Większa rezerwacja podniosła pojemność z 317 682 do 463 195 tokenów (+45,8%)
i pozostawiła około 4,2 GiB wolnego VRAM na każdej karcie. Nie poprawiła jednak
kosztu pojedynczego trafienia: różnice warm TTFT są poniżej 0,2%. Tryb
`dflash-full-piecewise-256k-cache-xxhash-3p2g` pozostaje diagnostyczny;
produkcji nie zmieniono. Większy cache ma sens dopiero przy przełączaniu wielu
długich sesji lub gałęzi, które nie mieszczą się równocześnie w 317 682
tokenach.

Artefakty:

- `logs/benchmarks/glm53-prefix-cache/baseline-32k/summary.json`;
- `logs/benchmarks/glm53-prefix-cache/xxhash-3p2g-32k/summary.json`;
- `logs/benchmarks/glm53-prefix-cache/baseline-256k/summary.json`;
- `logs/benchmarks/glm53-prefix-cache/xxhash-3p2g-256k/summary.json`.

## Kandydat upstream v0.32

Dodano izolowany profil developerski
`profiles/dev/glm53-flash-new-upstream.json` oparty na vLLM
`031f5810c1957b6ec2a7666f65a3fe68c1f7e534` z 15 września 2026 r. Profil
produkcyjny nadal używa v0.31. Nowa receptura zachowuje ten sam checkpoint
W4A16, TP8, FP8 KV, DFlash2 K4, prefix cache i limit 262144 tokenów, więc po
przełączeniu nadaje się do porównania A/B bez zmiany modelu.

Z 32 lokalnych patchy v0.31 pozostał jeden scalony, kryptograficznie związany
delta-patch. Usunięto duplikaty już obecne upstream, nieaktywne poprawki
Quark/MXFP4 i eksperymentalny filesystem offload. Zachowano poprawki potrzebne
przez W4A16 GLM na RDNA4: DFlash, kpool/sparse MLA, FP8/BF16 attention,
sharding projekcji pomocniczej, schematy narzędzi oraz retencję krótkiego
prefiksu EAGLE.

Receptura została zbudowana natywnie dla `gfx1201` i przeszła bramkę runtime
(230 pakietów, `pip check`, import Torch/vLLM/AITER). `llvm-readobj
--offloading` potwierdził obrazy `hipv4-amdgcn-amd-amdhsa--gfx1201` w
`_rocm_C.abi3.so`, `_C_stable_libtorch.abi3.so` i
`_moe_C_stable_libtorch.abi3.so`. Testy CPU przeniesionych zachowań zakończyły
się wynikiem 18/18.

Pierwszy test runtime ujawnił trzy luki w kandydacie v0.32. Dwie były
niezgodnościami upstream ROCm podczas startu: brak haka
`record_logical_topk_ready` w sparse MLA oraz brak delegacji `.compile()` w
leniwej otoczce TileLang. Naprawiają je patche `0002` i `0003`. Trzecia luka
była pominiętym podczas konsolidacji odpowiednikiem patcha v0.31 `0014`:
indexer kpool=4 przekazywał do decode tokenowy `max_model_len=262144` zamiast
pooled-state `65536`. Patch `0004` przywraca `ceil(max_model_len/index_kpool)`.

Powtarzalny A/B, raw completion, C1 i te same seedy:

| workload | v0.31 production | v0.32 przed `0004` | v0.32 po `0004` |
| --- | ---: | ---: | ---: |
| 128+512 TTFT | 177.7 ms | 230.0 ms | 177.5 ms |
| 128+512 decode | 90.1 tok/s | 52.9 tok/s | 46.8 tok/s |
| 8K+256 TTFT | 4.802 s | 4.897 s | 4.797 s |
| 8K+256 prefill | 2608.7 tok/s | 2325.6 tok/s | 2607.5 tok/s |
| 8K+256 decode | 88.3 tok/s | 57.4 tok/s | 47.9 tok/s |

`0004` usuwa więc regresję TTFT/prefill i przywraca poprawny wymiar workspace'u,
ale nie usuwa regresji decode v0.32. Zmienna akceptacja DFlash (dla krótkiego
testu 71.3% przed i 77.1% po patchu) nie tłumaczy wzrostu TPOT. Profil v0.32
pozostaje diagnostyczny; następnym krokiem jest odseparowanie target-only od
DFlash i porównanie zmian MRV2/DFlash pomiędzy commitami, a nie strojenie MoE.

Wpływ `0004` przy współbieżności czterech klientów sprawdzono osobnym,
kontrolowanym A/B na tym samym v0.32. Oba warianty używały async schedulera,
12 żądań (trzy fale C4), 8192 tokenów wejścia, 256 tokenów wyjścia i tych
samych seedów. Tryb `c4-full-indexer-workspace` ustawia
`MY_LLM_GLM53_FULL_INDEXER_WORKSPACE=1`, czyli przywraca tokenowy workspace
262144; kontrola `c4-compressed-indexer-workspace` pozostawia wymiar 65536.

| v0.32 C4, 8K+256 | pełny workspace, bez limitu `0004` | skompresowany workspace `0004` |
| --- | ---: | ---: |
| TTFT średni / p95 | 10.428 / 19.038 s | 12.854 / 22.223 s |
| effective prefill | 917.9 tok/s | 788.7 tok/s |
| decode na żądanie | 11.05 tok/s | 13.08 tok/s |
| aggregate output | 29.245 tok/s | 29.261 tok/s |
| E2E średni | 33.501 s | 32.346 s |
| DFlash acceptance | 68.7% | 61.3% |

Łączny throughput różni się tylko o 0,06%, więc `0004` nie jest ograniczeniem
przepustowości C4. Pełny workspace poprawił średni TTFT o 18,9%, ale pogorszył
decode na żądanie o 15,5% i średni E2E o 3,6%. Różnica acceptance oznacza, że
pojedynczy przebieg nie uzasadnia promocji żadnego z wariantów; domyślnie
pozostaje poprawny, skompresowany wymiar, a pełny workspace jest wyłącznie
trybem diagnostycznym.

Artefakty:

- `logs/benchmarks/glm53-flash-new-upstream/matched-v031-seed20264216/`;
- `logs/benchmarks/glm53-flash-new-upstream/patched-indexer-seed20264216/`;
- `logs/benchmarks/glm53-flash-new-upstream/matched-production-8k-seed427543/`;
- `logs/benchmarks/glm53-flash-new-upstream/patched-indexer-8k-seed427543/`;
- `logs/benchmarks/glm53-flash-new-upstream/c4-full-indexer-workspace-seed424242/`;
- `logs/benchmarks/glm53-flash-new-upstream/c4-compressed-indexer-workspace-seed424242/`.

## Minimalny RDNA4 sparse MLA (`0006`)

Minimalna receptura v0.32 otrzymała osobny kernel Triton dla rzeczywistej
geometrii GLM-5.3-Flash: rope-free `512/512/0`. Cache pozostaje FP8 w pamięci
globalnej; każdy kafel jest ładowany tylko raz, skalowany w rejestrach do BF16
i używany ponownie jako K oraz V. Nie powstaje pełny pośredni bufor BF16.

Triton wygenerował `_rdna4_fp8_glm_sparse_attn_ragged_kernel` dla
`hip:gfx1201`, wave32, z 16 KiB LDS. IR oraz ISA zawierają RDNA4 WMMA
`v_wmma_f32_16x16x16_fp8_fp8` dla QK oraz
`v_wmma_f32_16x16x16_bf16` dla PV; oba dot-producty akumulują do FP32. Q jest
kwantyzowane dynamicznie per head do FP8, natomiast latent KV pozostaje FP8
dla QK i jest skalowany w kernelu do BF16 dopiero dla PV. Nie ma pełnego
bufora pośredniego ani osobnego kernela dekwantyzacji.

Strojenie top-k 2048 na jednej R9700 wybrało `BLOCK_H=8`, `BLOCK_K=16`,
`num_stages=2` i `num_warps=8`. Zmiana 4 -> 8 warpów skróciła mikrobenchmark
BF16-QK z 1,6016 ms do 0,3953 ms i zmniejszyła spilly VGPR z 102 do 36.
Włączenie natywnego FP8-QK skróciło go dalej do 0,3280 ms. Finalny code object
używa 224 VGPR, nie ma spillów VGPR, ma 5 spillów SGPR i zerowy private
segment; wcześniejszy BF16-QK używał 256 VGPR, 36 spillów VGPR, 22 spilli SGPR
i 148 B private segmentu.

Pierwotny test BF16-QK względem referencji FP32 dał `max_abs=3.98e-4` oraz
`mean_abs=5.05e-5`. FP8-QK porównano dodatkowo z BF16-QK dla top-k
`1/17/511/2048`, trzech seedów i włączonego attention sink: wszystkie wyniki
były skończone, a względny błąd L2 wyniósł 0,04--0,95%. Pełny runtime TP8
przeszedł warmup i bramkę API bez wywołań `asm_mla.cu`.

Kontrolne A/B target-only, C1, pojedynczy zimny request i ten sam generator
promptu:

| 8K+256 | TTFT | observed prefill | decode |
| --- | ---: | ---: | ---: |
| BF16-QK, 4 warpy | 8,527 s | 960,7 tok/s | 11,21 tok/s |
| BF16-QK, 8 warpów | 7,547 s | 1085,5 tok/s | 11,73 tok/s |
| FP8-QK, 8 warpów | 7,103 s | 1153,4 tok/s | 11,98 tok/s |

Finalny wariant poprawia względem punktu wyjścia TTFT o 16,7%, observed
prefill o 20,1% i decode o 6,9%. Sam FP8-QK ponad już poprawionym wariantem
8-warpowym dodaje 6,3% prefill i 2,1% decode. Test vLLM Bench z trzema
requestami dla wariantu po zmianie liczby warpów, lecz przed FP8-QK,
potwierdził decode
12,0 tok/s (TPOT 83,37 ms); jego średnie prefill było skażone trafieniem
prefix-cache, dlatego nie służy jako wynik A/B.

Artefakty:

- `logs/validation/api-glm53-flash-upstream-minimal-kernel-20260916.json`;
- `logs/benchmarks/glm53-flash-upstream-minimal-kernel-c1-128x64-20260916.json`;
- `logs/benchmarks/glm53-flash-upstream-minimal-kernel-c1-8192x256-20260916.json`;
- `logs/benchmarks/glm53-flash-upstream-minimal-kernel-nw8-c1-8192x256-20260916.json`;
- `logs/validation/api-glm53-flash-upstream-minimal-kernel-fp8qk-20260916.json`;
- `logs/benchmarks/glm53-flash-upstream-minimal-kernel-fp8qk-c1-8192x256-20260916.json`;
- `logs/benchmarks/glm53-flash-upstream-minimal-kernel-nw8-llm-bench-20260916/`.
