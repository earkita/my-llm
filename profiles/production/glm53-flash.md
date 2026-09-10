# GLM 5.3 Flash — profil produkcyjny

Źródłem prawdy jest sąsiedni plik [`glm53-flash.json`](glm53-flash.json).
Ten opis należy aktualizować w tym samym commicie co profil.

## W skrócie

| Parametr | Wartość |
|---|---|
| Status | `production-ready` |
| Model | `glm53-flash-quark-mxfp4` |
| Checkpoint | `/mnt/ai/models/glm/GLM-5.3-Flash-Quark-MXFP4` |
| Runtime | vLLM na ROCm 10 |
| Nazwa runtime’u | `glm53-flash-quark-mxfp4-rocm10-mrv2-8xr9700-tp8-noep-mxfp4gemv-dflash2-k4-fp8kv-512k-vision-weights` |
| Recepta | `vllm_glm53flashrocm10_v0.29` |
| GPU | 8 × R9700 |
| Równoległość | TP8 / PP1 / DP1, bez expert parallelism |
| Maksymalny kontekst | 524288 tokenów (512K) |
| Maksymalna liczba sekwencji | 1 |
| Budżet batchowanego prefilla | 4096 tokenów |
| Wagi | AMD Quark MXFP4 |
| KV cache | FP8, 4 960 000 000 bajtów (`4960000000`), bloki po 16 tokenów, automatic prefix caching |
| DFlash | DFlash2, K=4, draft TP8 |
| Vision | włączone, do 8 obrazów, do 4096 tokenów na obraz |
| Alias LiteLLM | `glm-5.3-flash-high` |

## Backend i wykonanie

- attention: `ROCM_AITER_MLA_SPARSE`;
- attention enkodera Vision: `TRITON_ATTN`;
- wagi enkodera Vision są dzielone przez TP8;
- MXFP4 GEMV korzysta z implementacji Triton dla RDNA4;
- warstwy MoE i linear używają backendu `emulation`;
- scheduler jest synchroniczny i działa z `enforce_eager`;
- natywny vLLM automatic prefix caching jest włączony przez
  `--enable-prefix-caching` oraz `--prefix-cache-retention-interval 1280`;
  API raportuje `usage.prompt_tokens_details.cached_tokens`, a `/metrics`
  udostępnia `vllm:prefix_cache_queries_total` i
  `vllm:prefix_cache_hits_total`;
- limit kontekstu `524288` zachowuje stałą rezerwę KV `4960000000`;
- checkpoint jest ładowany z 62 shardów Safetensors;
- runtime wymaga 25 uporządkowanych patchy v0.29.

## Świadomie wyłączone

- `experimental_modes` — warianty robocze należą do `profiles/dev/`;
- LMCache oraz CPU/NVMe KV offload;
- expert parallelism;
- output-tiled BN8 z runtime v0.31.

## Stan kwalifikacji

Profil korzysta ze stabilnego wariantu v0.29 z kontekstem 512K i obsługą
Vision. Konfiguracja APC odpowiada przetestowanej geometrii tego samego runtime
na profilu uncensored: strony attention/DFlash 1280 i stany Mamba 16.
Konfiguracja produkcyjna nie zawiera wariantów eksperymentalnych. Stan
działającej usługi sprawdza się osobno poleceniem:

```bash
./run launcher status
```
