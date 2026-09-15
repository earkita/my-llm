# GLM 5.3 Flash W4A16 — profil produkcyjny

Źródłem prawdy jest sąsiedni plik [`glm53-flash-new.json`](glm53-flash-new.json).
Ten opis należy aktualizować w tym samym commicie co profil.

## W skrócie

| Parametr | Wartość |
|---|---|
| Status | `production-ready` |
| Model | `glm53-flash-w4a16-mtp` |
| Checkpoint | `/mnt/ai/models/glm/GLM-5.3-Flash-W4A16-MTP` |
| Runtime | vLLM na ROCm 10 |
| Nazwa runtime’u | `glm53-flash-w4a16-rocm10-8xr9700-tp8-dflash2-k4-fp8kv-apc-256k-full-piecewise` |
| Recepta | `vllm_glm53flashrocm10_v0.31` |
| GPU | 8 × R9700/gfx1201 |
| Równoległość | TP8 / PP1 / DP1, bez expert parallelism |
| Maksymalny kontekst | 262144 tokeny (256K) |
| Maksymalna liczba sekwencji | 1 |
| Budżet batchowanego prefilla | 4096 tokenów |
| Wagi | compressed-tensors W4A16 |
| KV cache | FP8, 2 200 000 000 bajtów (`2200000000`), automatic prefix caching |
| DFlash | DFlash2, K=4, draft TP8 |
| Grafy | `FULL_AND_PIECEWISE` z breakable HIP graph |
| Vision | wyłączone; language-model-only |
| Alias LiteLLM | `glm-5.3-flash-new-high` |

## Backend i wykonanie

- attention: `ROCM_AITER_MLA_SPARSE`;
- MoE W4A16: `triton`, w runtime wybierany jako `TritonWNA16Experts`;
- lineary: selektor `auto`;
- target i draft przechowują KV w FP8;
- DFlash2 K4 używa `TRITON_ATTN` i draft TP8;
- scheduler jest synchroniczny, a `enforce_eager=false`;
- `VLLM_USE_BREAKABLE_CUDAGRAPH=1` umożliwia pełne i odcinkowe grafy HIP;
- automatic prefix caching ma interwał retencji 1280 tokenów;
- LMCache, CPU/NVMe KV offload, PyNccl i custom all-reduce są wyłączone;
- checkpoint jest ładowany z 11 shardów Safetensors;
- runtime wymaga 32 uporządkowanych patchy recepty v0.31.

## Stan kwalifikacji

Pełny request graniczny `262016 + 128` przeszedł z TTFT 214,1 s,
prefillem 1223,9 tok/s i decode 35,7 tok/s. Test dwóch requestów
`262080 + 32` ze wspólnym prefiksem obniżył TTFT z 214,44 s do 2,729 s,
czyli 78,6 razy. Batch 8192 był o 1,1% wolniejszy dla prefilla 256K i nie
dał istotnego zysku decode, dlatego produkcja zachowuje 4096.

Profil nie zawiera `experimental_modes`. Powiązany raport pomiarowy znajduje
się w [`../../docs/glm53-flash-new-benchmarks.md`](../../docs/glm53-flash-new-benchmarks.md).

Uruchomienie:

```bash
./run launcher switch glm53-flash-new
```
