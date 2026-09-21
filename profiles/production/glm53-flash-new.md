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
| Nazwa runtime’u | `glm53-flash-w4a16-rocm10-8xr9700-tp8-dflash2-k4-fp8kv-apc-vision-256k-v032` |
| Recepta | `vllm_glm53_v0.32` |
| GPU | 8 × R9700/gfx1201 |
| Równoległość | TP8 / PP1 / DP1, bez expert parallelism |
| Maksymalny kontekst | 262144 tokeny (256K) |
| Maksymalna liczba sekwencji | 1 |
| Budżet batchowanego prefilla | 4096 tokenów |
| Wagi | compressed-tensors W4A16 |
| KV cache | FP8, 2 200 000 000 bajtów (`2200000000`), automatic prefix caching |
| DFlash | DFlash2, K=4, draft TP8 |
| Grafy | `FULL_AND_PIECEWISE` z breakable HIP graph |
| Vision | włączone; maks. 8 obrazów, bez video, `TRITON_ATTN` |
| Alias LiteLLM | `glm-5.3-flash-high` |

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
- Vision używa rozproszenia wag enkodera (`encoder_tp_mode=weights`) i limitu
  4096 tokenów obrazu;
- runtime wymaga 8 uporządkowanych patchy minimalnej recepty v0.32.

## Stan kwalifikacji

Minimalny runtime v0.32 osiągnął 60,9 tok/s decode dla `128 + 512`
(96,7% akceptacji DFlash2) oraz 50,4 tok/s dla `8K + 256` (81,6%
akceptacji). Test Vision na deterministycznym PNG przeszedł, a drugi request
z promptem 2717 tokenów wykorzystał dokładnie 1280 tokenów prefix cache.
Limit 256K i rezerwacja FP8 KV odpowiadają wcześniejszej kwalifikacji W4A16;
pełny test graniczny 256K należy jeszcze powtórzyć na minimalnej recepturze
v0.32.

Profil nie zawiera `experimental_modes`. Powiązany raport pomiarowy znajduje
się w [`../../docs/glm53-flash-new-benchmarks.md`](../../docs/glm53-flash-new-benchmarks.md).

Uruchomienie:

```bash
./run launcher switch glm53-flash-new
```
