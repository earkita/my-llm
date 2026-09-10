# GLM 5.3 Flash UNCENSORED — profil produkcyjny

Źródłem prawdy jest sąsiedni plik
[`glm53-flash-uncensored.json`](glm53-flash-uncensored.json). Ten opis należy
aktualizować w tym samym commicie co profil.

## W skrócie

| Parametr | Wartość |
|---|---|
| Status | `production-ready` |
| Model | `glm53-flash-uncensored-quark-mxfp4` |
| Checkpoint | `/mnt/ai/models/glm/GLM-5.3-Flash-UNCENSORED-Quark-MXFP4` |
| Źródło | `dealignai/GLM-5.3-Flash-UNCENSORED-FP8` @ `3591f3347b44ba8f0cc6769d85a82b10cddcd572` |
| Runtime | vLLM na ROCm 10 |
| Nazwa runtime’u | `glm53-flash-uncensored-quark-mxfp4-rocm10-mrv2-8xr9700-tp8-noep-mxfp4gemv-dflash2-k4-fp8kv-768k-vision-weights` |
| Recepta | `vllm_glm53flashrocm10_v0.29` |
| GPU | 8 × R9700 |
| Równoległość | TP8 / PP1 / DP1, bez expert parallelism |
| Maksymalny kontekst | 786432 tokenów |
| Maksymalna liczba sekwencji | 1 |
| Budżet batchowanego prefilla | 4096 tokenów |
| Wagi | lokalna konwersja AMD Quark MXFP4 |
| KV cache | FP8, 4 960 000 000 bajtów (`4960000000`), bloki po 16 tokenów |
| DFlash | DFlash2, K=4, draft TP8 |
| Vision | włączone, do 8 obrazów, do 4096 tokenów na obraz |
| Alias LiteLLM | `glm-5.3-flash-uncensored-high` |

## Backend i wykonanie

- attention: `ROCM_AITER_MLA_SPARSE`;
- attention enkodera Vision: `TRITON_ATTN`;
- wagi enkodera Vision są dzielone przez TP8;
- MXFP4 GEMV korzysta z implementacji Triton dla RDNA4;
- warstwy MoE i linear używają backendu `emulation`;
- scheduler jest synchroniczny i działa z `enforce_eager`;
- checkpoint jest ładowany z 62 shardów Safetensors;
- runtime wymaga 25 uporządkowanych patchy v0.29.

## Pochodzenie i weryfikacja modelu

Checkpoint powstał przez lokalną konwersję wag FP8 do Quark MXFP4. Walidacja
offline objęła 76108 tensorów i nie wykazała braków ani różnic w nazwach,
kształtach lub typach. Hash 3280 tensorów i skal zachowanych bez konwersji był
zgodny ze źródłem. Cztery reprezentatywne próbki po dekwantyzacji osiągnęły
cosine similarity od 0.99284 do 0.99336. Szczegóły konwersji są zapisane w
`conversion-provenance.json` obok checkpointu.

## Świadomie wyłączone

- `experimental_modes` — warianty robocze należą do `profiles/dev/`;
- prefix caching;
- CPU offload;
- expert parallelism;
- output-tiled BN8 z runtime v0.31.

## Stan kwalifikacji

Checkpoint przeszedł pełną lokalną weryfikację tożsamości, shardów, hashy i
artefaktów DFlash. Dnia 2026-09-10 runtime v0.29 osiągnął stan `ready` na
8 × R9700 i ukończył minimalną inferencję chat z odpowiedzią `OK`. Stan
działającej usługi sprawdza się osobno poleceniem:

```bash
./run service status
```
