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
| Źródło | `dealignai/GLM-5.3-Flash-UNCENSORED-FP8` @ `d21b19569d30e6f471c433b11e672b3bbb80552a` |
| Runtime | vLLM na ROCm 10 |
| Nazwa runtime’u | `glm53-flash-uncensored-quark-mxfp4-rocm10-mrv2-8xr9700-tp8-noep-mxfp4gemv-dflash2-k4-fp8kv-512k-vision-weights` |
| Recepta | `vllm_glm53flashrocm10_v0.29` |
| GPU | 8 × R9700 |
| Równoległość | TP8 / PP1 / DP1, bez expert parallelism |
| Maksymalny kontekst | 524288 tokenów (512K) |
| Maksymalna liczba sekwencji | 1 |
| Budżet batchowanego prefilla | 4096 tokenów |
| Wagi | lokalna konwersja AMD Quark MXFP4 |
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
  `--enable-prefix-caching`; API raportuje
`usage.prompt_tokens_details.cached_tokens`, a `/metrics` udostępnia liczniki
  `vllm:prefix_cache_queries_total` i `vllm:prefix_cache_hits_total`;
- `--prefix-cache-retention-interval 1280` zachowuje checkpoint na każdej
  wspólnej stronie cache; domyślne `0` nie dawało trafień w układzie
  Mamba `16` + MLA/DFlash `1280`;
- szablon Claude Code wymusza `"totalTokensReminder": "off"` na głównym
  poziomie ustawień, ponieważ zmienny licznik w system prompt przerywał
  identyczny prefiks po 34 560 tokenach; ustawienie obowiązuje po uruchomieniu
  nowej sesji Claude Code;
- limit kontekstu `524288` zapewnia zapas względem pojemności hybrydowego cache;
  zachowuje dotychczasową rezerwę KV `4960000000`;
- warmup renderera Vision jest ograniczony do budżetu prefilla `4096`, stare
  stany Mamba są zwalniane także przez luki `null`, a wznowienie po trafieniu
  APC indeksuje tabelę w jednostkach `mamba_block_size`;
- checkpoint jest ładowany z 62 shardów Safetensors;
- runtime wymaga 29 uporządkowanych patchy v0.29.

## Pochodzenie i weryfikacja modelu

Checkpoint powstał przez lokalną konwersję wag FP8 do Quark MXFP4. Walidacja
offline objęła 76108 tensorów i nie wykazała braków ani różnic w nazwach,
kształtach lub typach. Hash 3280 tensorów i skal zachowanych bez konwersji był
zgodny ze źródłem. Cztery reprezentatywne próbki po dekwantyzacji osiągnęły
cosine similarity od 0.99284 do 0.99336. Szczegóły konwersji są zapisane w
`conversion-provenance.json` obok checkpointu. Konwersja używa rewizji źródła
`d21b19569d30e6f471c433b11e672b3bbb80552a`, zawierającej poprawione wagi,
`clear_thinking=true` w szablonie rozmowy i `repetition_penalty=1.1` w
`generation_config.json`.

## Świadomie wyłączone

- `experimental_modes` — warianty robocze należą do `profiles/dev/`;
- LMCache oraz CPU/NVMe KV offload;
- expert parallelism;
- output-tiled BN8 z runtime v0.31.

## Stan kwalifikacji

Checkpoint przeszedł pełną lokalną weryfikację tożsamości, shardów, hashy i
artefaktów DFlash. Dnia 2026-09-10 runtime v0.29 osiągnął stan `ready` na
8 × R9700 przy kontekście 524288 tokenów. Trzy żądania po 10241 tokenów ze
wspólnym prefiksem 10240 tokenów zwróciły `cached_tokens`: 0, 8960 i 8960.
TTFT wyniósł odpowiednio 16,404 s, 1,638 s i 1,401 s, a liczniki Prometheusa
potwierdziły 17920 trafionych tokenów. Log nie zawiera błędów MLA/DSA,
wyrównania bloków, DFlash/MTP ani FP8 KV. Stan
działającej usługi sprawdza się osobno poleceniem:

```bash
./run launcher status
```
