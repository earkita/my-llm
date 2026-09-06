# Dowody i ograniczenia

Maszynowo czytelne pochodzenie profili i recept zapisano w
`provenance.json`. Artefakty z przebiegów GPU trafiają do ignorowanych
katalogów `.runtime/` i `logs/`; wybrane raporty kwalifikacyjne ROCm 10 są
wersjonowane pod `profiles/dev/glm53-flash-rocm10/results/`.

## DeepSeek V4 Flash

Zweryfikowany profil vLLM 0.28 TP1/PP6 używa sześciu R9700, partycji
`7,8,8,8,8,4`, DSpark K5 i 1,280 MiB KV na GPU. Test graniczny wykonał
1,044,480 tokenów promptu oraz 4,096 tokenów wyjścia: TTFT 562.05 s,
prefill 1,858.35 tok/s i decode 24.48 tok/s.

## GLM-5.3-Flash Quark/MXFP4 na ROCm 10

Profil produkcyjny `glm53-flash-rocm` przypina:

- target `amd/GLM-5.3-Flash-Quark-MXFP4` w rewizji
  `b5688f25491202978c19c4d036eef579f61bbe07`;
- DFlash2 `incoai/GLM-5.3-Flash-DFlash2` w rewizji
  `bf582e4eacc1810f76656d1811693ff6c6737d2a`;
- vLLM `main` `7fbd44cbe0a90b9c8fd3a94a0f0401ac4b1bc719`;
- receptę ROCm 10 z PyTorch `2.13.0+rocm10.0.0` i AITER v0.1.21
  `7ff5155f3ba772e534b6cf8dddc0099932327b9b`.

Domyślna konfiguracja to MRV2, TP8/EP8, PP1, DFlash2 K7 z draft TP8, BF16
KV, 262,144 tokeny, `gpu_memory_utilization=0.97`, concurrency 1, bez CPU
offload i bez prefix cache. Target używa `ROCM_AITER_MLA_SPARSE`, a
niekauzalny draft `TRITON_ATTN`.

Każdy tryb 32K przeszedł bramki API oraz trzy pomiary 4096 wejścia + 128
wyjścia. DFlash2 K7 osiągnął 868.34 tok/s obserwowanego prefill, 26.213 tok/s
średniego decode oraz 465/504 zaakceptowanych draftów (92.26%).

Pełne próby graniczne domyślnego runtime:

| Granica | Prompt + output | TTFT | Prefill | Decode | Drafty |
|---|---:|---:|---:|---:|---:|
| 32K | 32,640 + 128 | 54.432 s | 599.65 tok/s | 23.24 tok/s | zaliczone |
| 256K | 262,016 + 128 | 431.810 s | 606.78 tok/s | 26.99 tok/s | 111/111 |

Przy 256K silnik zaalokował około 5.8 GiB KV cache na GPU i zgłosił 480,827
tokenów pojemności. Maksymalny próbkowany hotspot wyniósł 91°C. Wszystkie
liczniki ECC pozostały zerowe, a w sprawdzonych logach aplikacji i kernela nie
było OOM, illegal memory access, resetu GPU, błędu HSA/amdgpu, AER ani MCE.

Szczegółowe wersje, hashe i artefakty:

- `profiles/dev/glm53-flash-rocm10/results/qualification-20260905.md`;
- `profiles/dev/glm53-flash-rocm10/results/qualification-256k-20260906.md`;
- `profiles/dev/glm53-flash-rocm10/UPSTREAM_STATUS.md`.

Profil `glm53-flash` zachowuje poprzedni, niezależny deployment ROCm 7.14 w
`profiles/production/glm53-flash.json`. Promocja ROCm 10 go nie zastępuje.

## Qwen3.8 Flash-Next

Profil vLLM 0.28 TP8/EP8 MTP K2 wykonał dokładnie
`261,120 + 1,024 = 262,144` tokenów. Zmierzył 1,239.33 tok/s cold prefill,
29.28 tok/s decode, TTFT 210.69 s i E2E 245.64 s. Identyczny replay obniżył
TTFT do 2.74 s. Przy 64K MTP2 zaakceptował 696 z 700 draftów i osiągnął
29.72 tok/s wobec 11.70 tok/s bez spekulacji.

## Ograniczenia

- Kwalifikacja GLM ROCm 10 obejmuje concurrency 1 i kontekst do 256K.
- Konteksty powyżej 256K, FP8 KV, concurrency > 1 i długi thermal soak nie są
  zakwalifikowane do produkcyjnego serwowania.
- Prefix cache oraz natywne AITER FP4 BMM/ASM pozostają wyłączone; na gfx1201
  priorytetem jest poprawność.
- Krótkie benchmarki są bramką regresji, a nie pomiarem przepustowości pod
  dużym współbieżnym obciążeniem.
