# Dowody i ograniczenia

Maszynowo czytelne pochodzenie profili i recept zapisano w
`provenance.json`. Artefakty z przebiegów GPU trafiają do ignorowanych
katalogów `.runtime/` i `logs/`.

## DeepSeek V4 Flash

Zweryfikowany profil vLLM 0.28 TP1/PP6 używa sześciu R9700, partycji
`7,8,8,8,8,4`, DSpark K5 i 1,280 MiB KV na GPU. Test graniczny wykonał
1,044,480 tokenów promptu oraz 4,096 tokenów wyjścia: TTFT 562.05 s,
prefill 1,858.35 tok/s i decode 24.48 tok/s.

## GLM-5.3-Flash Quark/MXFP4 na ROCm 10

Profil produkcyjny `glm53-flash` przypina:

- target `amd/GLM-5.3-Flash-Quark-MXFP4` w rewizji
  `b5688f25491202978c19c4d036eef579f61bbe07`;
- DFlash2 `incoai/GLM-5.3-Flash-DFlash2` w rewizji
  `bf582e4eacc1810f76656d1811693ff6c6737d2a`;
- vLLM `main` `7fbd44cbe0a90b9c8fd3a94a0f0401ac4b1bc719`;
- receptę ROCm 10 z PyTorch `2.13.0+rocm10.0.0` i AITER v0.1.21
  `7ff5155f3ba772e534b6cf8dddc0099932327b9b`.

Domyślna konfiguracja to MRV2, TP8 bez Expert Parallel, PP1, packed RDNA4
MXFP4 decode GEMV, DFlash2 K4 z draft TP8, FP8 KV, 1,048,576 tokenów,
`gpu_memory_utilization=0.995`, concurrency 1, bez CPU offload i bez prefix
cache. Target używa `ROCM_AITER_MLA_SPARSE`, a niekauzalny draft
`TRITON_ATTN`.

Reprezentatywny test dziewięciu krótkich przebiegów decode zmierzył dla K4
29.28 tok/s wobec 27.03 tok/s dla K3. Dokładna próba graniczna wykonana przed
promocją K4, z DFlash2 K7 i tą samą geometrią FP8 KV, wykonała
`1,048,560 + 16 = 1,048,576` tokenów przy około 607 tok/s obserwowanego
prefill i 28.31 tok/s decode. NIAH na pełnym kontekście przeszedł 4/4 położeń
igły: 5%, 35%, 65% i 95%. W tym przebiegu nie odnotowano ECC, AER ani OOM.

Jawne tryby alternatywne zachowują K3 i K7 jako rollback. Tryb
`mxfp4-gemv-dflash2-k7-256k` używa BF16 KV oraz limitu 262,144 tokenów.

## Qwen3.8 Flash-Next

Profil vLLM 0.28 TP8/EP8 MTP K2 wykonał dokładnie
`261,120 + 1,024 = 262,144` tokenów. Zmierzył 1,239.33 tok/s cold prefill,
29.28 tok/s decode, TTFT 210.69 s i E2E 245.64 s. Identyczny replay obniżył
TTFT do 2.74 s. Przy 64K MTP2 zaakceptował 696 z 700 draftów i osiągnął
29.72 tok/s wobec 11.70 tok/s bez spekulacji.

## Ograniczenia

- Kwalifikacja GLM ROCm 10 obejmuje concurrency 1 i pełny kontekst 1M z FP8 KV.
- Concurrency > 1 i długi thermal soak nie są zakwalifikowane do produkcyjnego
  serwowania. Diagnostyczny tryb GLM K2/FP8/256K/C4 przeszedł 8 września 2026
  krótkie testy czterech równoległych żądań API oraz wywołań narzędzi przez
  LiteLLM; nie stanowi to kwalifikacji czterech pełnych kontekstów 256K.
- Prefix cache oraz natywne AITER FP4 BMM/ASM pozostają wyłączone; na gfx1201
  priorytetem jest poprawność.
- Krótkie benchmarki są bramką regresji, a nie pomiarem przepustowości pod
  dużym współbieżnym obciążeniem.
