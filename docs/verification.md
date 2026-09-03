# Dowody i ograniczenia

Maszynowo czytelne pochodzenie profili i recept zapisano w
`provenance.json`. Artefakty z przebiegów GPU trafiają wyłącznie do
ignorowanych katalogów `.runtime/` i `logs/`.

## DeepSeek V4 Flash

Zweryfikowany profil vLLM 0.28 TP1/PP6 używa sześciu R9700, partycji
`7,8,8,8,8,4`, DSpark K5 i 1,280 MiB KV na GPU. Test graniczny wykonał
1,044,480 tokenów promptu oraz 4,096 tokenów wyjścia: TTFT 562.05 s,
prefill 1,858.35 tok/s i decode 24.48 tok/s.

## GLM-5.3-Flash Quark/MXFP4

Target `amd/GLM-5.3-Flash-Quark-MXFP4` zawiera 62 shardy safetensors i
185,066,521,464 bajty tensorów. Przetestowany build używa oficjalnego vLLM
`main` `c7e6e36fa93a5b8cb95b74fa96e4abdf2f0be51d`, już po scaleniu PR #53906,
oraz MRV2, TP8/EP8, BF16 KV i limitu 32K. Target-only przeszedł load na ośmiu
`gfx1201`, prefill, decode oraz sześć bramek OpenAI API; odpowiedź była spójna.

DFlash2 K7 jest jawnym trybem eksperymentalnym. Po usunięciu kolizyjnego
overlay KV oraz dodaniu wyrównanego layoutu, wielotokenowego verify Triton,
12-elementowego ringa kpool i walidacji indeksów uzyskano:

| Tryb | Próbki | Acceptance | Średnia długość acceptance | Wynik |
|---|---:|---:|---:|---|
| DFlash2 K1 | 2 | 52/74 (70.3%) | 1.62-1.80 | spójny tekst, API 6/6 |
| DFlash2 K7 | 3 | 139/385 (36.1%) | 3.32-3.71 | spójny tekst, wszystkie pozycje K7 akceptowane, API 6/6 |

Ostatni świeży start K7 dał 46/119 (38.7%) zaakceptowanych draftów i średnią
3.71 tokenu na krok targetu. Benchmark `256 + 128`, concurrency 1, trzy pomiary
po jednym warm-upie:

| TTFT mean/p95 | Prefill | Decode mean/min | E2E mean/p95 |
|---:|---:|---:|---:|
| 0.573/0.576 s | 446.60 tok/s | 23.63/21.65 tok/s | 5.967/6.373 s |

Testy mapowania ringa przeszły 14/14 przypadków CPU i 10/10 wybranych testów
kerneli na `gfx1201`. Osobny test GPU z PR #55201, obejmujący ujemne i dodatnie
indeksy poza zakresem ukończonych pooli, przeszedł 1/1.

Tryb nie jest domyślny: wymagane #55239 i #55201 są nadal otwarte, #55219 jest
draftem, a pełny kontekst, concurrency > 1 i deterministyczna losslessness nie
zostały zakwalifikowane. Target-only sam zmienia hash tokenów między
identycznymi seeded restartami na tym stosie EP/emulacji, więc równość tokenów
między restartami jest sygnałem diagnostycznym, a nie jedyną bramką DFlash.

## Qwen3.8 Flash-Next

Profil vLLM 0.28 TP8/EP8 MTP K2 wykonał dokładnie
`261,120 + 1,024 = 262,144` tokenów. Zmierzył 1,239.33 tok/s cold prefill,
29.28 tok/s decode, TTFT 210.69 s i E2E 245.64 s. Identyczny replay obniżył
TTFT do 2.74 s. Przy 64K MTP2 zaakceptował 696 z 700 draftów i osiągnął
29.72 tok/s wobec 11.70 tok/s bez spekulacji.

## Ograniczenia

- Krótkie benchmarki są testem regresji, nie testem przepustowości pod dużym
  współbieżnym obciążeniem.
- GLM ma skonfigurowane 32K, lecz pełne żądanie 32K z DFlash nie było testowane.
- Prefix cache i natywne FP4BMM pozostają wyłączone; na `gfx1201` poprawność ma
  pierwszeństwo przed tuningiem.
- Pełny fuzz kernela przeszedł 29/30 przypadków; seed 9 nadal różni się o jeden
  bajt w skwantowanym KV. Skupione testy ringa i bounds-checku są zielone, a
  odpowiadającej degradacji E2E nie zaobserwowano.
- DFlash2 K7 wymaga jawnego `--runtime-mode dflash2` do czasu domknięcia
  kwalifikacji i scalenia upstreamowych poprawek.
