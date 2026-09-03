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

Aktualny target to `amd/GLM-5.3-Flash-Quark-MXFP4`: 62 shardy safetensors i
185,066,521,464 bajty tensorów. Działa na ośmiu `gfx1201` przez vLLM
`glm-release` z MRV2, TP8/EP8, BF16 KV i kontekstem kwalifikacyjnym 32K.
Target-only przeszedł load, użycie wszystkich GPU, prefill, decode i sześć
bramek API; odpowiedź chat była spójna.

Szybki benchmark `256 + 64`, concurrency 1, dwa pomiary po jednym warm-upie:

| TTFT | Prefill | Decode | E2E |
|---:|---:|---:|---:|
| 0.574 s | 445.74 tok/s | 3.74 tok/s | 17.40 s |

DFlash2 jest przypięty rewizją i SHA-256, ale pozostaje trybem
diagnostycznym. Minimalny obraz bliski PR #53906 dał:

| Tryb | Zgodny początek z targetem | Acceptance | Wynik |
|---|---:|---:|---|
| K=1 | 6 tokenów | 1/62 | rozjazd po rollbacku |
| K=7 | 3 tokeny | 2/427 | degradacja verify i powtarzany token |

Patch #54163 dotyczący cache między turami pogarszał ten układ: K=1
rozjeżdżał się już na tokenie zero. Został usunięty, ponieważ prefix cache jest
wyłączony i nie jest potrzebny do kwalifikacji. Próba przeniesienia poprawki
FP32 causal-conv z #52905 również nie poprawiła acceptance i została wycofana.
Pozostały symptom odpowiada niezależnym błędom DFlash/MRV2 oraz GLM
multi-token target verification; produkcja używa wyłącznie target-only.

## Qwen3.8 Flash-Next

Profil vLLM 0.28 TP8/EP8 MTP K2 wykonał dokładnie
`261,120 + 1,024 = 262,144` tokenów. Zmierzył 1,239.33 tok/s cold prefill,
29.28 tok/s decode, TTFT 210.69 s i E2E 245.64 s. Identyczny replay obniżył
TTFT do 2.74 s. Przy 64K MTP2 zaakceptował 696 z 700 draftów i osiągnął
29.72 tok/s wobec 11.70 tok/s bez spekulacji.

## Ograniczenia

- Szybkie benchmarki 256+64 są testem regresji, nie testem przepustowości pod
  współbieżnym obciążeniem.
- GLM został zakwalifikowany przy 32K; pełny limit checkpointu nie był testowany.
- DFlash K1/K7 nie przechodzi bramki lossless i nie jest uruchamiany domyślnie.
- Nie wykonano strojenia FP4BMM; na `gfx1201` stabilność i jakość mają
  pierwszeństwo przed wydajnością.
