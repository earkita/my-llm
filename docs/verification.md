# Dowody i ograniczenia

Profile zostały skopiowane wyłącznie z katalogu `production` źródłowego repo.
Maszynowo czytelne pochodzenie zapisano w `provenance.json`.

## DeepSeek V4 Flash

Zweryfikowany profil vLLM 0.28 TP1/PP6 używa sześciu R9700, partycji
`7,8,8,8,8,4`, DSpark K5 i 1,280 MiB KV na GPU. Test graniczny wykonał
1,044,480 tokenów promptu oraz 4,096 tokenów wyjścia: TTFT 562.05 s,
prefill 1,858.35 tok/s i decode 24.48 tok/s.

## GLM-5.3-Flash

Target to sześć shardów UD-Q4_K_XL, łącznie 199,707,321,347 bajtów. DFlash2
BF16 ma 2,352,022,432 bajty i przypięty SHA-256. Zmierzone przebiegi DFlash2:

| Prompt + output | Prefill | Decode |
|---:|---:|---:|
| 4,096 + 128 | 449.35 tok/s | 54.83 tok/s |
| 65,536 + 32 | 126.23 tok/s | 24.33 tok/s |
| 131,072 + 32 | 80.43 tok/s | 17.80 tok/s |

Limit 1,048,576 pochodzi z modelu i konfiguracji; nie jest dowodem pełnego
przebiegu na granicy 1M. Źródłowy host miał również nowe, korygowalne zdarzenia
PCIe AER podczas osobnej bramki lifecycle. Model/API działały, ale pełna
kwalifikacja sprzętowa wymaga czystego powtórzenia RAS.

## Qwen3.8 Flash-Next

Profil vLLM 0.28 TP8/EP8 MTP K2 wykonał dokładnie
`261,120 + 1,024 = 262,144` tokenów. Zmierzył 1,239.33 tok/s cold prefill,
29.28 tok/s decode, TTFT 210.69 s i E2E 245.64 s. Identyczny replay obniżył
TTFT do 2.74 s. Przy 64K MTP2 zaakceptował 696 z 700 draftów i osiągnął
29.72 tok/s wobec 11.70 tok/s bez spekulacji.

## Zakres walidacji nowego repo

Podczas ekstrakcji nie uruchamiano GPU. Wykonano walidację JSON, zgodności
model/runtime, kompletności i hashy wszystkich zachowanych patchy, dry-run
każdej recepty oraz testy kontrolera. Dane wydajności są przeniesionymi
dowodami z identycznych profili źródłowych, nie nowym benchmarkiem.
