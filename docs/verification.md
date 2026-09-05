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
185,066,521,464 bajty tensorów. Aktualna receptura przypina oficjalny vLLM
`main` `6cbb3c154ef1449d2b3c9131a237f36faa695734`, po scaleniu PR #53906 oraz
poprawek DFlash #54826 i #54374, i wymusza MRV2. Domyślna konfiguracja to
TP8/EP8, PP1, DFlash2 K7 z draft
TP8, BF16 KV, 262,144 tokeny, `gpu_memory_utilization=0.97`, concurrency 1,
bez CPU offload i bez prefix cache. Target używa `ROCM_AITER_MLA_SPARSE`, a
niekauzalny draft `TRITON_ATTN`.

Po migracji nowy obraz `6cbb3c154ef1` przeszedł build, importy, skupione testy
CPU oraz świeże testy GPU/API `target-only-32k`, DFlash2 K1 32K i krótki K7 w
konfiguracji 256K. Wszystkie sześć bramek API i tożsamości przeszło w każdym
trybie, a odpowiedzi były poprawne. Target-only osiągnął 213,07 tok/s prefill i
3,93 tok/s decode; K1 osiągnął 7,10 tok/s decode i zaakceptował 68/70 tokenów;
K7 osiągnął 21,66 tok/s decode i zaakceptował 115/133 tokenów. Nie pojawiły się
nowe błędy AER, MCE ani GPU fault. Pełny wynik graniczny 256K poniżej wykonano
jednak na wcześniejszym pinie `c7e6e36fa93a`; nie należy przypisywać go nowemu
buildowi, dopóki nie przejdzie ponownego testu `262016 + 128`.

Oprócz wcześniejszych poprawek DFlash/kpool bieżący obraz zawiera:

- `0019`: ograniczenie tymczasowego workspace prefill indexera;
- `0020`: wyrównanie ROCm do `index_kpool * 64`, co przy `index_kpool=4`
  prowadzi do 768-tokenowego bloku hybrydowego i 192 stanów kpool;
- `0021`: jawne zgłoszenie backendowi stron kernela 128/256 tokenów, czyli
  32/64 skompresowanych stanów. Dzięki temu 768-tokenowy blok storage zostaje
  poprawnie rozpisany na trzy wirtualne strony po 256 tokenów;
- `0022`: guard z PR #54296, który nie pozwala slot-mappingowi czytać poza
  szerokością block table. To dodatkowa ochrona, a nie zamiennik poprawnej
  geometrii z `0021`;
- `0023`: poprawka z PR #55222, która liczy prefill workspace indexera w
  czterotokenowych stanach kpool zamiast w tokenach modelu;
- `0024`: jawny audyt dtype, skali oraz fizycznych i logicznych alokacji cache;
- `0025`: ograniczony do RDNA4 czytnik sparse MLA FP8 E4M3FN w Tritonie z
  istniejącą skalą vLLM `k_scale`; przypięty AITER nie ma kodu MLA dla gfx1201;
- `0026`: analogiczne przeliczenie decode-logits workspace indexera na stany
  kpool.
- `0027`: podział projekcji auxiliary DFlash po TP8 zamiast pełnej repliki na
  każdym GPU;
- `0028`: etapowa dekwantyzacja wag ekspertów OCP-MX, ograniczająca chwilowy
  pik VRAM w emulacyjnym backendzie MoE.

Źródło wcześniejszych awarii było deterministyczne. Niepodzielona tablica dla
bloku storage 768 i kernela 256 kończyła się około 87,552 tokenów w profilu
256K oraz 136,704 tokenów w profilu 400K. Dlatego 134,000 przechodziło na 400K,
natomiast pełne 256K i 409,000 przekraczały odpowiadający im stary próg.

Po `0021` i `0022` pełna granica domyślnego profilu przeszła:

| Prompt + output | TTFT | E2E | Prefill | Decode | Drafty | Wynik |
|---:|---:|---:|---:|---:|---:|---|
| 262,016 + 128 | 437.137 s | 442.463 s | 599.39 tok/s | 23.84 tok/s | 111/111 | spójna odpowiedź, bez błędu runtime |

Artefakty to
`logs/benchmarks/glm53-flash-long-context-256k-dflash2-c1-262016x128-20260904T122236.json`
oraz
`logs/validation/api-glm53-flash-long-context-256k-dflash2-20260904T122236.json`.
API gate przeszedł 6/6, a usage benchmarku wynosi dokładnie 262,144 tokeny.

Dedykowana telemetria finalnej próby 256K zawiera 1,832 próbki GPU. P95 dla
mocy/edge/hotspot/pamięci wynosiło odpowiednio 234 W, 85°C, 103°C i 94°C;
maksima wyniosły 370 W, 87°C, 109°C i 106°C. Statyczny limit PPT0 był
potwierdzony jako 285 W na wszystkich kartach przed i po teście, mimo ośmiu
chwilowych odczytów mocy powyżej tej wartości. Maksymalny hotspot był tylko
1°C poniżej progu slowdown, a pamięć 2°C poniżej niego, dlatego wynik jest
kwalifikacją pojemności i poprawności, nie długim testem termicznym.

Wcześniejsza kwalifikacja krótkich żądań wykazała:

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

Target-only `long-context-1m-fp8` przeszedł etap alokacji: wszystkie 62 shardy
załadowały się na 8 GPU, MLA miał faktyczny backing FP8, silnik zgłosił
1,187,115 tokenów pojemności przy granicy 1,048,576 i ukończył warm-up. Dla
samego 1M latent MLA maleje z 11.00 GiB/GPU w BF16 do 5.50 GiB/GPU w FP8;
spakowany indexer pozostaje 0.354 GiB/GPU. Mały tail kpool pozostał BF16.

Nie jest to jeszcze kwalifikacja inference. Podczas startu wystąpiło 12
poprawialnych `BadTLP` warstwy Data Link: 11 na torze do logicznego AMD GPU 5 i
1 do GPU 7. Nie było błędu niekorygowalnego, GPU fault, resetu AMDGPU, MCE,
EDAC ani OOM, a zatrzymanie było kontrolowane. API, porównanie jakości z BF16 i
needle-in-haystack pozostają do wykonania po ustabilizowaniu PCIe.

Powtórka przygotowana dla dokładnego żądania `32640 + 128 = 32768` również nie
dotarła do API. Podczas startu przybyło 10 poprawialnych `BadTLP` na torze GPU 5
i jeden na nowym torze GPU 6. Silnik ponownie ukończył alokację i warm-up, po
czym został łagodnie zatrzymany. Wynik bramki inference 32K to „niewykonana”, a
nie „nieudana odpowiedź modelu”.

Pełny PR #55219 nie jest przeniesiony celowo. Profil bierze jego potrzebną
semantykę 12-slotowego ringa z commitu `de63c847`, ale nie szeroki refactor
generic packed layout, który pozostaje draftem, nie ma end-to-end walidacji
GLM/MTP na ROCm i zmienia kod niezwiązany z odtworzonym błędem. Rozszerzenie
backportu ma sens dopiero po wskazaniu brakującej poprawki przez test albo po
ustabilizowaniu finalnego kształtu PR.

DFlash2 K7 jest teraz domyślny, ale kwalifikacja obejmuje concurrency 1 i
granice 256K. Target-only 32K pozostaje fallbackiem. Target-only sam zmienia
hash tokenów między identycznymi seeded restartami na tym stosie EP/emulacji,
więc równość tokenów między restartami jest sygnałem diagnostycznym, a nie
jedyną bramką DFlash.

400K pozostaje diagnostyczne. Pierwsza próba zbiegła się z fatalnym
CPU/Data-Fabric MCE i resetem hosta. Powtórka przy zweryfikowanym limicie 285 W
utrzymała host, ale ponownie wywołała `illegal memory access`. Obie próby były
przed `0021`/`0022`; finalny obraz nie został ponownie przetestowany przy 400K.
Dedykowany strumień telemetrii GPU osiągnął 255 W, 82°C edge, 106°C hotspot i
88°C pamięci, odpowiednio 30 W, 28°C, 4°C i 20°C poniżej skonfigurowanego
limitu/progów krytycznych.
Nie ma dowodu na thermal trip, lecz 400K nie ma kwalifikacji stabilności.

## Qwen3.8 Flash-Next

Profil vLLM 0.28 TP8/EP8 MTP K2 wykonał dokładnie
`261,120 + 1,024 = 262,144` tokenów. Zmierzył 1,239.33 tok/s cold prefill,
29.28 tok/s decode, TTFT 210.69 s i E2E 245.64 s. Identyczny replay obniżył
TTFT do 2.74 s. Przy 64K MTP2 zaakceptował 696 z 700 draftów i osiągnął
29.72 tok/s wobec 11.70 tok/s bez spekulacji.

## Ograniczenia

- Nowy obraz `6cbb3c154ef1` przeszedł krótkie testy target-only, DFlash K1 i
  DFlash K7; jego pełna granica 256K nadal oczekuje ponownego testu.
- Krótkie benchmarki są testem regresji, nie testem przepustowości pod dużym
  współbieżnym obciążeniem.
- Wcześniejszy obraz GLM `c7e6e36` ma zweryfikowaną pełną granicę 256K przy
  concurrency 1; 400K, 512K, 1M oraz concurrency > 1 nie są zakwalifikowane do
  serwowania, a nowy obraz wymaga powtórzenia testu 256K.
- Prefix cache i natywne FP4BMM pozostają wyłączone; na `gfx1201` poprawność ma
  pierwszeństwo przed tuningiem.
- Oficjalny plik testów kernela z `c7e6e36` przechodzi 29/30 wywołań: 19/20
  seedów fuzz oraz wszystkie 10 wariantów deterministycznych. Nasz rozszerzony
  plik przechodzi 35/36; w obu odpada tylko seed 9. `max diff 1` oznacza wartość
  różnicy kodu FP8, nie liczbę bajtów: różnią się dokładnie 2 z 270 336 bajtów
  KV, każdy o jeden sąsiedni kod FP8; skala i tail cache są identyczne. Próba
  1000 seedów znalazła łącznie 5 takich bajtów w 3 seedach, bez różnic skali lub
  taila, a produkcyjne writery prefill i decode były bitowo zgodne dla wszystkich
  751 zapisanych wektorów. Seed 9 odtwarza się również na czystym oficjalnym
  commicie, więc nie pochodzi z naszych patchy ringa ani bounds-checku.
- DFlash2 K7 jest domyślny; `--runtime-mode dflash2` jest zgodnościowym aliasem,
  a `target-only-32k` pozostaje fallbackiem.
