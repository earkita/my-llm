# Plan dostrojenia kerneli GLM-5.3-Flash W4A16 na 8x R9700

Status: **etapy 0--1 wykonane 2026-09-16/17; pełne wspierane/przycięte
strojenie decode MoE (384 konfiguracje, 352 poprawne) odrzucone po teście
end-to-end; minimalna recepta v0.32 z DFlash2, Vision i APC została promowana
do `glm53-flash-new` 2026-09-17**. Nie znaleziono wariantu MoE poprawiającego
jednocześnie C1 i C4 o wymagane co najmniej 5%. Szczegóły:
`docs/glm53-v032-moe-e2e-tuning.md`.

## Cel i obecny punkt odniesienia

Celem jest poprawa opóźnienia i przepustowości profilu
`profiles/production/glm53-flash-new.json` bez zmiany checkpointu, limitu
262144 tokenów, poprawności odpowiedzi ani pojemności KV. Najważniejszy jest
decode przy 32K--256K, następnie TTFT długich promptów.

Obecny runtime to vLLM `vllm_glm53_v0.32`, TP8, W4A16, BF16
aktywacje, FP8 KV, sparse MLA przez AITER i DFlash2 K4. Log startowy potwierdza:

- `CompressedTensorsWNA16MoEMethod` i `TritonWNA16Experts`;
- `ROCM_AITER_MLA_SPARSE` dla attention;
- brak dokładnego pliku tuningu MoE
  `E=288,N=256,device_name=AMD_Radeon_R9700,dtype=int4_w4a16.json`, więc vLLM
  używa konfiguracji domyślnej;
- w dotychczasowym audycie sparse MLA uruchamia natywny kernel
  `_rdna4_fp8_paged_mqa_logits_kernel`, a jego ISA zawiera
  `v_wmma_f32_16x16x16_fp8_fp8`.

Pomiar referencyjny, którego nie wolno mieszać z innym endpointem, cache ani
concurrency:

| workload | TTFT | prefill | decode | DFlash acceptance |
| --- | ---: | ---: | ---: | ---: |
| 8K + 256, C1 | 4.802 s srednio | 2608.7 tok/s | 88.3 tok/s | 92.7% |
| 32K + 128, cold | 27.668 s | 1184.3 tok/s | 61.9 tok/s | 76.8% |
| 262016 + 128, cold | 214.085 s | 1223.9 tok/s | 35.7 tok/s | 22.9% |

Szczegóły i artefakty bazowe są w
`docs/glm53-flash-new-benchmarks.md`.

## Zasady eksperymentu

1. Utworzyć płaski profil `profiles/dev/glm53-flash-new-kernel-tuning.json`;
   produkcji nie zmieniać w trakcie strojenia.
2. Zmieniać jedną rzecz naraz i wykonywać co najmniej trzy kompletne przebiegi
   po osobnym warm-upie.
3. Używać vLLM Bench, surowego endpointu completion, greedy sampling, stałego
   seeda i dokładnej liczby tokenów wyjściowych.
4. Prefix cache mierzyć w osobnym teście. Porównania kerneli wykonywać na cold
   promptach, aby trafienie cache nie fałszowało TTFT.
5. Zapisywać profil, rewizję recipe, temperaturę/moc GPU, log runtime, wynik
   benchmarku i trace `rocprofv3` dla każdego kandydata.
6. Nie promować wyniku na podstawie samego mikrobenchmarku. Kernel musi wygrać
   także end-to-end i przejść test jakości/stabilności.

## Decyzja: wyrównanie stron indexera ROCm

Minimalny profil upstream zachowuje patch wyrównujący blok GLM kpool do
`index_kpool * 64`. Dla `index_kpool=4` zmienia to blok schedulera z 1152 na
1280 i utrzymuje natywną stronę paged-MQA równą 64. Retencja prefix cache musi
być wielokrotnością bloku schedulera, dlatego profil używa również wartości
1280.

Kontrolowane A/B target-only, eager, C1, 8192 wejścia + 256 wyjścia dało:

| wariant | TTFT | decode | E2E | pojemność KV |
| --- | ---: | ---: | ---: | ---: |
| blok 1152 | 7.057 s | 11.56 tok/s | 29.12 s | 351737 tokenów |
| blok 1280 | 7.007 s | 11.64 tok/s | 28.91 s | 350341 tokenów |

Nie stwierdzono regresji wydajności; różnice około 1% traktujemy jako szum.
Koszt pojemności wynosi 1396 tokenów (0.40%), a profil nadal raportuje 1.34x
pojemności dla żądania 262144 tokenów. Wariant 1280 został przyjęty jako
warunek bezpieczeństwa przed testami DFlash/MTP. Artefakty A/B znajdują się w
`logs/validation/glm53-indexer-alignment-a-b/`.

## Etap 0: powtarzalny baseline i trace

Przed zmianami powtórzyć cztery workloady:

- 128 input + 512 output: koszt decode bez długiego attention;
- 8192 + 256: typowa sesja agenta;
- 32768 + 256: dłuższa sesja kodowa;
- 131072 + 128 oraz 262016 + 128: długi kontekst i granica profilu.

Dla 32K i 256K zebrać trace `rocprofv3` osobno dla prefill i stabilnego
decode. Raport ma zawierać nazwę kernela, liczbę wywołań, łączny czas, średni
czas, urządzenie/rank oraz udział w całym czasie GPU. Dodatkowo zebrać
wykorzystanie GPU, moc, VRAM i zegary z `amd-smi`.

Z trace należy rozdzielić co najmniej:

- dwa GEMM-y ekspertów W4A16 i ich skalowanie/dequantyzację;
- routing/top-k i permutację tokenów MoE;
- sparse MLA FP8 i operacje KV cache;
- attention draftera DFlash;
- RCCL/all-reduce TP8;
- casty, kopie oraz małe elementwise kerneli między powyższymi operacjami.

Przed optymalizacją potwierdzić, że code objects mają target `gfx1201` i że nie
ma przejścia przez CPU, kodu CUDA/CUTLASS/FlashInfer ani pełnej dekwantyzacji
W4 do BF16 przed GEMM.

## Etap 1: W4A16 MoE -- najwyższy priorytet

To obecnie jedyna ścieżka, dla której runtime sam zgłasza brak dopasowanego
tuningu. Należy wygenerować dokładny plik:

```text
E=288,N=256,device_name=AMD_Radeon_R9700,dtype=int4_w4a16.json
```

Plan:

1. Uruchomić tuner `vllm/benchmarks/kernels/benchmark_moe.py` dla lokalnego
   checkpointu, TP8, `--dtype int4_w4a16` i rzeczywistych kształtów modelu.
2. Strojenie objąć bucketami liczby tokenów występującymi w decode i chunked
   prefill, w szczególności 1--5 oraz potęgi dwójki do 4096. Nie wybierać jednej
   konfiguracji tylko dla dużego prefill.
3. Przeszukać obsługiwane wartości `BLOCK_SIZE_M/N/K`, `GROUP_SIZE_M`,
   `num_warps` i `num_stages`; odrzucać konfiguracje niedokładne lub niestabilne
   przed porównaniem czasu.
4. Upstreamowy tuner importuje `ray`. Nie instalować go doraźnie do
   produkcyjnego venv. Najpierw użyć osobnego, przypiętego środowiska
   narzędziowego albo dodać mały seryjny runner jako patch recipe.
5. Dodać wynik do nowego recipe/patcha, a nie bezpośrednio do wygenerowanego
   drzewa `.runtime`. Po restarcie log nie może już zawierać ostrzeżenia o
   brakującym pliku.
6. Porównać czas samych MoE GEMM oraz pełne workloady 8K, 32K i 256K.

Jeśli najlepsza konfiguracja mikrobenchmarku nie poprawi decode end-to-end,
trace ma rozstrzygnąć, czy ograniczeniem jest routing, synchronizacja TP8 lub
attention. Wtedy pliku nie promować.

Kontrolne A/B na minimalnym runtime target-only z graphami potwierdziło użycie
pliku tuningu w logu, ale nie uzasadniło promocji. Dla 8K stabilna mediana
decode wzrosła tylko o około 2.4% (jedna z trzech próbek była odstającym
wynikiem 20.0 tok/s), a przy 32K średni decode wzrósł z 13.32 do 13.43 tok/s
(0.8%) i E2E poprawił się o 0.7%. Tryb `target-graphs-moe-tuned` pozostaje
diagnostyczny. Artefakty są w `logs/validation/glm53-moe-tuning-a-b/`.

Po wstępnym screeningu wykonano właściwe pełne przeszukanie małego decode:
384 kombinacje `BLOCK_N/K`, warps i waves z ograniczeniami wspieranej ścieżki
ROCm, 352 poprawne konfiguracje, 40 kandydatów powtórzonych na wszystkich
ośmiu GPU oraz 14 finalistów po 500 iteracji per GPU. Najlepszy mikro wariant
`N16/K64/warps1/waves4` przyspieszał M=4 o około 17.5% i M=8 o około 25.4%,
ale w pełnym 8K+256 regresował C1 o 8.0%. Dla C4 dawał +9.3% aggregate, lecz
hybryda pozostawiająca default dla M=1 traciła ten zysk i osiągała tylko
poziom baseline. Parowana kontrola ze wspólnym seedem dała hybrydzie jedynie
+1.9% C4, poniżej progu promocji. Wniosek: graph-captured C1--C4 nadal spędza
większość decode w efektywnym M=1, a dalszy krok wymaga trace rozkładu M i
narzutów całego grafu. Kandydat nie jest promowany; komplet wyników jest w
`docs/glm53-v032-moe-e2e-tuning.md`.

## Etap 2: sparse MLA FP8 i KV cache

Obecna ścieżka już korzysta z natywnego RDNA4 FP8 WMMA, więc jej nie
zastępować bez pomiaru. Należy:

1. Potwierdzić w nowym trace dokładną nazwę kernela i ISA `gfx1201` dla 32K,
   128K i 256K.
2. Sprawdzić dtype wejścia, akumulacji i wyjścia oraz brak FP8 -> BF16/FP32
   przed operacją macierzową. Akumulacja FP32 wewnątrz WMMA jest oczekiwana;
   osobny pełny cast tensora nie jest.
3. Profilować occupancy, VGPR, LDS i bandwidth. Dopiero gdy kernel dominuje w
   decode, stroić rozmiar bloków, liczbę warps i grupowanie stron KV.
4. Oddzielnie zmierzyć append/copy/reshape KV. FP8 cache ma pozostać włączony,
   chyba że trace wykaże koszt konwersji większy niż oszczędność bandwidth.
5. Każdy wariant sprawdzić na granicy 256K oraz testem poprawności sparse
   retrieval; szybszy błędny attention jest niedopuszczalny.

## Etap 3: DFlash zależnie od długości kontekstu

Przy 256K akceptacja K4 spada z ponad 90% do około 23%, dlatego stałe K4 może
nie być najlepsze dla całego zakresu.

Porównać K1, K2, K4 oraz target-only na tych samych promptach przy 8K, 32K,
128K i 256K. Raportować jednocześnie decode tok/s, accepted tokens per draft,
czas draftera, czas verify i czas target modelu. Jeżeli optimum zależy od
długości, dopiero wtedy rozważyć bezpieczną politykę progową; najpierw należy
udowodnić, że vLLM potrafi ją zastosować bez restartu i bez psucia grafów.

## Etap 4: grafy, fuzje i cold start

`FULL_AND_PIECEWISE` jest obecnie wyraźnie szybszy niż eager i
`FULL_DECODE_ONLY`, więc pozostaje baseline'em. Trace powinien jednak wykazać:

Kontrolny test minimalnego runtime target-only po przyjęciu wyrównania 1280
potwierdził tę przewagę również bez DFlash. Dla 8192+256, C1, trzy próbki po
warm-upie:

| wariant | TTFT | prefill | decode | E2E |
| --- | ---: | ---: | ---: | ---: |
| eager | 7.007 s | 1169.2 tok/s | 11.64 tok/s | 28.91 s |
| `FULL_AND_PIECEWISE`, capture `[1]` | 7.027 s | 1165.8 tok/s | 13.35 tok/s | 26.13 s |

Grafy poprawiły decode o 14.6% i E2E o 9.6%, przy zmianie TTFT/prefill o 0.3%.
Capture trwał około 3 s i zużył 0.23--0.29 GiB na GPU. Wynik kwalifikuje tryb
do kolejnego pomiaru 32K. Na 32768+256 graphy poprawiły TTFT o 19.7%, prefill
o 24.5%, decode o 13.1% i E2E o 16.2%. Wynik został więc powtórzony w drugim
workloadzie, a `target-graphs` jest najlepszym trybem minimalnego profilu.
Artefakty są w `logs/validation/glm53-target-graphs-a-b/`.

- czy rozmiary capture 1--5 są rzeczywiście wykorzystywane przez DFlash K4;
- gdzie występują graph breaks i synchronizacje CPU/GPU;
- czy `fuse_norm_quant` i `fuse_act_quant` eliminują osobne casty;
- które pierwsze kształty uruchamiają Triton JIT i czy można je bezpiecznie
  prewarmować przy starcie.

Test prewarm musi rozdzielać czas gotowości serwera od pierwszego TTFT. Nie
zwiększać listy capture sizes w ciemno, bo kosztuje pamięć i start.

## Etap 5: komunikacja TP8

Profil PyNccl był już 6.4% wolniejszy, a custom/AITER all-reduce nie jest
dostępny dla tej konfiguracji, więc komunikacja nie jest pierwszym celem.
Wracamy do niej tylko wtedy, gdy trace pokaże co najmniej 10% czasu decode w
RCCL/all-reduce lub wyraźny straggler jednego ranku.

W takim przypadku porównać bieżący RCCL z alternatywą na identycznym profilu,
zachowując kolejność GPU i ustawienia transportu. Zebrać czasy per rank; wynik
średni może ukrywać wolne łącze lub nierówną synchronizację.

## Kryteria akceptacji i promocji

Kandydat może przejść do produkcji, gdy jednocześnie:

- poprawia medianę decode co najmniej 5% w co najmniej dwóch workloadach, w
  tym 32K lub 256K;
- nie pogarsza prefill ani TTFT o więcej niż 2%;
- nie zmniejsza pojemności KV i nadal przechodzi 262016 + 128;
- zachowuje poprawność odpowiedzi, tool calling i reasoning parser;
- nie powoduje NaN, OOM, preemption ani wyjścia workerów w teście długim;
- wynik powtarza się po zimnym restarcie runtime.

Po kwalifikacji utworzyć nową wersję recipe, dodać patch i hashe do manifestu,
uruchomić `make unit`, `./run install --profile glm53-flash-new --dry-run` oraz
pełny benchmark regresji. Dopiero potem przenieść komplet ustawień do płaskiego
profilu produkcyjnego i uzupełnić `docs/glm53-flash-new-benchmarks.md`.

## Kolejność realizacji

1. Baseline + `rocprofv3` dla 32K i 256K.
2. Dokładny tuning Triton W4A16 MoE dla `E=288,N=256`.
3. A/B end-to-end nowego pliku MoE.
4. Analiza sparse MLA/KV tylko jeśli nadal dominuje w trace.
5. Macierz DFlash K1/K2/K4/target-only zależna od kontekstu.
6. Grafy/JIT, a komunikacja TP8 na końcu i tylko na podstawie trace.

Najbardziej prawdopodobnym pierwszym zyskiem jest usunięcie domyślnej
konfiguracji W4A16 MoE. Największym ograniczeniem przy 256K może pozostać niska
akceptacja DFlash i koszt attention zależny od długości; dlatego obu problemów
nie należy mieszać w jednym eksperymencie.

## Minimalny v0.32 z DFlash2 (17 września 2026)

Do recipe `vllm_glm53_v0.32` dodano mały patch `0008`, który
przenosi brakujące wsparcie GLM/EAGLE3 z upstreamowego PR #56983, zachowuje
osobną geometrię KV draftera, nie dziedziczy EP do gęstego draftera, sharduje
dużą projekcję pomocniczą po TP8 oraz powiększa pierścień K-poola na potrzeby
rollbacku odrzuconych draftów. Tryb `dflash2-k4-graphs` pozostaje diagnostyczny;
`target-graphs` nadal jest wybranym baseline'em.

Start i bramka API przeszły na ośmiu R9700. Runtime rozpoznał
`DFlash2DraftModel`, użył K4, FP8 KV oraz graph capture `[1,2,3,4,5]`.
Pomiary C1, trzy próbki po jednym warm-upie:

| workload | TTFT | prefill | decode | akceptacja |
| --- | ---: | ---: | ---: | ---: |
| 128 + 512 | 209.4 ms | 611.4 tok/s | 60.9 tok/s | 96.7% |
| 8192 + 256 | 5.182 s | 2422.4 tok/s | 50.4 tok/s | 81.6% |

Artefakty znajdują się w
`logs/benchmarks/glm53-flash-upstream-minimal-dflash2-k4-graphs/`. Wynik
potwierdza rzeczywiste draftowanie, ale 8K decode jest około 12% wolniejszy od
wcześniejszego pełnego v0.32 DFlash (57.4 tok/s), więc minimalnego trybu nie
promowano. Następny pomiar powinien rozdzielić koszt wyrównania K-poola 1280,
nowego rollback ring i brakujących poprawek prefix/Mamba z pełnego recipe.

W tym samym trybie włączono następnie vision z `TRITON_ATTN`, TP8
`weights`, limitem 8 obrazów i 4096 tokenów obrazu. Deterministyczny test PNG
32x32 rozpoznał kolor `Red` i wykazał 16 tokenów multimodalnych; artefakt jest
w `logs/validation/glm53-minimal-vision/vision-smoke.json`. Prefix cache także
został sprawdzony rzeczywistym powtórzeniem promptu: pierwszy przebieg utworzył
1280 tokenów cache, a drugi zgłosił dokładnie 1280 `cached_tokens`. DFlash2 nie
obsługuje zewnętrznych embeddingów multimodalnych, dlatego dla promptu z
obrazem drafter dostaje wejście tekstowe, a pełny target weryfikuje wynik z
obrazem; poprawność smoke testu została zachowana, ale wydajność i akceptacja
vision wymagają osobnego pomiaru.
