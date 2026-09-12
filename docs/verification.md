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

Domyślna konfiguracja to recepta v0.31, MRV2, TP8 bez Expert Parallel, PP1,
output-tiled BN8 RDNA4 MXFP4 decode GEMV, DFlash2 K4 z draft TP8, FP8 KV,
786,432 tokenów, `max_num_batched_tokens=4096`, stała rezerwa KV 4.96 GB,
concurrency 1, bez CPU offload i bez prefix cache. Vision jest aktywne z
wagami encoder sharded przez TP8, `TRITON_ATTN`, limitem ośmiu obrazów i 4096
tokenów obrazu. Target językowy używa `ROCM_AITER_MLA_SPARSE`.

Tryb diagnostyczny `prefix-cache` zachowuje całą tę topologię i przełącza tylko
automatic prefix caching. Został przygotowany bez zmiany aktywnego procesu;
nie jest jeszcze wynikiem kwalifikacji ani domyślną konfiguracją produkcyjną.

Produkcyjnie zakwalifikowany tryb `prefix-cache-offload` rozszerza ten
kandydat o natywny vLLM
`OffloadingConnector`, 32 GiB RAM i ograniczony do 128 GiB tier filesystem z
rezerwą 192 GiB wolnego miejsca. Tryb wyłącza `expandable_segments`, ponieważ
vLLM odrzuca connector z remapowalną pamięcią KV. Granica trybu jest obniżona
do `695040 = 543 × 1280`, czyli jedną pełną stronę poniżej wykrytej pojemności
`696320`, dzięki czemu używa wyłącznie pełnych stron
hybrydowego cache. Rezerwa KV 4,42 GB zwalnia 530 MB względem testu 4,95 GB,
który uruchamiał serwer, lecz zakleszczył workery podczas pierwszego realnego
chunku prefill. Dla ROCm `OffloadingConnector` recepta
wstępnie wykonuje trzy małe BF16 GEMM odpowiadające ścieżkom GLM decode, aby
hipBLASLt załadował potrzebne moduły przed stałą rezerwacją KV. Recepta
podtrzymuje też pracę silnika podczas asynchronicznych lookupów filesystem,
aby żądanie odroczone na secondary tier nie zatrzymało pętli scheduler-a.
Bloki DFlash/EAGLE kończące się przed granicą promptu pozostają zapisywalne
również wtedy, gdy ten sam krok schedulera rozpoczął już decode; zapobiega to
dziurze, która zerowała wynik późniejszego lookupu całego prefiksu. Dodatkowa
retencja jednego poprzedniego, wyrównanego stanu Mamba zachowuje stan na
granicy fallbacku EAGLE; bez niej lookup grup Mamba redukował wspólny wynik do
zera mimo trafienia grup attention.

Kwalifikacja persistent cache 10 września 2026 dała następujące wyniki:

- cold miss `4096 x 16` osiągnął TTFT 3.728 s, prefill 1,098.70 tok/s i
  decode 43.76 tok/s; powtórzenie trafiło 2,560 tokenów lokalnego cache,
  obniżyło TTFT do 1.781 s i podniosło efektywny prefill do 2,299.88 tok/s;
- po łagodnym restarcie bez czyszczenia cache pierwszy identyczny request
  odzyskał 2,560 tokenów z external prefix cache: filesystem trafił 14/14
  fragmentów, odczytał 859.47 MB i przeniósł 303.30 MB CPU->GPU;
- niezależny cold benchmark `4096 x 128`, C1 przeszedł API gate i osiągnął
  TTFT 3.633 s, prefill 1,127.51 tok/s oraz decode 49.26 tok/s;
- trwała usługa LiteLLM przeszła zarówno test OpenAI chat completions, jak i
  natywne endpointy Anthropic `/v1/messages` oraz `/v1/messages/count_tokens`;
  odpowiedź miała poprawny format Claude i lokalne liczenie zwróciło 15
  tokenów wejściowych zgodnie z raportem generacji;
- deterministyczny Vision smoke poprawnie rozpoznał czerwony obraz;
- dokładna granica `694,976 + 64 = 695,040` odnalazła właściwy sekret na 95%
  kontekstu: model przetworzył 694,976 tokenów promptu, wygenerował 38 tokenów
  i zakończył request po 631.36 s;
- w 610 stabilnych próbkach sekundowych granicy średnia aktywność GPU wyniosła
  99.97%, najniższa średnia ośmiu GPU 94.38%, maksymalny hotspot 80 C,
  maksymalna moc 250 W, a minimalny wolny VRAM 266 MiB; nie wystąpił OOM,
  reset GPU, traceback ani zakleszczenie schedulera.

Kwalifikacja 9 września 2026 dała następujące wyniki:

- focused kernel gate przeszedł 16/16, a każdy wariant tiled był zgodny ze
  scalar; w 500 powtórzeniach BN8 osiągnął 1.039x scalar dla up projection i
  2.261x dla down projection;
- deployment-exact A/B `32768 x 128`, C1, dziewięć próbek na wariant dał
  49.820 tok/s średniego decode dla BN8 wobec 49.065 dla scalar (+1.54%) oraz
  minimum 48.790 wobec 48.213 tok/s (+1.20%); średni TTFT zmienił się o
  +0.09%, a E2E poprawił o 0.05%;
- API smoke i deterministyczny Vision smoke przeszły;
- NIAH 256K przeszedł 4/4 położeń igły: 5%, 35%, 65% i 95%;
- dokładna granica `786,368 + 64 = 786,432` zwróciła właściwy sekret; model
  faktycznie przetworzył 786,368 tokenów promptu, wygenerował 18 tokenów i
  zakończył request po 746.09 s;
- 746 próbek telemetrycznych z granicy pokazało minimum 99.84% średniej
  aktywności GPU, maksymalną moc 238 W, co najmniej 302.93 GB dostępnej pamięci
  hosta i minimalny raportowany margines VRAM 20 MiB;
- w całym oknie nie pojawił się nowy MCE, watchdog, OOM, reset GPU ani
  traceback runtime.

Pierwszy izolowany pomiar rzeczywistego żądania Claude Code przez LiteLLM po
promocji zawierał 271,330 tokenów promptu i 375 tokenów wyjścia. Delty
liczników vLLM dały prefill 1,115.11 tok/s, decode 29.77 tok/s, TTFT 243.80 s
i E2E 256.36 s, bez kolejki. Nie jest to bezpośredni komparator testu A/B
32K: przy kontekście 271K większą część decode zajmują attention i odczyty KV.

Samowystarczalny profil `glm53-flash-v029-rollback` zachowuje poprzednią
receptę. Tryb `mxfp4-gemv-dflash2-k4-fp8-768k-vision-weights` jest dokładnym
rollbackiem topologii 768K Vision; pozostałe tryby v0.29 zachowują dawne K3,
K7, 1M i BF16-KV 256K jako jawne komparatory.

## Qwen3.8 Flash-Next

Historyczny profil upstream FP8 vLLM 0.28 TP8/EP8 MTP K2 wykonał dokładnie
`261,120 + 1,024 = 262,144` tokenów. Zmierzył 1,239.33 tok/s cold prefill,
29.28 tok/s decode, TTFT 210.69 s i E2E 245.64 s. Identyczny replay obniżył
TTFT do 2.74 s. Przy 64K MTP2 zaakceptował 696 z 700 draftów i osiągnął
29.72 tok/s wobec 11.70 tok/s bez spekulacji.

Profil `qwen38-flash-uncensored` to lokalnie przekwantyzowany UNCENSORED
MXFP4/FP8 na czterech R9700, TP4/EP4 i MTP2. Na rzeczywistym `gfx1201` przeszedł ładowanie,
chat, reasoning, tool calling oraz dokładny benchmark 256+64: 532.65 tok/s
prefill i 15.43 tok/s decode. Pełny opis pochodzenia tensorów, polecenie
konwersji i ograniczenia kwalifikacji znajdują się w
[raporcie Qwen3.8 UNCENSORED MXFP4/FP8](qwen38-uncensored-mxfp4-fp8.md).
Pełny kontekst 262,144 nie był wykonywany end-to-end na nowym checkpoincie;
wynik starego profilu FP8 nie jest przenoszony na niego automatycznie.

Stos Claude Code dla UNCENSORED przeszedł również kwalifikację 21/21 z
thinking włączonym dla ról głównych. Żądane przez Claude Code `high` zostało
zmienione w LiteLLM na obsługiwane przez Qwen `xhigh`; dziewięć bloków thinking,
dziewięcioturowa naprawa z `Read/Edit/Bash` i wbudowany non-thinking test aliasu
fast przeszły. Oba profile Qwen mają natywny prefix cache z interwałem 800, ale
pełna granica 262144 i pomiar trafień cache dla nowych checkpointów nie były
częścią tego przebiegu.

Profil `qwen38-flash` wskazuje zwykły checkpoint MXFP4/FP8. Jego 132 pliki,
5168 tensorów i 58 968 416 801 elementów zmiennoprzecinkowych przeszły pełny
skan kompletności oraz NaN/Inf. Zawiera 36 skal kalibracyjnych FP8 KV. Ten
wariant nie został jeszcze uruchomiony na `gfx1201`, dlatego nie dziedziczy
kwalifikacji runtime'u UNCENSORED.

## Qwen3.8-27B Quark W4A16: cztery workery R9700

Profil `qwen38-4x27b` uruchomił checkpoint AMD w dokładnej rewizji
`0f7ee2559e8dbc25879e1fe1677b2b10708b91a9` jako cztery niezależne repliki
TP1 na czterech R9700/gfx1201. Historyczny wariant MTP K1 ładował 17.44 GiB
modelu i udostępniał BF16 KV dla 155,316 tokenów. Bieżący DFlash2 K4 ładuje
17.9 GiB targetu i draftera oraz udostępnia FP8 KV dla 250,106 tokenów.
Lokalna konfiguracja ma 127 wyłączeń z kwantyzacji, w tym wszystkie 15 modułów
`mtp.*`; wcześniejsze internetowe raporty o 112 wyłączeniach dotyczyły starszej
wersji checkpointu.

Kwalifikacja 11 września 2026 dotyczyła MTP K1 i wykazała:

- pełny skan 2,191 tensorów i 3,621,110,512 elementów zmiennoprzecinkowych bez
  NaN/Inf oraz 496 kompletnych zestawów Quark INT4 group-128;
- poprawne reasoning i non-thinking, wymuszone wywołanie `get_weather` z
  argumentem `Warszawa`, prefix-cache hit 16,800 tokenów oraz MTP 110/121
  zaakceptowanych draftów w powtarzalnej kwalifikacji funkcji;
- dwanaście krótkich żądań rozłożonych po wszystkich rangach dało średnio
  990.43 tok/s prefill obserwowanego przez klienta i 31.69 tok/s decode na
  pojedynczą replikę; zagregowane wyjście wyniosło 111.18 tok/s, minimum
  pojedynczej karty 30.81 tok/s, a fairness Jaina 0.99947;
- dokładna granica `131008 + 64 = 131072` przeszła bez OOM: TTFT 490.69 s,
  prefill 266.99 tok/s, decode 2.27 tok/s i E2E 518.40 s. Jest to dowód
  poprawności granicy, nie zalecany interaktywny punkt pracy;
- oba aliasy LiteLLM trafiły do backendu workerów. Claude Code 2.1.197 przeszedł
  21/21 kontroli, w tym thinking, fast bez thinking oraz rzeczywistą pętlę
  `Read/Edit/Bash`, naprawę pliku i dwa testy jednostkowe.

Artefakty znajdują się w `logs/validation/qwen38-4x27b-*`. Karta AMD podaje
schemat W4A16 i ustawienia samplera, ale jej polecenie wymaga nieopublikowanej
jeszcze obsługi Quark W4A16 w vLLM. Zewnętrzna recepta jednej R9700 oparta na
AutoRound, FP8 KV i DFlash2 jest punktem odniesienia wydajnościowym, a nie
dowodem dla tego checkpointu AMD ani źródłem jego wag.

12 września 2026 profil workerów został przygotowany do użycia dokładnej
rewizji `4d30ec736ffc6b8688dc2ae2b502d9b48bdec279` draftera
`syvai/Qwen3.8-27B-DFlash2-W4A16`: K4, probabilistyczny sampler,
`TRITON_ATTN` i draft TP1. Pełny skan 153 tensorów draftera nie wykrył
NaN/Inf i potwierdził 36 symetrycznych, skompresowanych modułów W4A16 group-128.
Łatka `0018` usuwa założenie o gęstym `qkv_proj.weight`: wagi context-KV są
materializowane przez wybrany kernel kwantyzacji, więc używają wag i skal
draftera, a nie targetu ani referencji.

Bieżąca pula została przełączona łagodnym restartem wyłącznie worker-poola;
główny Qwen Flash i LiteLLM pozostały `ready`. Wszystkie cztery rangi przeszły
generowanie, reasoning/non-thinking, tool calling, prefix-cache replay i
DFlash2: hit wyniósł po 16,160 tokenów, a akceptacja po 271/390, czyli 69.49%.
Krótki benchmark 256/64 dał 984.85 tok/s prefill, 59.11 tok/s decode średnio na
kartę, minimum 56.31, 185.55 tok/s zagregowanego wyjścia oraz fairness 0.99840.

Czysty prefill 32K osiągnął 438.03 tok/s. Granica
`131008 + 64 = 131072` przeszła bez OOM z 144.01 tok/s prefill, TTFT 909.73 s,
2.43 tok/s decode i E2E 935.61 s. Podczas prefill aktywna R9700 raportowała
100% GFX oraz 6--10% UMC; przy TP1, braku CPU offloadu i lokalnym KV wynik jest
ograniczony long-context attention i kernelami, nie PCIe. Raporty znajdują się
w `logs/validation/qwen38-4x27b-dflash2-fp8-*`. Jest to lokalna kwalifikacja
R9700/gfx1201; wyniki MI350 ani checkpointu AutoRound nie zostały użyte jako
dowód.

Strojenie `max_num_batched_tokens` na identycznym czystym prefill 32K nie
wykazało korzyści z większych chunków. Warianty 2,048 / 4,096 / 8,192 osiągnęły
odpowiednio 438.03 / 431.13 / 410.13 tok/s. Jednocześnie peak aktywacji wzrósł
z 0.25 przez 0.66 do 1.50 GiB, a fizyczny cache zmalał z 250,106 przez 239,464
do 219,777 tokenów. Każdy wariant przeszedł gate funkcjonalny, po czym profil
produkcyjny został przywrócony do zwycięskiego 2,048.

Pierwszy start DFlash2 przy `gpu_memory_utilization=0.92` zatrzymał się przed
gotowością: vLLM raportował 10.24 GiB dostępnego KV wobec 11.07 GiB wymaganych
dla 131,072 tokenów i szacował limit 119,952. Profil zachowuje kontrakt 131,072
i podnosi wykorzystanie czterech dedykowanych kart workerów do 0.96; wynik
ponownego startu i testów jest zapisywany osobno od historycznej kwalifikacji
MTP K1.

Historycznie przy 0.96 alokacja przeszła: 11.53 GiB BF16 KV zapewniło 136,602 tokeny i
concurrency 1.04x. Start ujawnił następnie, że blok hybrydowego schedulera
DFlash2 ma 816 tokenów; profil wyrównuje więc interwał retencji prefix cache z
historycznych 800 do wymaganej pełnej strony. Bieżący wariant ustawia target i
draft na FP8 E4M3. Ponieważ checkpointy nie zawierają skal KV, a przypięty
runtime nie ma dynamicznego `calculate_kv_scales`, vLLM użyje skal 1.0; wariant
został dlatego jawnie objęty kwalifikacją jakości i granicy kontekstu na
gfx1201. Lokalny start zmierzył hybrydowy blok 1616 tokenów, więc retencja
wynosi teraz 1616.
Ta sama alokacja 11.53 GiB daje 250,106 tokenów KV na kartę, concurrency 1.91x
dla 131,072 i 248,864 tokeny po wyrównaniu w dół do 154 pełnych stron.

## Ograniczenia

- Kwalifikacja GLM ROCm 10 v0.31 obejmuje concurrency 1 i pełny kontekst 768K
  z FP8 KV oraz Vision. Granica przeszła, ale 20 MiB raportowanego wolnego VRAM
  oznacza, że zwiększenie rezerwy KV lub dodatkowy stały użytkownik VRAM wymaga
  ponownej kwalifikacji.
- Concurrency > 1 i długi thermal soak nie są zakwalifikowane do produkcyjnego
  serwowania. Diagnostyczny tryb GLM K2/FP8/256K/C4 przeszedł 8 września 2026
  krótkie testy czterech równoległych żądań API oraz wywołań narzędzi przez
  LiteLLM; nie stanowi to kwalifikacji czterech pełnych kontekstów 256K.
- Bazowy wariant 768K nadal nie używa prefix cache. Jawny wariant
  `prefix-cache-offload` jest zakwalifikowany produkcyjnie z granicą 695040;
  natywne AITER FP4 BMM/ASM pozostają wyłączone, ponieważ na gfx1201
  priorytetem jest poprawność.
- Krótkie benchmarki są bramką regresji, a nie pomiarem przepustowości pod
  dużym współbieżnym obciążeniem.
