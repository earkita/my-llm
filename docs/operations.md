# Operacje

## Inspekcja i instalacja

```bash
./run profiles list
./run config profile deepseek-v4-flash
./run install --profile deepseek-v4-flash --dry-run
./run install --profile deepseek-v4-flash
```

Analogicznie użyj `glm53-flash` albo `qwen38-flash`.
Dry-run sprawdza hashe constraints i patchy bez pobierania źródeł.

Przed instalacją proxy ustaw niepusty, losowy `LITELLM_MASTER_KEY` w lokalnym
`.env`. Brak klucza kończy start błędem; repo nie zawiera wspólnego sekretu
domyślnego.

## Wagi

```bash
./run model download qwen38-flash
./run model verify qwen38-flash
```

Dla istniejących wag:

```bash
./run model adopt qwen38-flash \
  --directory /mnt/ai/models/qwen/Qwen3.8-Flash-Next-FP8
./run model verify qwen38-flash
```

`glm53-flash` przypina target Quark/MXFP4 oraz domyślny drafter DFlash2 K4.
Plikiem draftera jest
`/mnt/ai/models/glm/GLM-5.3-Flash-DFlash2-HF-bf582e4/model.safetensors`.
Download pobiera go automatycznie, a adopt i start wymagają poprawnego rozmiaru
oraz SHA-256.

## Start i stop

Najprościej użyć launchera:

```bash
./run launcher                         # interaktywne menu
./run launcher list
./run launcher start qwen38-flash
./run launcher start qwen38-flash --with-litellm
./run launcher switch deepseek-v4-flash --with-litellm
./run launcher status
./run launcher logs --follow
./run launcher logs --component litellm --follow
./run launcher stop --with-litellm
```

`start` nie zastępuje aktywnego modelu. `switch` jest jawną operacją zmiany:
najpierw wykonuje bezpieczny stop, a dopiero potem start wybranego profilu.
Do podglądu bez zmian stanu służy `--dry-run` przy `start`, `switch` i `stop`.

Bez flagi launcher wystawia bezpośrednie API runtime na porcie `8000`.
`--with-litellm` uruchamia model, czeka na jego gotowość, uruchamia i testuje
LiteLLM na porcie `4000` oraz aktywuje ustawienia Claude Code z profilu. Jeżeli
proxy nie wystartuje po świeżym starcie modelu, cały start jest wycofywany.
Najpierw wykonaj jednorazowo `./run proxy install` i ustaw
`LITELLM_MASTER_KEY` w `.env`.

Ustawienie Claude Code `DISABLE_PROMPT_CACHING=1` wyłącza tylko znaczniki
`cache_control` specyficzne dla zarządzanego API Anthropic. Nie wyłącza
automatic prefix caching vLLM: lokalny silnik wykrywa identyczne tokenowe
prefiksy niezależnie od tych znaczników. Dzięki temu Claude Code nie oczekuje
anthropicowych pól rozliczeniowych cache, a runtime nadal korzysta z cache
GPU/RAM/filesystem.

Launcher używa poniższych skryptów cyklu życia. Można je nadal wywołać
bezpośrednio:

Najpierw upewnij się, że inny runtime nie jest aktywny:

```bash
./run service status
systemctl --user is-active r9700-runtime.service
```

Następnie:

```bash
skills/start-r9700-runtime/scripts/start-runtime.sh \
  --profile qwen38-flash
```

Bezpieczne zatrzymanie:

```bash
skills/stop-r9700-runtime/scripts/stop-runtime.sh
```

Zwykły start GLM uruchamia produkcyjny profil v0.31 ROCm 10 z MRV2, TP8 bez
Expert Parallel, output-tiled BN8 MXFP4 decode GEMV, DFlash2 K4, FP8 KV,
kontekstem 768K i Vision. Topology-exact rollback zachowuje v0.29, scalar
GEMV, DFlash2 K4, FP8 KV oraz 768K Vision:

```bash
./run launcher start glm53-flash
./run launcher start glm53-flash-v029-rollback \
  --runtime-mode mxfp4-gemv-dflash2-k4-fp8-768k-vision-weights
```

Tryb `prefix-cache` jest przygotowany do deployment-exact kwalifikacji, ale nie
jest jeszcze domyślną konfiguracją produkcyjną. Zachowuje BN8, DFlash2 K4,
FP8 KV, 768K i Vision, zmieniając wyłącznie automatic prefix caching. Jego
aktywacja wymaga kontrolowanej zmiany procesu:

```bash
./run launcher switch glm53-flash --runtime-mode prefix-cache --with-litellm
```

Nie wykonuj tej komendy tylko po to, aby sprawdzić konfigurację. Bez zatrzymania
działającego modelu użyj `--dry-run`; pełna kwalifikacja musi porównać cold miss,
warm hit tego samego prefiksu oraz poprawność odpowiedzi.

### Trwały prefix cache przez OffloadingConnector

Tryb `prefix-cache-offload` zachowuje konfigurację modelu, DFlash i Vision.
Używa granicy `695040 = 543 × 1280` tokenów, aby dokładnie wyrównać kontekst
do hybrydowej strony cache i pozostawić roboczy margines VRAM dla realnego
prefillu oraz modułów HIP przy stałych adresach pamięci.
Dodaje automatic prefix caching oraz natywny vLLM `OffloadingConnector` z
`TieringOffloadingSpec`:

- 32 GiB współdzielonego cache RAM w `/dev/shm`;
- filesystem tier w `/mnt/ai/r9700-kv-cache/glm53-flash-v031`;
- limit 128 GiB pełnych bloków KV i rezerwę 192 GiB wolnego miejsca;
- stabilne adresy pamięci KV przez `expandable_segments:False`, wymagane przez
  walidację connectora;
- wstępne załadowanie modułów hipBLASLt dla trzech rodzin BF16 GEMM używanych
  przez GLM decode, wykonane przed stałą rezerwacją KV i ograniczone wyłącznie
  do ROCm `OffloadingConnector`;
- rezerwę GPU KV `4,420,000,000` B dla wyrównanej granicy 543 pełnych stron po
  1280 tokenów; względem nieudanego testu 4,95 GB zwalnia to 530 MB VRAM dla
  prefillu, stagingu offloadu i modułów hipBLASLt;
- podtrzymanie kroków silnika podczas asynchronicznego sprawdzania tieru
  filesystem, także wtedy, gdy wszystkie żądania czekają na wynik lookupu;
- zachowanie pełnych bloków DFlash/EAGLE należących jeszcze wyłącznie do
  promptu, gdy jeden krok schedulera kończy prefill i rozpoczyna decode;
- zachowanie poprzedniego wyrównanego stanu Mamba na granicy fallbacku EAGLE,
  dzięki czemu wszystkie grupy hybrydowego cache uzgadniają wspólny trafiony
  prefiks;
- usuwanie najstarszych zapisanych bloków przed kolejnym zapisem, z ochroną
  bloków używanych przez aktywne transfery.

Przygotowanie katalogu i inspekcja nie zmieniają działającego modelu:

```bash
./run cache prepare --profile glm53-flash --runtime-mode prefix-cache-offload
./run cache status --profile glm53-flash --runtime-mode prefix-cache-offload
./run launcher switch glm53-flash --runtime-mode prefix-cache-offload \
  --with-litellm --dry-run
```

Późniejsze usunięcie trwałej warstwy wykonuje:

```bash
./run cache clear --profile glm53-flash --runtime-mode prefix-cache-offload
```

`cache clear` usuwa tylko warstwę filesystem. Odmawia działania, gdy dokładnie
ten tryb jest aktywny; cache GPU/RAM żywego procesu pozostaje bez zmian. Pełne
wyczyszczenie wszystkich poziomów wymaga łagodnego zatrzymania runtime, użycia
`cache clear`, a następnie ponownego startu. Katalog jest przygotowywany także
automatycznie podczas startu tego trybu.

Natywny XFS `/mnt/ai` jest zamontowany z `noquota`, dlatego limit jest
egzekwowany przez pojedynczego aktywnego writera tieru, a nie przez quota
filesystemu. Nie uruchamiaj równolegle drugiego procesu zapisującego do tego
samego katalogu. Tryb przeszedł testy cold/warm, odzyskanie prefiksu po
restarcie, regresję decode, Vision oraz dokładną granicę 695040 tokenów i jest
kwalifikowany produkcyjnie. Ponieważ pozostaje jawnym wariantem profilu,
uruchamiaj go z nazwą trybu:

```bash
./run launcher start glm53-flash --runtime-mode prefix-cache-offload \
  --with-litellm
```

Po testach zatrzymaj usługę przed zmianą profilu. Bazowa kwalifikacja v0.31
objęła Vision smoke, NIAH 256K 4/4 i dokładny test graniczny
`786,368 + 64 = 786,432`; produkcyjny wariant persistent-cache ma osobną
granicę `694,976 + 64 = 695,040`.

Live throughput podczas pracy Claude Code można odczytywać bezpośrednio z
metryk silnika vLLM:

```bash
.venv/bin/python scripts/watch-claude-throughput.py
```

Wiersz `LIVE` jest próbkowany co sekundę. Pokazuje liczbę aktywnych i
oczekujących żądań, zajęcie KV oraz wygładzoną estymatę wzrostu KV w oknie
10 sekund; ze względu na chunked prefill wartość chwilowa jest skokowa. Po
zakończeniu wiersz `COMPLETE` wylicza z delt liczników dokładne server-side
liczby tokenów, prefill tok/s, decode tok/s, TTFT i E2E. Gdy żądania się
nakładają, wynik `COMPLETE` jest agregatem tych żądań. Pomiar obejmuje również
ruch przechodzący przez LiteLLM.

Na aktualnej maszynie monitor działa także jako odczytowa usługa użytkownika:

```bash
journalctl --user -fu r9700-claude-throughput-monitor.service
```

Skrypt startowy wymaga PPT0 najwyżej 285 W na wszystkich widocznych GPU i
wykonuje host preflight. Niższy limit przechodzi kontrolę. Nie zastępuje
działającej usługi. Stop nigdy nie eskaluje do SIGKILL.

## Test API i benchmark

Przy działającym profilu:

```bash
skills/measure-r9700-model/scripts/test-and-benchmark.sh \
  --profile qwen38-flash \
  --prompt-tokens 8192 \
  --output-tokens 1024
```

Dla domyślnego GLM nie podawaj trybu. Pełną granicę skonfigurowanego kontekstu
można sprawdzić bez ręcznego liczenia promptu:

```bash
skills/measure-r9700-model/scripts/test-and-benchmark.sh \
  --profile glm53-flash \
  --output-tokens 128 \
  --full-context \
  --concurrency 1 --repetitions 1 --warmup 0
```

Przy trybie innym niż domyślny dodaj odpowiadające mu `--runtime-mode`; helper
przekaże tę samą tożsamość do bramki API i benchmarku.

Wyniki trafiają do ignorowanego `logs/`. Test API najpierw sprawdza tożsamość
zarządzanej usługi, `/health`, `/v1/models`, deklarowany kontekst i dokładne
liczniki tokenów.

## Diagnostyczny przebieg 400K

400K pozostaje tylko trybem diagnostycznym. Pierwsza próba graniczna zbiegła
się z fatalnym CPU/Data-Fabric MCE i resetem hosta. Powtórka z potwierdzonym
limitem 285 W nie zresetowała hosta, ale odtworzyła `illegal memory access` na
GPU; oba przebiegi poprzedzały `0021`/`0022`. Dedykowany zapis telemetryczny
GPU osiągnął maksymalnie 255 W (30 W poniżej limitu),
82°C edge (28°C poniżej progu 110°C), 106°C hotspot (4°C poniżej progu) i
88°C pamięci (20°C poniżej progu 108°C). Nie był to udokumentowany thermal
trip; profil 400K nadal nie ma kwalifikacji stabilności.
