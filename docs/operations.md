# Operacje

## Inspekcja i instalacja

```bash
./run profiles list
./run config profile deepseek-v4-flash
./run install --profile deepseek-v4-flash --dry-run
./run install --profile deepseek-v4-flash
```

Analogicznie użyj `glm53-flash` albo `qwen38-flash`. Dry-run sprawdza hashe
constraints i patchy bez pobierania źródeł.

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

`glm53-flash` przypina target Quark/MXFP4 oraz domyślny drafter DFlash2 K7.
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

Zwykły start GLM uruchamia zweryfikowany profil MRV2, TP8/EP8, DFlash2 K7,
BF16 KV i 256K. `dflash2` jest tylko zgodnościowym aliasem tego samego runtime.
Target-only i K1 pozostają jawnymi trybami kontrolnymi:

```bash
./run launcher start glm53-flash
./run launcher start glm53-flash --runtime-mode dflash2
./run launcher start glm53-flash --runtime-mode target-only-32k
./run launcher start glm53-flash --runtime-mode dflash2-k1
./run launcher start glm53-flash --runtime-mode extract-hidden-states-k1
```

Po testach zatrzymaj usługę przed zmianą trybu. K7 przeszedł bramkę API i pełny
test graniczny `262016 + 128 = 262144`, włącznie z poprawnym decode i 111/111
zaakceptowanymi draftami. Trybu `long-context-400k-dflash2` nie używaj do
serwowania: ostatni test graniczny, wykonany przed `0021`/`0022`, zakończył
się nielegalnym dostępem GPU i nie został jeszcze powtórzony na finalnym obrazie.

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
