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
./run launcher switch deepseek-v4-flash
./run launcher status
./run launcher logs --follow
./run launcher logs --component litellm --follow
./run launcher stop
```

`start` nie zastępuje aktywnego modelu. `switch` jest jawną operacją zmiany:
najpierw wykonuje bezpieczny stop, a dopiero potem start wybranego profilu.
Do podglądu bez zmian stanu służy `--dry-run` przy `start`, `switch` i `stop`.

Launcher domyślnie uruchamia model, czeka na jego gotowość, uruchamia i testuje
LiteLLM na porcie `4000` oraz aktywuje ustawienia Claude Code z profilu. `stop`
zatrzymuje oba komponenty w odwrotnej kolejności. Jeżeli proxy nie wystartuje
po świeżym starcie modelu, cały start jest wycofywany. Jawne `--runtime-only`
omija LiteLLM i służy wyłącznie do diagnostyki komponentu. Najpierw wykonaj
jednorazowo `./run proxy install` i ustaw `LITELLM_MASTER_KEY` w `.env`.

Ustawienie Claude Code `DISABLE_PROMPT_CACHING=1` wyłącza tylko znaczniki
`cache_control` specyficzne dla zarządzanego API Anthropic. Nie wyłącza
automatic prefix caching vLLM: lokalny silnik wykrywa identyczne tokenowe
prefiksy niezależnie od tych znaczników. Dzięki temu Claude Code nie oczekuje
anthropicowych pól rozliczeniowych cache, a runtime nadal korzysta z cache
GPU/RAM/filesystem.

Niskopoziomowe skrypty komponentów są dostępne do jawnej diagnostyki. Zwykłe
operacje produkcyjne powinny przechodzić przez launcher.

Najpierw upewnij się, że inny runtime nie jest aktywny:

```bash
./run service status
systemctl --user is-active r9700-runtime.service
```

Następnie:

```bash
./run launcher start qwen38-flash --runtime-only
```

Bezpieczne zatrzymanie:

```bash
./run launcher stop --runtime-only
```

Zwykły start GLM uruchamia produkcyjny profil v0.29 ROCm 10 z MRV2, TP8 bez
Expert Parallel, packed MXFP4 decode GEMV, DFlash2 K4, FP8 KV, kontekstem
768K i Vision. Profil UNCENSORED używa tego samego runtime z osobnym,
zweryfikowanym checkpointem:

```bash
./run launcher start glm53-flash
./run launcher start glm53-flash-uncensored
```

Live throughput podczas pracy Claude Code można odczytywać pasywnie,
bez wysyłania dodatkowego promptu. Dla całego `qwen-multi`, łącznie z
przypisaniem statystyk `amd-smi` do głównego TP4 i każdej repliki workera:

```bash
scripts/watch-qwen-live.sh
```

Domyślnie wrapper odświeża jedną tabelę na ekranie zamiast dopisywać kolejne
wiersze. `MAIN` oznacza główny model TP4, a `W0`--`W3` cztery osobne repliki
TP1. Tabela pokazuje fazę, kolejkę, KV, kroczącą szybkość decode, APC,
speculative acceptance oraz GFX/zegar/moc. `KV TOK/S~` jest tylko
wygładzoną estymatą przyrostu tokenów KV; przy cache hitach, zwalnianiu bloków
i przejściu do decode może być skokowe lub zerowe. Miarodajny wynik pojawia się
w sekcji `OSTATNIE ZAKOŃCZONE REQUESTY`: prefill liczy wszystkie tokeny
promptu, `uncached` odejmuje trafienia APC, a decode i TTFT pochodzą z delt
liczników vLLM dla konkretnego silnika; wyświetlany jest też czas E2E. Wyjście
z dashboardu: `Ctrl-C`.

Widok pojedynczego komponentu, zapis JSONL i ograniczona liczba próbek:

```bash
.venv/bin/python scripts/watch-claude-throughput.py --target main --gpu
.venv/bin/python scripts/watch-claude-throughput.py --target workers --gpu
.venv/bin/python scripts/watch-claude-throughput.py --target all --gpu --dashboard
.venv/bin/python scripts/watch-claude-throughput.py \
  --target all --json-lines --samples 60 > logs/qwen-live.jsonl
```

Domyślne okno `DEC~` i `KV TOK/S~` ma 10 sekund i można je zmienić przez
`--window`. Monitor obejmuje także ruch przechodzący przez LiteLLM. Jeżeli
zostanie dołączony w połowie requestu, oznaczy pierwszy wynik jako niepełny;
następne requesty będą liczone od początku.

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
