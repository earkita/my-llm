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

`glm53-flash` wymaga także przypiętego pliku
`/mnt/ai/models/glm/GLM-5.3-Flash-DFlash2-GGUF/GLM-5.3-Flash-DFlash2-BF16.gguf`.
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
./run launcher stop
```

`start` nie zastępuje aktywnego modelu. `switch` jest jawną operacją zmiany:
najpierw wykonuje bezpieczny stop, a dopiero potem start wybranego profilu.
Do podglądu bez zmian stanu służy `--dry-run` przy `start`, `switch` i `stop`.

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

Skrypt startowy wymaga poprawnego limitu mocy i wykonuje host preflight.
Nie zastępuje działającej usługi. Stop nigdy nie eskaluje do SIGKILL.

## Test API i benchmark

Przy działającym profilu:

```bash
skills/measure-r9700-model/scripts/test-and-benchmark.sh \
  --profile qwen38-flash \
  --prompt-tokens 8192 \
  --output-tokens 1024
```

Wyniki trafiają do ignorowanego `logs/`. Test API najpierw sprawdza tożsamość
zarządzanej usługi, `/health`, `/v1/models`, deklarowany kontekst i dokładne
liczniki tokenów.
