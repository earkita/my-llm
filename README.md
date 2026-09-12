# my-llm

Minimalne repozytorium uruchomieniowe dla zweryfikowanych modeli na
AMD Radeon AI PRO R9700. Nie zależy od innych checkoutów: receptury,
środowiska `.runtime`, stan usług i logi są lokalne. Checkpointy pozostają na
dedykowanym magazynie modeli wskazanym przez profile produkcyjne.

## Profile

| Profil | Backend | GPU | Równoległość | Kontekst | Spekulacja |
|---|---|---:|---|---:|---|
| `deepseek-v4-flash` | vLLM 0.28 | 6 | TP1/PP6 | 1,048,576 | DSpark K5 |
| `glm53-flash` | vLLM `main` `7fbd44c`, ROCm 10 | 8 | TP8/no-EP | 786,432 | packed MXFP4 GEMV + DFlash2 K4, FP8 KV |
| `glm53-flash-uncensored` | vLLM `main` `7fbd44c`, ROCm 10 | 8 | TP8/no-EP | 786,432 | packed MXFP4 GEMV + DFlash2 K4, FP8 KV |
| `qwen38-flash` | vLLM 0.28 | 4 | TP4/EP4 | 262,144 | MTP K2, FP8 KV |
| `qwen38-flash-uncensored` | vLLM 0.28 | 4 | TP4/EP4 | 262,144 | MTP K2, BF16 KV |
| `qwen38-4x27b` | vLLM 0.28 | 4 | DP4/TP1 | 131,072 | Quark W4A16 + DFlash2 K4, FP8 KV |
| `qwen-multi` | vLLM 0.28 | 8 | TP4/EP4 + DP4/TP1 | 262,144 + 131,072 | Qwen Flash main + 4 DFlash2 workers |

Każdy deployment jest jednym plikiem w `profiles/production/`. Plik zawiera
pin checkpointu, kompletną konfigurację runtime, topologię GPU, preset Claude
Code i informację o zakresie walidacji. `extends` jest zabronione i walidator
odrzuca każdy profil, w którym wystąpi.

Szablony Claude Code są pogrupowane według rodziny modelu w
`templates/.claude/MODEL/`. Wewnątrz katalogu każdy plik nosi nazwę profilu,
np. `glm53-flash/glm53-flash-uncensored.settings.local.json`. Wszystkie
szablony z grupy `glm53-flash` używają dla ról Claude wspólnej nazwy
`glm-5.3-flash-high`, a oba warianty Qwen używają
`qwen3.8-flash-next-thinking` dla ról głównych oraz
`qwen3.8-flash-next-fast` dla Haiku i zadań szybkich. Proxy wiąże oba aliasy
dynamicznie z checkpointem aktywnego profilu i normalizuje nieobsługiwane
przez Qwen poziomy effort `high` oraz `max` do `xhigh`.

## Przygotowanie kontrolera

```bash
cd my-llm
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
cp .env.example .env
```

`./run` zawsze korzysta z `.venv/bin/python` tego repo.
Przed uruchomieniem proxy ustaw własny losowy `LITELLM_MASTER_KEY` w `.env`;
kontroler nie ma wbudowanego klucza domyślnego.

Receptury i ich izolowane środowiska są zawsze przechowywane lokalnie pod
`.runtime/recipes/`. Kontroler nie korzysta z buildów należących do innych
checkoutów.

## Użycie

```bash
./run launcher
./run launcher list
./run launcher start glm53-flash
./run launcher start qwen38-flash
./run launcher start qwen38-flash-uncensored
./run launcher switch qwen38-flash
./run launcher switch deepseek-v4-flash
./run launcher status
./run launcher logs --follow
./run launcher logs --component litellm --follow
./run launcher stop

# dodatkowa pula czterech workerów, bez zastępowania modelu głównego
./run worker-pool start qwen38-4x27b
./run worker-pool status

# cały profil Qwen 4+1+1+1+1 oraz LiteLLM
./run launcher start qwen-multi
./run launcher status
# jawne, łagodne zatrzymanie wszystkich jego elementów
./run launcher stop --profile qwen-multi
./run worker-pool logs --follow
./run worker-pool stop

./run profiles list
./run profiles show glm53-flash

./run install --profile glm53-flash
./run model verify glm53-flash
./run model verify qwen38-flash
./run model verify qwen38-flash-uncensored

# domyślnie: v0.29, MRV2, TP8/no-EP, packed MXFP4 GEMV,
# DFlash2 K4, FP8 KV, 768K i Vision
./run launcher start glm53-flash

# osobny checkpoint UNCENSORED, ten sam zakwalifikowany runtime v0.29
./run launcher start glm53-flash-uncensored

# jawna diagnostyka pojedynczego komponentu
./run launcher start glm53-flash --runtime-only
./run launcher stop --runtime-only
```

`./run launcher` otwiera prosty interaktywny wybór modeli. Te same operacje są
dostępne jako podkomendy do skryptów i automatyzacji. `start` nigdy nie
zastępuje działającego modelu; zmiana wymaga jawnego `switch` albo wcześniejszego
`stop`. Start korzysta z trwałej jednostki użytkownika, sprawdza host oraz
limit PPT0 nieprzekraczający 285 W na każdej widocznej karcie, a następnie
czeka na gotowość API. Niższy limit jest akceptowany.

Zwykłe `start`, `switch` i `stop` zarządzają transakcyjnie całym stosem:
modelem na porcie `8000`, LiteLLM na porcie `4000`, testem proxy oraz
ustawieniami Claude Code. Błąd proxy po świeżym starcie powoduje bezpieczne
wycofanie uruchomionego runtime. Jawne `--runtime-only` jest przeznaczone dla
diagnostyki pojedynczego komponentu. Pełny start wymaga wcześniejszego
`./run proxy install` oraz ustawienia `LITELLM_MASTER_KEY` w `.env`.

Pełny stack z LiteLLM i ustawieniami Claude Code:

```bash
./run proxy install
./run stack presets
./run stack start --preset qwen38-flash
./run stack stop
```

Po starcie wybranego profilu Qwen interaktywny Claude Code uruchamia się przez
`scripts/claude-local.sh`. Odtwarzalny test agenta kodowego:

```bash
./run test claude-code \
  --profile qwen38-flash-uncensored \
  --output logs/validation/claude-code-qwen38-flash-uncensored-thinking.json
```

Wspólny profil `qwen-multi` ma również workflow pięcioagentowy. Główny
Flash-Next pozostaje aktywnym architektem, implementuje rdzeń i integruje
wynik, a cztery nazwane role kierują równoległe zadania do puli Qwen 27B DP4:

```bash
./run launcher start qwen-multi
scripts/claude-qwen-team.sh
```

Definicje są osadzone w profilu produkcyjnym i przechowywane odtwarzalnie w
`templates/.claude/qwen-multi/qwen-multi.agents.json`. Explorer i verifier są
pozbawieni edycji; dwa implementery muszą otrzymać rozłączne zakresy plików.
Powtarzalny gate routingu głównego modelu i czterech równoległych ról:

```bash
./run test claude-team \
  --profile qwen-multi \
  --output logs/validation/qwen-multi-claude-team.json
```

Start odbywa się wyłącznie przez użytkownikową jednostkę
`r9700-runtime.service`. Skrypty nie wykonują rebootu, resetu GPU ani SIGKILL.

## Checkpointy

Repo nie przechowuje wag. `./run model download PROFILE` pobiera przypiętą
rewizję, a `./run model adopt PROFILE --directory PATH` rejestruje istniejący
checkpoint. GLM dodatkowo przypina i sprawdza rozmiar oraz SHA-256 draftera
DFlash2. Serwis nie wystartuje, jeżeli którykolwiek wymagany artefakt ma inną
tożsamość.

Domyślny `glm53-flash` używa recepty v0.29 ROCm 10, MRV2, TP8 bez Expert
Parallel, packed RDNA4 MXFP4 decode GEMV, DFlash2 K4, FP8 KV,
kontekstu 786,432 tokenów i TP8-sharded Vision; prefix cache i CPU offload są
wyłączone. Profile produkcyjne nie zawierają `experimental_modes`; warianty
robocze pozostają w `profiles/dev/`.

Podczas pracy Claude Code można bez restartu obserwować kolejkę, zajęcie KV i
estymowany postęp prefillu, a po zakończeniu żądania dokładny server-side
prefill, decode, TTFT i E2E. Dla kompletnego układu Qwen 4+1+1+1+1:

```bash
scripts/watch-qwen-live.sh
```

Polecenie otwiera odświeżany dashboard terminalowy zamiast przewijanej listy.

Szczegóły: [architektura](docs/architecture.md),
[operacje](docs/operations.md), [dowody i ograniczenia](docs/verification.md),
[Qwen w Claude Code](docs/qwen38-claude-code.md),
[Qwen3.8-27B: pula 4×R9700](docs/qwen38-4x27b-workers.md),
[wspólny profil Qwen 4+1+1+1+1](docs/qwen-multi.md),
[plan eksperymentu v0.31](docs/glm53-gfx1201-mxfp4-tiled.md).
