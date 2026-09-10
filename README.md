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
| `qwen38-flash` | vLLM 0.28 | 8 | TP8/EP8 | 262,144 | MTP K2 |

Każdy deployment jest jednym plikiem w `profiles/production/`. Plik zawiera
pin checkpointu, kompletną konfigurację runtime, topologię GPU, preset Claude
Code i informację o zakresie walidacji. `extends` jest zabronione i walidator
odrzuca każdy profil, w którym wystąpi.

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
./run launcher switch qwen38-flash
./run launcher switch deepseek-v4-flash
./run launcher status
./run launcher logs --follow
./run launcher logs --component litellm --follow
./run launcher stop

./run profiles list
./run profiles show glm53-flash

./run install --profile glm53-flash
./run model verify glm53-flash

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
prefill, decode, TTFT i E2E:

```bash
.venv/bin/python scripts/watch-claude-throughput.py
```

Szczegóły: [architektura](docs/architecture.md),
[operacje](docs/operations.md), [dowody i ograniczenia](docs/verification.md),
[plan eksperymentu v0.31](docs/glm53-gfx1201-mxfp4-tiled.md).
