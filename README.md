# my-llm

Minimalne repozytorium uruchomieniowe dla trzech zweryfikowanych modeli na
AMD Radeon AI PRO R9700. Zostało wydzielone z
`/home/ea/ai/llm-runtime`; nie zawiera checkpointów, środowisk `.runtime`,
logów ani profili laboratoryjnych.

## Profile

| Profil | Backend | GPU | Równoległość | Kontekst | Spekulacja |
|---|---|---:|---|---:|---|
| `deepseek-v4-flash` | vLLM 0.28 | 6 | TP1/PP6 | 1,048,576 | DSpark K5 |
| `glm53-flash` | llama.cpp | 8 | split warstwowy | 1,048,576 limit modelu | DFlash2 K7 |
| `qwen38-flash` | vLLM 0.28 | 8 | TP8/EP8 | 262,144 | MTP K2 |

Każdy deployment jest jednym plikiem w `profiles/production/`. Plik zawiera
pin checkpointu, kompletną konfigurację runtime, topologię GPU, preset Claude
Code i informację o zakresie walidacji. `extends` jest zabronione i walidator
odrzuca każdy profil, w którym wystąpi.

## Przygotowanie kontrolera

```bash
cd /home/ea/ai/my-llm
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e .
cp .env.example .env
```

`./run` zawsze korzysta z `.venv/bin/python` tego repo.
Przed uruchomieniem proxy ustaw własny losowy `LITELLM_MASTER_KEY` w `.env`;
kontroler nie ma wbudowanego klucza domyślnego.

## Użycie

```bash
./run profiles list
./run profiles show glm53-flash

./run install --profile glm53-flash
./run model verify glm53-flash

skills/start-r9700-runtime/scripts/start-runtime.sh --profile glm53-flash
./run service status
skills/stop-r9700-runtime/scripts/stop-runtime.sh
```

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

Szczegóły: [architektura](docs/architecture.md),
[operacje](docs/operations.md), [dowody i ograniczenia](docs/verification.md).
