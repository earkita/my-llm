# my-llm

Minimalne repozytorium uruchomieniowe dla trzech zweryfikowanych modeli na
AMD Radeon AI PRO R9700. Nie zależy od innych checkoutów: receptury,
środowiska `.runtime`, stan usług i logi są lokalne. Checkpointy pozostają na
dedykowanym magazynie modeli wskazanym przez profile produkcyjne.

## Profile

| Profil | Backend | GPU | Równoległość | Kontekst | Spekulacja |
|---|---|---:|---|---:|---|
| `deepseek-v4-flash` | vLLM 0.28 | 6 | TP1/PP6 | 1,048,576 | DSpark K5 |
| `glm53-flash` | vLLM `main` `7fbd44c`, ROCm 10 | 8 | TP8/no-EP | 1,048,576 | packed MXFP4 GEMV + DFlash2 K4, FP8 KV |
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
./run launcher start qwen38-flash --with-litellm
./run launcher switch qwen38-flash
./run launcher switch deepseek-v4-flash --with-litellm
./run launcher status
./run launcher logs --follow
./run launcher logs --component litellm --follow
./run launcher stop --with-litellm

./run profiles list
./run profiles show glm53-flash

./run install --profile glm53-flash
./run model verify glm53-flash

# domyślnie: v0.31, MRV2, TP8/no-EP, tiled BN8 MXFP4 GEMV,
# DFlash2 K4, FP8 KV, 768K i Vision
./run launcher start glm53-flash

# topology-exact rollback: v0.29, scalar GEMV, 768K i Vision
./run launcher start glm53-flash-v029-rollback \
  --runtime-mode mxfp4-gemv-dflash2-k4-fp8-768k-vision-weights

skills/start-r9700-runtime/scripts/start-runtime.sh --profile glm53-flash
./run service status
skills/stop-r9700-runtime/scripts/stop-runtime.sh
```

`./run launcher` otwiera prosty interaktywny wybór modeli. Te same operacje są
dostępne jako podkomendy do skryptów i automatyzacji. `start` nigdy nie
zastępuje działającego modelu; zmiana wymaga jawnego `switch` albo wcześniejszego
`stop`. Start korzysta z trwałej jednostki użytkownika, sprawdza host oraz
limit PPT0 nieprzekraczający 285 W na każdej widocznej karcie, a następnie
czeka na gotowość API. Niższy limit jest akceptowany.

Tryb bez flagi udostępnia bezpośrednie API modelu na porcie `8000`. Flaga
`--with-litellm` uruchamia transakcyjnie cały stack: model, LiteLLM na porcie
`4000`, test proxy oraz ustawienia Claude Code. Błąd proxy po świeżym starcie
powoduje bezpieczne wycofanie uruchomionego runtime. Wymaga wcześniejszego
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

Domyślny `glm53-flash` używa recepty v0.31 ROCm 10, MRV2, TP8 bez Expert
Parallel, output-tiled BN8 RDNA4 MXFP4 decode GEMV, DFlash2 K4, FP8 KV,
kontekstu 786,432 tokenów i TP8-sharded Vision; prefix cache i CPU offload są
wyłączone. W deployment-exact A/B BN8 poprawił średni decode o 1.54%, a
minimum o 1.20% względem scalar. Vision smoke, NIAH 256K 4/4 i dokładna
granica `786,368 + 64 = 786,432` przeszły 9 września 2026. Osobny profil
`glm53-flash-v029-rollback` zachowuje poprzednią receptę i wszystkie jej tryby.

Podczas pracy Claude Code można bez restartu obserwować kolejkę, zajęcie KV i
estymowany postęp prefillu, a po zakończeniu żądania dokładny server-side
prefill, decode, TTFT i E2E:

```bash
.venv/bin/python scripts/watch-claude-throughput.py
```

Szczegóły: [architektura](docs/architecture.md),
[operacje](docs/operations.md), [dowody i ograniczenia](docs/verification.md),
[plan eksperymentu v0.31](docs/glm53-gfx1201-mxfp4-tiled.md).
