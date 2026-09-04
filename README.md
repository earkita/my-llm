# my-llm

Minimalne repozytorium uruchomieniowe dla trzech zweryfikowanych modeli na
AMD Radeon AI PRO R9700. Nie zależy od innych checkoutów: receptury,
środowiska `.runtime`, stan usług i logi są lokalne. Checkpointy pozostają na
dedykowanym magazynie modeli wskazanym przez profile produkcyjne.

## Profile

| Profil | Backend | GPU | Równoległość | Kontekst | Spekulacja |
|---|---|---:|---|---:|---|
| `deepseek-v4-flash` | vLLM 0.28 | 6 | TP1/PP6 | 1,048,576 | DSpark K5 |
| `glm53-flash` | vLLM `main` `c7e6e36` po PR #53906 | 8 | TP8/EP8 | 262,144 | DFlash2 K7 domyślnie |
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

# domyślnie: MRV2, TP8/EP8, DFlash2 K7, BF16 KV, 256K
./run launcher start glm53-flash

# zgodnościowy alias domyślnego trybu oraz diagnostyczne fallbacki
./run launcher start glm53-flash --runtime-mode dflash2-k1
./run launcher start glm53-flash --runtime-mode dflash2
./run launcher start glm53-flash --runtime-mode target-only-32k

# eksperymentalnie: target-only, FP8 KV, pełny kontekst 1M
./run launcher start glm53-flash --runtime-mode long-context-1m-fp8

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

Domyślny `glm53-flash` używa MRV2, TP8/EP8, DFlash2 K7, BF16 KV oraz kontekstu
262,144 tokenów; prefix cache i CPU offload są wyłączone. Pełna próba
`262016 + 128` zakończyła się spójną odpowiedzią, 599.39 tok/s obserwowanego
prefill, 23.84 tok/s decode oraz 111/111 zaakceptowanych draftów. Poprawki
`0020`-`0022` wyrównują strony kpool, zgłaszają rzeczywiste strony kernela
128/256 tokenów i zabezpieczają odczyty block table. Tryb target-only 32K
pozostaje jawnym fallbackiem; diagnostyczny 400K nie jest kwalifikowany.
Patche `0023`-`0026` zmniejszają workspace indexera, raportują faktyczne
alokacje cache i dodają dla gfx1201 czytnik sparse MLA FP8 oparty na Tritonie.
Tryb `long-context-1m-fp8` zaalokował cache o pojemności 1,187,115 tokenów i
ukończył warm-up, ale nie jest jeszcze kwalifikowany jakościowo: przed testem
API został bezpiecznie zatrzymany po nowych poprawialnych błędach PCIe
`BadTLP`.

Szczegóły: [architektura](docs/architecture.md),
[operacje](docs/operations.md), [dowody i ograniczenia](docs/verification.md).
