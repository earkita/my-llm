# DeepSeek V4 Flash Vision Exp na 8× R9700 przez llama.cpp

Status: **wstrzymane na życzenie użytkownika (2026-09-04)**.

Po zapisaniu tego dokumentu repo ma pozostać w działającym stanie sprzed
integracji. Aktywny GLM 5.3/vLLM nie został zatrzymany ani zmieniony.

## Cel

Uruchomić istniejący checkpoint:

```text
/mnt/ai/models/deepseek/DeepSeek-V4-Flash-Vision-Exp-GGUF
├── UD-Q4_K_XL/*.gguf
└── mmproj-BF16.gguf
```

przez oficjalny `ggml-org/llama.cpp`, najpierw jako text-only, a następnie z
rzeczywistym wejściem obrazu, na 8× AMD Radeon AI PRO R9700 (`gfx1201`) z ROCm.
Docelowy serwer ma słuchać na `0.0.0.0:8080` i udostępniać API zgodne z OpenAI.

## Ustalony stan modelu

- Lokalny `README.md` został przeczytany w całości.
- W katalogu modelu nie ma żadnych `*.json`; wymóg przeczytania wszystkich
  plików JSON jest więc spełniony pustym zbiorem.
- Repozytorium HF: `unsloth/DeepSeek-V4-Flash-Vision-Exp-GGUF`.
- Przypięta rewizja HF: `37044a3cff5ed45e3832f3e833bf1cff91f7f168`.
- Pięć shardów jest kompletnych i ma rozmiary zgodne z HF. Łączny rozmiar
  shardów modelu to `155095241184` B.
- Mały shard `00001` (`5257472` B) jest prawidłowy: zawiera metadane i
  tokenizer, ma `0` tensorów. Pozostałe shardy mają łącznie `1328` tensorów
  (`414 + 432 + 412 + 70`). Loaderowi należy podać właśnie shard `00001`;
  pozostałe zostaną znalezione automatycznie.
- GGUF v3, architektura `deepseek4`, `43` bloki, natywny kontekst `1048576`,
  embedding `4096`, `256` ekspertów i `6` aktywnych ekspertów.
- `mmproj-BF16.gguf`: `934462656` B, `427` tensorów, architektura `clip`,
  `clip.projector_type=deepseek4v`, encoder vision włączony, wejście `672×672`.

Tożsamości plików z HF:

| Plik | SHA-256 |
|---|---|
| `...-00001-of-00005.gguf` | `e9d59e58c1b04c78b6cee7c139ea4ae078f5b04315469cec34dc5613e33d573f` |
| `...-00002-of-00005.gguf` | `6a9aa38747abed3bf6a546d903ac4787a8580b63dd3d0353af725df2b5687b3b` |
| `...-00003-of-00005.gguf` | `e72d563a52466a56c2445eaf1fc31b611cdb8b536847a5e9feaae06d0c386859` |
| `...-00004-of-00005.gguf` | `e1edff48e98877be958ae6ca3b96ec2e780152b6d38fbeb73dd90b5b09b33f51` |
| `...-00005-of-00005.gguf` | `ea50c2b54db7bfc433adcfe69ce42c78d8eac12dc8adfd6b134fcbf0717a5656` |
| `mmproj-BF16.gguf` | `e4914c6c8063d01f4cbb6dafdf2f959c7d06fbe8ad11ae5b11ad032edd42642e` |

## Krytyczny blocker jakości vision

Oficjalny PR [#28133](https://github.com/ggml-org/llama.cpp/pull/28133)
dodał obsługę vision/libmtmd, ale późniejszy PR
[#28154](https://github.com/ggml-org/llama.cpp/pull/28154) poprawił dwie rzeczy
konieczne dla prawidłowego obrazu: osobny bias routingu ekspertów dla wejścia
vision oraz wyłączenie SWA dla non-causal attention. Autor zaznaczył, że tekstowy
GGUF musi zostać ponownie skonwertowany.

Lokalne shardy (i zdalna rewizja HF) mają `exp_probs_b.bias` oraz
`ffn_gate_tid2eid`, ale **nie mają `exp_probs_b_vl`**. Powstały przed #28154;
sam `mmproj` został dodany później. Konsekwencje:

- text-only można poprawnie kwalifikować;
- aktualny llama.cpp może załadować projector i obsłużyć żądanie z obrazem;
- nie wolno uznać jakości vision za poprawną tylko dlatego, że żądanie się nie
  wywraca;
- przed wznowieniem trzeba najpierw sprawdzić, czy Unsloth opublikował nową
  rewizję shardów zawierającą `exp_probs_b_vl`;
- jeżeli nie, jest to blocker checkpointu. Zgodnie z wymaganiami nie
  konwertujemy ani nie modyfikujemy lokalnego checkpointu jako obejścia.

Kontrola po ewentualnym odświeżeniu modelu:

```bash
for shard in /mnt/ai/models/deepseek/DeepSeek-V4-Flash-Vision-Exp-GGUF/UD-Q4_K_XL/*.gguf; do
  dd if="$shard" bs=1M count=16 status=none | strings | rg 'exp_probs_b_vl|exp_probs_b\.bias'
done
```

## Wybrany upstream

Pin do użycia po wznowieniu:

```text
tag:    b10793
commit: d230ddd763ffe27781c7ffd237ea78b639b36b6d
repo:   https://github.com/ggml-org/llama.cpp.git
```

Release [b10793](https://github.com/ggml-org/llama.cpp/releases/tag/b10793)
zawiera oba merge'e vision oraz późniejszą poprawkę konwersji DSpark:

- #28133, merge `7798007a29a90e3053e799394da48cf53a2f8e0f`;
- #28154, merge `9400c8946e4da5e7694f2c26d6d4e50e14b690fa`;
- #28294, poprawka konwersji DSpark.

Nie używać semver `v0.3.0`: został wydany przed PR-ami vision. Nie używać forka
Unsloth ani przypadkowego brancha. Baseline ma być czystym upstreamem bez
patchy.

## Znane ryzyka ROCm/HIP

- [#26399](https://github.com/ggml-org/llama.cpp/issues/26399) jest nadal
  otwarte: `GGML_OP_TOP_K` może przechodzić na CPU już po około 3–4K kontekstu,
  co w zgłoszeniu obniża decode około 6,4×. Nie jest to błąd jakości, lecz ważny
  punkt benchmarku i analizy logu.
- [#27021](https://github.com/ggml-org/llama.cpp/issues/27021) jest zamknięte,
  ale opisuje historyczny crash HIP TOP_K powyżej około 128K. Startowe 32K nie
  trafia w ten warunek; nie zwiększać od razu ponad 128K.
- [#21170](https://github.com/ggml-org/llama.cpp/pull/21170) pozostaje otwarte.
  Dotyczy kontekstu urządzenia HIP przy zapisie/przywracaniu stanu na wielu GPU
  i może być istotne dla prompt cache. Nie przenosić starego patcha GLM bez
  reprodukcji.
- Bazowy build ma mieć `GGML_HIP_RCCL=OFF`. Layer split nie potrzebuje RCCL,
  a jego włączenie nie jest automatycznie korzystne.
- Używać `--split-mode layer`. Tryb `tensor` istnieje, ale jest eksperymentalny
  i nie jest rozsądnym baseline'em dla HIP; `row` jest przestarzały.

## Plan zmian w repo

Repo wymaga dokładnie trzech płaskich plików produkcyjnych. Dlatego nie dodawać
czwartego profilu:

1. Zastąpić zawartość `profiles/production/deepseek-v4-flash.json` nowym
   deploymentem llama.cpp, zachowując nazwę profilu `deepseek-v4-flash`.
2. Dodać recipe `llamacpp_deepseekv4vision_b10793`:
   - `manifest/llamacpp_deepseekv4vision_b10793.json`;
   - źródło, build i binaria wyłącznie pod
     `.runtime/recipes/llamacpp_deepseekv4vision_b10793/`;
   - official `ggml-org/llama.cpp`, pin pełnego SHA;
   - bez patchy na pierwszy build;
   - `foundation_recipe=vllm_glm53flash_v0.28`, aby użyć lokalnego ROCm 7.14.
3. Usunąć nieużywany recipe `vllm_deepseekv4flash_v0.28`, jego manifest i
   patche. Nie zostawiać go tylko jako foundation: po zmianie profilu jego
   post-install smoke sprawdzałby inny backend.
4. Zmienić domyślny recipe vLLM w `manifest/recipes.json` na istniejący recipe
   GLM.
5. Rozszerzyć `r9700/backends/llama_cpp.py`:
   - opcjonalny `model.llama_cpp.mmproj_file` → `--mmproj`;
   - nie dodawać `--mmproj`, gdy
     `runtime.multimodal.language_model_only=true`;
   - budować i atestować zarówno `llama-server`, jak i `llama-cli`;
   - akceptować `CMAKE_HIP_ARCHITECTURES=gfx1201` w komunikacie instalatora.
6. Walidować `model_file` i `mmproj_file` jako bezpieczne ścieżki względne.
7. Zaktualizować testy:
   - nazwa recipe ma dopuszczać nightly `b[0-9]+`;
   - asset/constraints checks muszą rozróżniać manifest vLLM i llama.cpp;
   - komenda DeepSeek ma zawierać pierwszy shard, `--mmproj`, layer split i
     osiem wag `--tensor-split`;
   - tryb `text-only` ma nie zawierać `--mmproj`;
   - testy `patch`/`gpu` nie mogą bezwarunkowo zakładać `torchrun` i źródeł
     vLLM;
   - dodać test odrzucający absolutne/uciekające ścieżki projectora.
8. Zaktualizować `config/litellm.yaml`, README, docs, `provenance.json` i po
   realnych pomiarach `summary.md`.

## Początkowa konfiguracja runtime

```text
backend:          llama-cpp / HIP
context:          32768
GPU:              8
GPU order:        0,1,2,3,7,4,5,6
split mode:       layer
tensor split:     1,1,1,1,1,1,1,1
GPU layers:       all
parallel slots:   1
batch / ubatch:   4096 / 512 (zacząć konserwatywnie)
KV:               f16 / f16
flash attention:  on
fit:              off
load mode:        mmap
CPU offload:      0
speculation:      off
prompt cache:     off na baseline
RCCL:             off
```

Profil bazowy może zawierać projector, natomiast w tym samym płaskim pliku ma
być jawny `experimental_modes.text-only`, który ustawia tylko:

```json
{
  "multimodal": {
    "language_model_only": true
  }
}
```

To pozwala przeprowadzić oba etapy przez istniejący launcher, bez nowego
frameworka ani launchera.

## Kolejność wykonania po wznowieniu

1. Ponownie sprawdzić bieżącą rewizję HF i obecność `exp_probs_b_vl`.
2. Wprowadzić opisane zmiany i uruchomić:

   ```bash
   make unit
   ./run install --profile deepseek-v4-flash --dry-run
   ```

3. Zbudować czysty upstream:

   ```bash
   ./run install --profile deepseek-v4-flash
   ./run test runtime --profile deepseek-v4-flash
   ```

4. Zaadoptować lokalne pliki. To wykona pełne SHA-256 około 155 GB i utworzy
   `.model-source.json`:

   ```bash
   ./run model adopt deepseek-v4-flash
   ./run model verify deepseek-v4-flash
   ```

5. Dopiero teraz użyć skillu bezpiecznego stopu i zatrzymać aktywny GLM. Nie
   używać SIGKILL i nie resetować GPU.
6. Uruchomić text-only na porcie 8080:

   ```bash
   skills/start-r9700-runtime/scripts/start-runtime.sh \
     --profile deepseek-v4-flash \
     --runtime-mode text-only \
     --host 0.0.0.0 \
     --port 8080 \
     --ready-timeout 1800
   ```

7. Sprawdzić w logu:
   - dokładny commit/tag;
   - HIP/gfx1201;
   - załadowanie wszystkich pięciu shardów;
   - przydział warstw na osiem urządzeń;
   - brak CPU offloadu i brak nieoczekiwanych tensorów na CPU.
8. W czasie prefill i decode zebrać `amd-smi` dla wszystkich ośmiu GPU.
   Samo wykrycie urządzeń nie jest dowodem użycia wszystkich kart.
9. Wykonać test `/health`, `/v1/models` i
   `/v1/chat/completions`. Odpowiedź ma być logiczna i bez zapętleń/repeated
   `<`. Zwrócić uwagę na problem microbatch z dawnego #26471.
10. Zatrzymać text-only, uruchomić bazowy profil z `--mmproj` tym samym skillem
    na `0.0.0.0:8080`.
11. Potwierdzić w `/v1/models` capability/modality vision. Wysłać rzeczywisty
    obraz jako OpenAI content part:

    ```json
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/jpeg;base64,..."
      }
    }
    ```

    Nie uznawać samego HTTP 200 za sukces jakości. Odpowiedź musi trafnie
    opisywać jednoznaczne elementy obrazu; brak `exp_probs_b_vl` pozostaje
    zastrzeżeniem nawet przy pozornie poprawnej odpowiedzi.
12. Wykonać szybki benchmark klienta:

    ```bash
    ./run benchmark \
      --profile deepseek-v4-flash \
      --url http://127.0.0.1:8080 \
      --prompt-tokens 2048 \
      --output-tokens 128 \
      --concurrency 1 \
      --warmup 1 \
      --repetitions 3 \
      --output logs/benchmarks/deepseek-v4-vision-32k.json
    ```

    Dodatkowo wykonać pojedynczy test z promptem 8192 tokenów, aby zobaczyć,
    czy występuje zgłoszony fallback TOP_K do CPU.
13. Dopiero po przejściu jakości rozważyć 64K, potem 128K. Nie przekraczać
    128K bez osobnej kwalifikacji.
14. Prompt cache włączyć dopiero jako osobny A/B po stabilnym baseline. Jeśli
    reprodukuje się problem kontekstu urządzenia HIP, wtedy ocenić minimalny
    backport #21170; nie dodawać go prewencyjnie.
15. Zapisać w `summary.md`: commit llama.cpp, backend HIP, ROCm, model/revision,
    kontekst, split, mapa GPU, czasy ładowania, test tekstu/obrazu, prefill i
    decode tok/s oraz wszystkie ograniczenia.
16. Uruchomić `make check`, sprawdzić czysty diff, wykonać scoped Conventional
    Commit i push przez klucz `~/.ssh/git`.

## Kryteria zakończenia

- Czysty, przypięty upstream ładuje model bez konwersji i bez patcha.
- Wszystkie osiem R9700 ma rzeczywistą alokację oraz aktywność podczas
  inference.
- Prefill i decode działają przy 32K.
- `/v1/chat/completions` zwraca sensowną odpowiedź tekstową.
- Serwer z `mmproj` rozpoznaje realny obraz, a ograniczenie checkpointu po
  #28154 jest rozwiązane albo jawnie oznaczone jako blocker.
- Benchmark ma dokładne liczniki tokenów i zapisany wynik.
- Repo przechodzi pełne testy, a dokumentacja nie deklaruje więcej niż zostało
  realnie zmierzone.
