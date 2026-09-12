# Qwen multi: profil 4+1+1+1+1

`qwen-multi` jest wspólnym profilem produkcyjnym dla wszystkich ośmiu R9700:

- główny `Qwen3.8-Flash-Next-UNCENSORED-MXFP4-FP8` działa na czterech kartach
  jako TP4/EP4, na porcie `8000`;
- `Qwen3.8-27B-Quark-AWQ-INT4-W4A16` działa jako cztery niezależne repliki
  TP1, czyli DP4, na porcie `8100`, z zewnętrznym drafterem
  `Qwen3.8-27B-DFlash2-W4A16` K4;
- LiteLLM publikuje aliasy obu klas na porcie `4000`;
- Claude Code domyślnie używa głównego Flash-Next; aliasy workerów można
  wybrać jawnie.

Profil nie używa `extends`. Osadza kompletny model, runtime i ustawienia stosu
głównego, a komponent workerów wiąże nazwą i SHA-256 kompletnego,
samowystarczalnego profilu `qwen38-4x27b`. Loader odrzuci start, jeśli któryś
z przypiętych profili zmieni się bez aktualizacji kontraktu `qwen-multi`.

## Obsługa

Uruchomienie całego stosu:

```bash
./run launcher start qwen-multi
```

Jeśli zgodne elementy już działają, są przejmowane bez restartu. Inny główny
model lub inna pula workerów powodują błąd zamiast niejawnej podmiany.

Stan wszystkich trzech usług:

```bash
./run launcher status
```

Jawne zatrzymanie wspólnego profilu zatrzymuje najpierw proxy, potem pulę
workerów i na końcu główny runtime. Wszystkie etapy są łagodne i nie używają
SIGKILL:

```bash
./run launcher stop --profile qwen-multi
```

Niezależna diagnostyka nadal jest dostępna przez `./run service ...`,
`./run worker-pool ...` i `./run proxy ...`.

## Routing

Alias główny: `qwen3.8-flash-next-thinking`; szybki wariant bez thinking:
`qwen3.8-flash-next-fast`. Workery są dostępne jako
`qwen3.8-27b-workers-thinking` i `qwen3.8-27b-workers-fast`.

Szablon Claude Code znajduje się w
`templates/.claude/qwen-multi/qwen-multi.settings.local.json`. Thinking dla
głównego modelu pozostaje włączony, a effort `high`/`max` jest normalizowany
przez LiteLLM do obsługiwanego przez Qwen `xhigh`.

## Kwalifikacja na 8×R9700/gfx1201

11 września 2026 poprzedni wariant MTP K1 został aktywowany nad już
działającymi, zgodnymi komponentami. 12 września sama pula workerów została
łagodnie przełączona na przypięty DFlash2 K4 z FP8 KV; główny runtime i proxy
nie zostały zatrzymane. `launcher status` potwierdził wszystkie trzy jednostki
jako `active` oraz `ready`, a działająca pula spełnia bieżący hash komponentu.

`./run model verify qwen-multi` potwierdził indeks 131 shardów głównego
checkpointu (125,790,261,498 bajtów plików), a
`./run model verify qwen38-4x27b` sprawdza przypiętą rewizję AMD, pojedynczy
shard 19,512,909,752 bajtów oraz dokładne hashe pliku wag i konfiguracji
DFlash2. Pełne skany skończoności tensorów oraz poprzednie kwalifikacje
reasoning, tool calling, prefix cache, DFlash2, granicy kontekstu i Claude Code
są opisane w `docs/verification.md`. Kwalifikacja FP8 objęła każdą z czterech
rang i dokładną granicę 131,072 na lokalnej R9700/gfx1201.

Historyczny smoke test przez wspólny endpoint LiteLLM zwrócił HTTP 200 i `finish_reason`
`stop` dla wszystkich czterech aliasów. Oba aliasy `thinking` zwróciły osobne
`reasoning_content`, a oba aliasy `fast` odpowiedziały bez reasoning. Plan
`./run launcher stop --profile qwen-multi --dry-run` zawarł kolejno proxy, pulę
workerów i główny runtime; żadna usługa nie została zatrzymana podczas tej
kontroli.

Minimalna powtarzalna kontrola po starcie:

```bash
./run launcher status
./run model verify qwen-multi
./run model verify qwen38-4x27b
./run proxy test --timeout 120
./run test unit
./run install --profile qwen-multi --dry-run
./run doctor --profile qwen-multi
```
