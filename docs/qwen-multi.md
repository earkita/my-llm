# Qwen multi: profil 4+1+1+1+1

`qwen-multi` jest wspólnym profilem produkcyjnym dla wszystkich ośmiu R9700:

- główny `Qwen3.8-Flash-Next-UNCENSORED-MXFP4-FP8` działa na czterech kartach
  jako TP4/EP4, na porcie `8000`;
- `Qwen3.8-27B-Quark-AWQ-INT4-W4A16` działa jako cztery niezależne repliki
  TP1, czyli DP4, na porcie `8100`, z zewnętrznym drafterem
  `Qwen3.8-27B-DFlash2-W4A16` K4;
- LiteLLM publikuje aliasy obu klas na porcie `4000`;
- zwykły Claude Code domyślnie używa głównego Flash-Next, a dedykowany
  workflow zespołowy uruchamia go jako aktywnego leada wraz z czterema
  nazwanymi workerami 27B.

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
`templates/.claude/qwen-multi/qwen-multi.settings.local.json`, a definicje
agentów w `templates/.claude/qwen-multi/qwen-multi.agents.json`. Thinking dla
głównego modelu pozostaje włączony, a effort `high`/`max` jest normalizowany
przez LiteLLM do obsługiwanego przez Qwen `xhigh`.

## Workflow: główny Qwen + cztery workery

Główna sesja nie jest pasywnym routerem. Typ `qwen-team-lead` działa na
`qwen3.8-flash-next-thinking` i odpowiada za pełną analizę, architekturę,
implementację przekrojowego lub najbardziej ryzykownego fragmentu, decyzje
integracyjne, końcowy przegląd oraz walidację. Cztery role boczne działają na
puli DP4:

| Rola | Alias | Zakres |
| --- | --- | --- |
| `qwen-worker-explorer` | `qwen3.8-27b-workers-fast` | read-only discovery i zależności |
| `qwen-worker-implementer-a` | `qwen3.8-27b-workers-thinking` | pierwszy rozłączny zakres plików |
| `qwen-worker-implementer-b` | `qwen3.8-27b-workers-thinking` | drugi rozłączny zakres plików |
| `qwen-worker-verifier` | `qwen3.8-27b-workers-thinking` | read-only testy, regresje i review |

Launcher materializuje osadzone definicje jako ignorowany plik
`.claude/agents.local.json`. Wrapper przekazuje go przez `claude --agents`,
więc role są dostępne również wtedy, gdy skrypt zostanie uruchomiony z katalogu
innego projektu. Dedykowane uruchomienie pięcioagentowe:

```bash
./run launcher start qwen-multi
scripts/claude-qwen-team.sh
```

Pasywny podgląd szybkości głównego modelu i wszystkich czterech workerów,
bez generowania dodatkowego ruchu:

```bash
scripts/watch-qwen-live.sh
```

Każdy worker ma osobny wiersz `WORKER/e0`--`WORKER/e3`. Wynik `COMPLETE`
podaje dokładny server-side prefill/decode, TTFT, E2E, trafienia prefix cache
i akceptację DFlash2; `LIVE` pokazuje kroczący decode, kolejkę, KV oraz
telemetrię przypisanej R9700.

Można także przekazać pierwsze zadanie bez otwierania pustej sesji:

```bash
scripts/claude-qwen-team.sh \
  "Zaimplementuj zmianę, używając głównego Qwena i dokładnie czterech workerów."
```

Ustawienie `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1` oraz tryb
`teammateMode=in-process` są częścią profilu. Lead ma zachować własny zakres
implementacji, przydzielić workerom rozłączne pliki, czekać na wszystkie cztery
raporty i samodzielnie zintegrować wynik. Workery nie mogą wykonywać commitów,
push, stash, resetów ani zarządzać usługami. Explorer i verifier nie mają
narzędzi edycji.

Oba aliasy workerów wskazują wspólny endpoint DP4. Cztery równoległe żądania
są rozkładane na cztery repliki TP1; uruchamianie większej liczby aktywnych
workerów nie zwiększa fizycznej równoległości i tworzy kolejkę. Ponieważ
teammates w trybie in-process współdzielą checkout, zakresy zapisujących
workerów muszą być rozłączne. Do konkurencyjnych implementacji tych samych
plików należy użyć osobnych Git worktree zamiast wspólnego zespołu.

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

12 września workflow Claude Code `2.1.197` przeszedł dodatkowy gate 24/24.
Główny `qwen-team-lead` został rozpoznany jako
`qwen3.8-flash-next-thinking`, wywołał dokładnie cztery typy workerów z
unikalnymi nazwami i `run_in_background=true`, a wszystkie cztery starty
nastąpiły przed pierwszym zakończeniem. Osobne, równoczesne próby ról
potwierdziły `qwen3.8-27b-workers-fast` dla explorera oraz
`qwen3.8-27b-workers-thinking` dla obu implementerów i verifiera. Czasy prób
wyniosły odpowiednio 3.84, 2.45, 2.83 i 2.62 s; wszystkie przedziały nakładały
się w czasie. Nie wystąpiły odmowy narzędzi ani zmiany worktree. Headless
`--print` kończy fazę leada po pierwszej odpowiedzi otrzymanej już po czterech
startach, dlatego komplet odpowiedzi i aliasy ról gate sprawdza w osobnej
równoległej fazie; interaktywna sesja pozostaje otwarta i zbiera powiadomienia
teammates normalnie.

Powtarzalna komenda i ignorowany raport:

```bash
./run test claude-team \
  --profile qwen-multi \
  --output logs/validation/qwen-multi-claude-team.json \
  --timeout 300
```

Minimalna powtarzalna kontrola po starcie:

```bash
./run launcher status
./run model verify qwen-multi
./run model verify qwen38-4x27b
./run proxy test --timeout 120
./run test claude-team \
  --profile qwen-multi \
  --output logs/validation/qwen-multi-claude-team.json
./run test unit
./run install --profile qwen-multi --dry-run
./run doctor --profile qwen-multi
```
