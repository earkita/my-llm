# Qwen3.8 Flash-Next w Claude Code

Repo zawiera dwa samodzielne profile produkcyjne i dwa odpowiadające im
template'y Claude Code:

| Profil | Checkpoint | KV cache | Stan kwalifikacji |
| --- | --- | --- | --- |
| `qwen38-flash` | `/mnt/ai/models/qwen/Qwen3.8-Flash-Next-MXFP4-FP8` | FP8 z 36 kalibrowanymi skalami | checkpoint sprawdzony; start na R9700 oczekuje na test |
| `qwen38-flash-uncensored` | `/mnt/ai/models/qwen/Qwen3.8-Flash-Next-UNCENSORED-MXFP4-FP8` | BF16/auto | vLLM, API, narzędzia i Claude Code sprawdzone na 4 × R9700 |

Oba profile używają TP4/EP4, CPU offload tabeli PLE, MTP2, natywnego
kontekstu 262144 i pierwszych czterech kart `gfx1201`. Pozostałe cztery karty
są pozostawione dla instancji bocznych układu `4+1+1+1+1`. Główne MXFP4
działa przez emulację OCP, ponieważ zgodność MI350 nie dowodzi obsługi
natywnego kernela na RDNA4.

## Uruchomienie

Pełny launcher materializuje właściwy template jako
`.claude/settings.local.json`, uruchamia vLLM i LiteLLM, a następnie sprawdza
proxy. Nie zastępuje niejawnie działającego modelu.

```bash
# zwykły model
./run launcher start qwen38-flash

# albo UNCENSORED
./run launcher start qwen38-flash-uncensored

# w repo, po uruchomieniu wybranego stosu
scripts/claude-local.sh
```

Gdy inny model już działa, jego zmiana wymaga jawnego `launcher switch`.
Template'y źródłowe znajdują się w:

- `templates/.claude/qwen38-flash/qwen38-flash.settings.local.json`;
- `templates/.claude/qwen38-flash/qwen38-flash-uncensored.settings.local.json`.

LiteLLM wiąże z aktywnym checkpointem dwa aliasy przez adapter `hosted_vllm`.
Role główna, Sonnet i Opus używają `qwen3.8-flash-next-thinking`; Haiku i
`ANTHROPIC_SMALL_FAST_MODEL` używają `qwen3.8-flash-next-fast`. Dzięki temu
zadania agentowe zachowują reasoning, a małe operacje mogą korzystać z
krótszej ścieżki. Oba endpointy, `/v1/messages` i `/v1/chat/completions`,
pozostają lokalne.

Template ustawia natywne dla Claude Code `effortLevel=high`. Hook LiteLLM na
granicy deploymentu zmienia dla Qwen `high` i `max` na obsługiwane `xhigh`,
pozostawiając `low`, `medium` i `xhigh` bez zmian. Zmienne
`CLAUDE_CODE_DISABLE_THINKING` i `MAX_THINKING_TOKENS` nie są ustawiane.
Alias thinking stosuje receptę Qwen: `temperature=1`, `top_p=0.95`,
`top_k=20`, `min_p=0`, `presence_penalty=0`, `repetition_penalty=1`,
`enable_thinking=true` i `preserve_thinking=true`. Alias fast stosuje tryb
non-thinking:
`temperature=0.7`, `top_p=0.8`, `top_k=20`, `min_p=0`,
`presence_penalty=1.5`, `repetition_penalty=1`. `preserve_thinking=false`
zapobiega przenoszeniu śladów reasoning w szybkiej ścieżce.

Oba profile mają natywny prefix cache vLLM z interwałem retencji 800 tokenów,
bez zewnętrznego offloadu KV. Auto-compaction Claude Code działa przy 90%
okna 262144 tokenów. Włączenie `preserve_thinking` w głównej ścieżce zachowuje
spójność historii wieloturowej; pełny test granicy kontekstu i skuteczności
cache na nowych checkpointach pozostaje poza obecną kwalifikacją.

Publiczna recepta checkpointu R9700 proponuje TP4, PLE CPU offload, FP8 KV,
budżet prefilla 4096 i MTP3. Profile zachowują ustawienia potwierdzone lokalnie:
TP4/EP4, `max_num_seqs=1`, budżet 2048 i MTP2. Zwykły checkpoint korzysta z
jego kalibrowanych skal FP8 KV; UNCENSORED nie może ich odziedziczyć i używa
BF16/auto. Oficjalna recepta vLLM potwierdza sens TP4/TEP, prefix cache,
reasoning i parsera narzędzi, ale jej wyniki MI355X/Hopper nie są dowodem dla
R9700. Lokalny `qwen3_coder` i wskazywany upstream `qwen3_xml` rozwiązują się
w użytej wersji vLLM do tego samego parsera Qwen3.

## Kwalifikacja Claude Code

Gate niczego nie uruchamia ani nie zatrzymuje. Najpierw wymaga dokładnej
tożsamości działającego profilu, runtime'u i konfiguracji LiteLLM. Potem
sprawdza literalną odpowiedź, widoczny blok reasoning oraz naprawę małego
projektu w jednorazowym katalogu z realnym użyciem `Read`, `Edit` i `Bash`:

```bash
./run test claude-code \
  --profile qwen38-flash-uncensored \
  --output logs/validation/claude-code-qwen38-flash-uncensored-thinking.json \
  --timeout 600
```

Dla zwykłego modelu po jego jawnym uruchomieniu używa się tej samej komendy z
`--profile qwen38-flash`. Raport JSON zawiera wersję Claude Code, hash
template'u, tożsamość procesu, statystyki bloków thinking, liczbę tur,
kolejność narzędzi, użycie tokenów, rezultat testów i wszystkie warunki
gate'u.

Kwalifikacja UNCENSORED z 11 września 2026 r. po włączeniu thinking przeszła
21/21 warunków. Claude Code 2.1.197 zwrócił dziewięć bloków thinking: jeden w
próbie literalnej, jeden w zadaniu rachunkowym i siedem w naprawie kodu.
Naprawa zajęła dziewięć tur, użyła kolejno `Bash`, `Bash`, `Read`, `Read`,
`Bash`, `Edit`, `Bash`, `Bash`, zużyła 14302 tokeny wejścia i 1324 wyjścia,
zmieniła tylko implementację i zakończyła testy sukcesem. Wbudowany w ten sam
gate przypadek fast zwrócił dokładne `FAST_OK` oraz zero bloków thinking.

## Źródła parametrów

- [oficjalny model card Qwen3.8-Flash-Next FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8);
- [oficjalna recepta vLLM Qwen3.8-Flash-Next](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next);
- [checkpoint i recepta MXFP4/FP8 dla R9700](https://huggingface.co/tcclaviger/Qwen3.8-Flash-Next-MXFP4-FP8);
- [ustawienia modeli i effort w Claude Code](https://code.claude.com/docs/en/model-config).
