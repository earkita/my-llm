# Architektura konfiguracji

## Jeden plik deploymentu

Publiczną jednostką konfiguracji jest plik
`profiles/production/<nazwa>.json`. Zawiera trzy sekcje:

- `model`: repozytorium Hugging Face, pełny commit, oczekiwane pliki i
  możliwości backendu;
- `runtime`: receptę oprogramowania, topologię GPU, równoległość, cache,
  scheduler, transport i spekulację;
- `stack`: aliasy LiteLLM oraz kompletny obiekt ustawień Claude Code.

Loader sprawdza cały dokument atomowo, odrzuca `extends`, wymaga statusu
`production-ready` i weryfikuje zgodność rodziny modelu z runtime. Pole `name`
musi odpowiadać nazwie pliku.

Model, runtime i preset nie są rozwiązywane z osobnych drzew. Komendy
`config model` i `config runtime` zwracają tylko odpowiednią sekcję tego samego
pliku, co pozwala zachować istniejące moduły backendów bez utraty płaskiego
kontraktu.

## Recepty

Manifesty w `manifest/` są niezmiennymi instrukcjami budowy oprogramowania,
nie profilami użytkownika. Rejestr zawiera wyłącznie:

- `vllm_deepseekv4flash_v0.28` dla DeepSeek;
- `vllm_qwen38flash_pr53896` dla Qwen;
- `llamacpp_glm53flash_pr27754` dla GLM.

Nazwa receptury ma format `backend_target_version`. Wszystkie jej patche,
również dotyczące bibliotek pomocniczych, znajdują się wyłącznie w katalogu
`patches/<nazwa-receptury>/`.

Manifest wskazuje dokładny commit, constraints oraz uporządkowane patche z
SHA-256. llama.cpp korzysta z ROCm SDK 7.14 przygotowanego przez zachowaną
receptę vLLM 0.28; instalator realizuje ten fundament automatycznie.

## Stan procesu

Kontroler zapisuje zweryfikowaną tożsamość procesu w `.runtime/service.json`.
Start systemd używa `KillMode=control-group`, `KillSignal=SIGINT` oraz
`SendSIGKILL=no`. Stop sprawdza PID, boot ID, start ticks i cgroup zamiast
zgadywać procesy.
