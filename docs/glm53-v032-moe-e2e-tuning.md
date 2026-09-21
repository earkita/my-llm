# GLM-5.3-Flash v0.32: pełne strojenie decode W4A16 MoE

Data pomiaru: 2026-09-16/17. Sprzęt: 8x AMD Radeon AI PRO R9700
(`gfx1201`), TP8. Runtime: `vllm_glm53_v0.32`, target-only,
FP8 KV, prefix cache, `FULL_AND_PIECEWISE`, synchroniczny scheduler i
`max_num_seqs=8`. Strojenie dotyczy małych bucketów decode; nie jest
strojeniem dużych GEMM-ów prefill.

## Zakres właściwego przeszukania

Dodano niezależny od Ray runner
`scripts/tune-glm53-w4a16-moe.py`. Używa on rzeczywistych kształtów GLM po
TP8: `E=288`, hidden 4096, shard intermediate 512, top-k 8, BF16 activation i
pakowane INT4 z grupą 128. Wyniki są sprawdzane numerycznie względem
konfiguracji domyślnej przed pomiarem.

Przeszukano całą wspieraną i przyciętą przez upstream przestrzeń ROCm dla
małego decode:

- `BLOCK_SIZE_N/K=16/32/64/128/256`;
- `num_warps=1/2/4/8`;
- `waves_per_eu=0/1/2/4`;
- wymagane przez tę ścieżkę: `BLOCK_SIZE_M=16`, `GROUP_SIZE_M=1`,
  `SPLIT_K=1`, `num_stages=2`;
- odrzucono kombinacje przekraczające limit 64 KiB LDS.

Dało to 384 konfiguracje. 352 przeszły kompilację i kontrolę numeryczną, a 32
zostały odrzucone jako `OutOfResources`. Etap 1 mierzył każdy wariant dla
`M=1/2/3/4/8` na deterministycznym shardzie jednej z ośmiu kart. Etap 2
powtórzył 40 najlepszych kandydatów na każdej z ośmiu kart po 50 iteracji.
Etap 3 powtórzył 12 finalistów, konfigurację domyślną i wcześniejszego
kandydata na każdej karcie po 500 iteracji.

Najlepszy klaster mikrobenchmarku miał `BLOCK_N=16`, `BLOCK_K=64`, jeden warp
i dwa stages. Warianty różniły się tylko `waves_per_eu`:

| waves | M1 | M2 | M3 | M4 | M8 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| default | 153.21 us | 157.16 us | 202.94 us | 256.78 us | 457.92 us |
| 4 | 156.04 us | 156.92 us | 188.57 us | 218.53 us | 365.28 us |
| 2 | 156.70 us | 157.15 us | 188.17 us | 217.95 us | 365.36 us |
| 0 | 157.52 us | 156.29 us | 198.46 us | 217.26 us | 351.35 us |

Default pozostaje najlepszy dla M=1, natomiast ręczne konfiguracje wygrywają
dla M=3/4/8. To właśnie ten konflikt sprawdzono następnie w pełnym runtime.
Surowe wyniki strojenia są w `logs/tuning/glm53-w4a16-full/`.

## Kwalifikacja end-to-end 8K + 256

Każdy wariant uruchomiono po pełnym restarcie i świeżym capture grafów dla
rozmiarów 1/2/3/4/8. C1 zawiera pięć mierzonych requestów po warmupie, a C4
12 requestów, czyli trzy rzeczywiście równoległe fale. `OUT tok/s` jest łączną
przepustowością C4.

| wariant | C1 TTFT | C1 decode | C4 TTFT | C4 decode/request | C4 OUT tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| default | 5.822 s | 14.80 | 9.769 s | 6.88 | 21.82 |
| waves=4 dla M=1--12 | 5.826 s | 13.62 | 9.529 s | 7.65 | 23.84 |
| waves=2 dla M=1--12 | 5.847 s | 13.24 | 9.817 s | 6.59 | 21.07 |
| waves=0 dla M=1--12 | 5.838 s | 15.08 | 13.768 s | 7.78 | 21.95 |
| default M=1, waves=4 M=2--12 | 5.822 s | 15.01 | 9.798 s | 6.90 | 21.80 |

Pełny `waves=4` zwiększa aggregate C4 o 9.3%, ale regresuje C1 decode o 8.0%.
Hybryda zachowuje C1, lecz nie zachowuje zysku C4. Oznacza to, że także przy
C4 scheduler wykonuje znaczną część kroków decode z efektywnym M=1; nie można
bez kosztu dla pojedynczej sesji przełączyć tylko „batchowego” kernela M=4.

Wykonano dodatkową parę hybryda/default z identycznym seedem datasetu. C4
wyniosło odpowiednio 21.9 i 21.5 OUT tok/s (+1.9%), czyli nadal poniżej progu
promocji. C1 wykazało duży rozrzut między zimnymi restartami (hybryda 15.0 i
16.6 tok/s mimo identycznego wpisu default dla M=1), dlatego pojedynczy lepszy
restart nie jest uznany za efekt tuningu. Artefakty pełnej kwalifikacji są w
`logs/benchmarks/glm53-moe-full/`.

## Wcześniejszy screening 8K + 128

Przetestowano konfiguracje kernela Triton W4A16 dla małych bucketów MoE:

- `BLOCK_SIZE_N=32/64`;
- `num_warps=2/4`;
- `BLOCK_SIZE_M=16`, `BLOCK_SIZE_K=64`, `num_stages=2`,
  `waves_per_eu=0`.

Każdy wariant uruchamiano po pełnym restarcie runtime i świeżym capture grafów.
Screening obejmował 8192 tokenów wejścia, 128 tokenów wyjścia i rzeczywiście
równoległe fale C1/C2/C3/C4. Log vLLM podczas C4 potwierdzał jednocześnie
`Running: 4 reqs, Waiting: 0`. Finaliści zostali dodatkowo sprawdzeni przy
32K, a najlepszy wynik screeningu powtórzono z 256 tokenami wyjścia oraz przy
128K.

`decode` oznacza średnią szybkość pojedynczego requestu, a `aggregate` łączną
przepustowość wszystkich równoległych requestów.

| wariant | C1 decode | C2 decode / aggregate | C3 decode / aggregate | C4 decode / aggregate |
| --- | ---: | ---: | ---: | ---: |
| bez pliku tuningu | 13.07 | 8.10 / 11.59 | 6.99 / 12.89 | 5.55 / 13.93 |
| N32, W2 | 15.85 | 9.50 / 11.42 | 6.04 / 13.28 | 4.96 / 14.42 |
| N64, W2 | 12.75 | 7.95 / 11.47 | 5.44 / 12.34 | 4.31 / 13.02 |
| N32, W4 | 15.71 | **11.21 / 12.76** | 6.29 / 13.76 | 4.91 / 14.37 |
| N64, W4 | **20.23** | 8.06 / 11.68 | **7.01 / 14.92** | **5.14 / 14.89** |

N64/W4 wygrał C1, C3 i C4, natomiast N32/W4 wygrał C2. Próby połączenia obu
ustawień w jednym pliku pogorszyły wyniki. vLLM wybiera konfigurację według
najbliższego `M=num_tokens`; wyniki wskazują, że pełny graph-captured przebieg
korzysta z kilku bucketów w trakcie zmian aktywnego batcha. Mieszane warianty
osiągały tylko 13.26--13.48 tok/s w C1, 11.42--12.29 tok/s aggregate w C2 i
12.89--13.62 tok/s aggregate w C3.

## Długi kontekst i replikacja

| wariant | workload | TTFT | decode | aggregate output |
| --- | --- | ---: | ---: | ---: |
| N64/W4 | 32K + 256, C1 | 19.125 s | 17.14 tok/s | 7.53 tok/s |
| N32/W4 | 32K + 256, C1 | 19.634 s | 13.43 tok/s | 6.63 tok/s |
| N64/W4 | 32K + 128, C4 | 45.828 s | 1.95 tok/s | 4.60 tok/s |
| N32/W4 | 32K + 128, C4 | 36.943 s | 1.68 tok/s | 4.54 tok/s |
| N64/W4 | 128K + 128, C1 | 59.002 s | 13.55 tok/s | 1.87 tok/s |

Krótki wynik 20.23 tok/s nie powtórzył się po wydłużeniu generacji. Zimna
replikacja N64/W4 na 8K + 256, pięć mierzonych requestów po warmupie, dała
TTFT 5.773 s i **14.59 tok/s**. Porównywalny baseline v0.32 bez pliku
tuningu (`logs/validation/glm53-v031-v032-target-ab/v032-8k`) osiągał
**16.032 tok/s**, więc kandydat regresował o 9.0%. Baseline 32K + 256 wynosił
15.070 tok/s; dodatni wynik N64/W4 przy 32K nie wystarcza do promocji wobec
regresji i braku powtarzalności 8K.

Wysoki raportowany efektywny prefill w części przebiegów 32K/128K zawiera
trafienia prefix cache i nie jest używany do wyboru kernela decode.

## Decyzja

Żaden z 352 poprawnych wariantów ani wariant hybrydowy nie spełnia kryterium
promocji. Produkcyjny plik DFlash/v0.31 pozostaje bez zmian. Minimalny v0.32
wraca do trybu `target-graphs` bez użytkowego pliku MoE. Plik w
`tuning/moe/glm53-flash-search/current` pozostaje wyłącznie artefaktem
diagnostycznym i nie jest najlepszym trybem profilu.

Najważniejszy wniosek: małych bucketów W4A16 nie wolno wybierać wyłącznie z
mikrobenchmarku ani krótkiego 128-tokenowego decode. Pełne przeszukanie
pokazało, że sam kernel MoE ma duży zapas dla M=3/4/8, lecz rzeczywisty decode
C1--C4 jest zdominowany przez kroki M=1 i narzut całego grafu. Następny krok to
trace faktycznego rozkładu `M` i czasu kerneli w graph-captured decode, a nie
dalsze losowe rozszerzanie przestrzeni tile'i.

Starszy screening znajduje się w `logs/benchmarks/glm53-moe-e2e-search/`, a
właściwe pełne strojenie w `logs/tuning/glm53-w4a16-full/` i
`logs/benchmarks/glm53-moe-full/`.
