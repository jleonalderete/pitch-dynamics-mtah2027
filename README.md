# Pitch Dynamics

Posesión de pelota por frame para la Fase 1 de
[More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027) (Metrica Sports,
FC Barcelona y Mobile World Capital Barcelona). Equipo **BOCABOCABOCA**.

Para cada uno de los 199.319 frames del video táctico de Barcelona-Athletic se predice `H` (local),
`A` (visitante), `D` (juego detenido) o `X` (fuera de las mitades evaluadas). El modelo se entrena
sólo con datos externos con licencia y usa únicamente el tracking automático (ATD) del partido.
Método, decisiones y limitaciones: [informe técnico](docs/technical-report.md).

## Cómo funciona

1. **Pelota completada desde el broadcast.** Se alinea el ATD broadcast con la línea de tiempo
   táctica y su pelota llena los frames tácticos sin pelota (`broadcast.py`).
2. **Features por frame.** 99 señales de geometría, último toque, golpes, presión y forma de equipo
   (`features.py`, `design.py`).
3. **Clasificador.** Ensamble de dos MLP entrenado con 7 partidos de Bundesliga (DFL) degradados
   para parecerse al ATD (`matches.py`, `mlp.py`).
4. **Decodificación.** Viterbi por mitad con costo por cambio de estado, para no fragmentar las
   posesiones (`decoding.py`).

| Conjunto                                | spellF1 |
| --------------------------------------- | ------: |
| Desarrollo DFL (dejando un partido)     |   0,566 |
| Reserva DFL (2 partidos)                |   0,540 |
| Reserva LaLiga (2 Clásicos SkillCorner) |   0,434 |

## Reproducir la entrega

Requiere Python 3.15 y [uv](https://docs.astral.sh/uv/). Los datos del desafío no se redistribuyen:
copiá los ATD del bundle oficial en `data/` (o indicá otra carpeta con `MTAH_DATA_DIR` o
`--data-dir`):

```
data/
├─ tactical/   BAR-ATH_tactical_atd.xml, BAR-ATH_tactical_atd.txt
└─ broadcast/  BAR-ATH_broadcast_atd.xml, BAR-ATH_broadcast_atd.txt
```

```bash
uv sync --locked
uv run pitch-dynamics data      # descarga y prepara los 7 partidos DFL (unos 5 minutos)
uv run pitch-dynamics predict   # la entrega: artifacts/predict/submission.csv
```

`predict` escribe `submission.csv` y `provenance.json` (commit, versiones, plataforma y resumen de
la alineación) y corre el validador oficial. Semillas e hilos están fijos: en la misma plataforma
el archivo es idéntico byte a byte.

## Otros comandos

```bash
uv run pitch-dynamics evaluate              # desarrollo: dejando un partido DFL afuera
uv run pitch-dynamics holdout --set laliga  # reserva: una sola vez, con el modelo congelado
uv run pitch-dynamics align                 # sólo la alineación del broadcast, con reporte
```

`holdout`, `align` y `predict` leen datos protegidos y se niegan a correr dentro de agentes de IA
(variable `CLAUDECODE`): los datos del partido no se pasan a servicios alojados.

## Estructura

```
src/pitch_dynamics/
  cli.py                  línea de comandos
  submission.py           pipeline de la entrega
  challenge.py            lectura del ATD del desafío y escritura del CSV
  broadcast.py            alineación del broadcast y relleno de la pelota táctica
  sources.py, matches.py  datos externos, splits y degradación
  features.py, design.py  evidencia por frame
  mlp.py                  clasificador
  decoding.py             Viterbi por mitad
  evaluation.py           métricas oficiales, desarrollo y reservas
  guard.py, official.py   bloqueo para agentes y acceso al kit oficial
official/                 evaluador, validador y constantes de los organizadores, sin cambios
tests/                    tests con datos sintéticos
docs/technical-report.md  informe técnico
```

## Desarrollo

```bash
uv run pre-commit install
uv run pytest
uv run pre-commit run --all-files
```

Reglas de código y del desafío: [CONVENTIONS.md](CONVENTIONS.md) y [AGENTS.md](AGENTS.md).

## Datos y licencias

- DFL/IDSSE: Bassek et al., _Scientific Data_ 12, 195 (2025),
  [CC-BY 4.0](https://doi.org/10.1038/s41597-025-04505-y).
- SkillCorner open data, MIT.
- `official/`: kit de evaluación de los organizadores, MIT (ver `official/LICENSE`). La verdad
  pública de 5 minutos y los baselines están en el
  [repositorio oficial](https://github.com/metrica-sports/more-than-a-hack-2027).
- Los datos del desafío no se redistribuyen y no están en este repositorio.

Código bajo licencia MIT, ver [LICENSE](LICENSE).
