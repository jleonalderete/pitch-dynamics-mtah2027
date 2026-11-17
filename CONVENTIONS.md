# Convenciones

Obligatorias para personas y agentes. El objetivo es un paquete pequeño que un revisor lea rápido.

## Código

- Todo el código, los identificadores y los docstrings van en inglés.
- Type hints en todas las firmas (`mypy` corre con `disallow_untyped_defs`). Arrays con `numpy.typing.NDArray`.
- Docstring de una línea en módulos, clases y funciones públicas. Más solo cuando el porqué no es evidente.
- Comentarios solo para razones no obvias. Nada de referencias internas (decisiones numeradas, PRs, personas, fechas, experimentos pasados, puntajes).
- Lo mínimo: sin código muerto, opciones sin uso ni abstracciones por si acaso.
- Constantes en MAYÚSCULAS con la unidad en el nombre (`_M`, `_S`, `_FRAMES`, `_MPS`).
- Imports absolutos (`from pitch_dynamics.decoding import viterbi`), nunca relativos. `pathlib` en lugar de `os.path`.
- `print` solo en `cli.py`. No usar `from __future__ import annotations` (un dataclass compara `field.type is bool`).

## Herramientas

- `uv sync` crea el entorno; `uv.lock` se versiona y CI usa `uv sync --locked`.
- `uv run pre-commit install` instala los hooks de commit y de mensaje de commit.
- `uv run pre-commit run --all-files` ejecuta ruff (lint y formato), prettier (markdown, yaml, json), mypy y los chequeos de archivos.
- `uv run pytest` corre los tests. Usan datos sintéticos y nunca datos de partido.
- Los archivos de `official/` no se editan ni se formatean.

## Git

- Commits en Conventional Commits con alcance: `feat(model)`, `fix(decoding)`, `refactor(features)`, `test`, `docs`, `build`, `ci`.
- Ramas `tipo/descripcion`, por ejemplo `feat/nuevo-modelo` o `fix/viterbi-empates`.
- PRs chicos, con la plantilla completa y squash merge. El título del PR es el commit final.
- Sin trailers de coautoría de agentes (`Co-Authored-By`) ni firmas de agente.
- Las actualizaciones de dependencias van en su propio PR e informan el cambio medido en las salidas (spellF1 de desarrollo antes y después).

## Modelo

- `mlp.predict` nunca lee `match.labels`, respeta `match.intervals` (no decodifica entre mitades) y fuerza `match.forced_dead` a `D`.
- Todo cambio que altere las salidas informa el spellF1 de desarrollo antes y después, y también macroF1.

## Datos

- Ningún dato de partido (tracking, video, etiquetas, extractos) entra al repositorio ni a un LLM alojado.
- Entrenar, ajustar y elegir variantes solo con datos externos con licencia.
- Las reservas (holdouts) se consultan una sola vez, con un candidato congelado.
- El guard (`guard.refuse_agent`) debe cubrir todo comando que lea el partido del desafío o una reserva.
