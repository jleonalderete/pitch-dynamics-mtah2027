# Pitch Dynamics

Etiqueta cada frame del video táctico de un partido (199.319 frames) con `H` (posesión local), `A`
(visitante), `D` (juego detenido) o `X` (fuera de los tiempos evaluados), a partir del tracking
automático (ATD). Los modelos se entrenan solo con datos externos con licencia.

`CONVENTIONS.md` es obligatorio: leerlo y cumplirlo antes de cambiar código.

## Reglas del desafío

- Las etiquetas salen de un algoritmo. Ninguna etiqueta manual del partido, ni siquiera para validar.
- No comparar la salida con el video.
- No entrenar, ajustar ni calibrar con la verdad de posesión ni con el video. Entrenar y elegir solo con datos externos con licencia.
- No pasar datos del partido, videos ni extractos a LLMs alojados. No abrir `data/`, `.cache/` ni archivos de tracking o video.
- El repositorio debe regenerar exactamente el archivo entregado.

## Contratos de datos

- ATD: `video_frame = frame_id - 1`.
- Los slots son observaciones con equipo, no personas persistentes.
- Pelota ausente no significa juego detenido.
- Cada mitad se decodifica por separado.
- El tramo `[110808, 113742)` es siempre `D`.
- `X` queda fuera de las mitades evaluadas.
- El evaluador oficial de `official/` nunca se edita. spellF1 es la métrica de ranking; reportar también macroF1.

## Comandos

```bash
uv sync
uv run pytest
uv run pre-commit run --all-files
uv run pitch-dynamics --help
```

Los comandos `holdout`, `align` y `predict` leen datos protegidos y se niegan a correr dentro de un
agente.
