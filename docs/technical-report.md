# Informe técnico

Pitch Dynamics · More than a Hack 2027 · Fase 1: posesión por frame

## 1. Enfoque

La tarea es asignar a cada uno de los 199.319 frames del video táctico una etiqueta `H`, `A`, `D` o
`X`. Las reglas aclaradas por la organización prohíben etiquetar el partido a mano, comparar la
salida con el video y entrenar, ajustar o calibrar con la verdad de posesión o con datos del
partido. Por eso todos los modelos se entrenan y se eligen **sólo con datos externos con licencia**,
y el partido del desafío se usa únicamente para predecir.

El sistema tiene tres etapas:

1. **Preparación.** Se cargan partidos externos con tracking y etiquetas de posesión del proveedor,
   y se degradan para que se parezcan al tracking automático del desafío. En el partido del
   desafío, los huecos de la pelota táctica se completan con la pelota del ATD broadcast alineado.
2. **Clasificación por frame.** Un modelo estima, para cada frame, la probabilidad de `H`, `A` y
   `D` a partir de evidencia geométrica del tracking.
3. **Decodificación.** Viterbi elige, por mitad, la secuencia de etiquetas que maximiza la
   evidencia total menos un costo por cada cambio de estado. Los costos se cuentan en las
   transiciones de las etiquetas de entrenamiento. Sin este paso, la salida alterna entre equipos
   en cada disputa y spellF1 penaliza cada posesión falsa.

## 2. Datos

| Fuente                                                                                        | Uso                                              | Licencia                    |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------ | --------------------------- |
| DFL/IDSSE, 7 partidos de Bundesliga a 25 fps (Bassek et al., _Scientific Data_ 12, 195, 2025) | Entrenamiento (5 de desarrollo), reserva DFL (2) | CC-BY 4.0                   |
| SkillCorner open data, 2 Clásicos de LaLiga                                                   | Reserva LaLiga                                   | MIT                         |
| Metrica Sports sample data, juegos 1 a 3                                                      | Sólo entrenamiento del modelo híbrido            | Atribución a Metrica Sports |
| ATD táctico del desafío                                                                       | Entrada para predecir                            | Licencia del desafío        |
| ATD broadcast del desafío                                                                     | Sólo completa la pelota del ATD táctico          | Licencia del desafío        |

Cada frame DFL trae el estado de la pelota (viva o muerta) y el equipo en posesión; se traducen a
`D`, `H` y `A` sin intervención manual. Los splits se fijaron con la lista de partidos, antes de
leer datos: desarrollo `J03WMX`, `J03WOH`, `J03WOY`, `J03WPY`, `J03WQQ`; reserva DFL `J03WN1`,
`J03WR9`; reserva LaLiga `skc_2841`, `skc_3442`.

**Datos etiquetados.** Ninguna etiqueta viene del partido del desafío y el equipo no etiquetó
ningún partido. Todas las etiquetas son las que publica cada proveedor:

| Conjunto                 | Quién etiquetó                                                            | Cuánto                                                   | Uso                                      |
| ------------------------ | ------------------------------------------------------------------------- | -------------------------------------------------------- | ---------------------------------------- |
| DFL/IDSSE, 7 partidos    | El proveedor: estado de pelota y equipo en posesión de cada frame oficial | 1.002.644 frames: 293.931 `H`, 259.174 `A` y 449.539 `D` | Entrenamiento de todos los modelos       |
| SkillCorner, 2 Clásicos  | El proveedor: equipo del jugador que controla la pelota, sin `D`          | 2 partidos a 10 fps                                      | Sólo evaluación (reserva LaLiga)         |
| Metrica sample, 3 juegos | El proveedor: eventos del partido, de los que se derivan `H`, `A` y `D`   | 3 partidos                                               | Entrenamiento de `hybrid`, junto con DFL |

Los 714.273 frames de los 5 partidos de desarrollo se usan para elegir, y los 7 partidos completos,
para entrenar el modelo que etiqueta el partido. La verdad pública de 5 minutos no se usó para
entrenar ni para elegir. Se consultó para chequear una vez `mlp` y una vez `mlp_xgb`, ya
congelados, y para dos preguntas de definición que no puntuaron ninguna predicción: si hay
posesiones de menos de 3 s después de un tramo del rival (las hay, de 0,8 s a 2,8 s) y cómo se
relaciona con el baseline público del jugador más cercano.

**Assets del desafío usados y descartados.** Se usan el ATD táctico y, para completar su pelota, el
ATD broadcast. La velocidad de pelota del ATD no es confiable, así que las velocidades se
recalculan desde las posiciones. El Smart Tagging no se usa porque los datos externos no tienen un
equivalente: usarlo sólo al predecir crearía una diferencia entre entrenamiento y uso. La
homografía y los dos videos no se usan.

**Pelota completada desde el broadcast.** Las dos grabaciones no comparten línea de tiempo y no
hay tabla de correspondencia. Dentro de una toma en vivo el broadcast avanza a la par del táctico,
así que `frame táctico = frame broadcast + offset` con pocos offsets en todo el partido; las
repeticiones y los planos cerrados no encajan con ninguno. El comando `align`:

1. compara, para cada ventana de 30 s del broadcast, las dos pelotas en todos los desplazamientos
   posibles con correlaciones por FFT y se queda con los 5 mejores;
2. acepta un desplazamiento sólo si la mediana de la distancia de cada jugador del broadcast al
   compañero de equipo táctico más cercano es menor a 2 m (la pelota sola engaña: en el aire se
   proyecta distinto en cada cámara);
3. conserva los offsets que votan al menos 4 ventanas;
4. asigna a cada frame el offset que encajan sus jugadores, en el propio frame y en la mediana de
   un segundo alrededor, para que una repetición pegada a una toma en vivo nunca herede la
   asignación.

Después copia la pelota del broadcast sólo donde falta la táctica; una pelota táctica observada
nunca se pisa. Los umbrales se fijaron a priori y el algoritmo se probó con datos sintéticos; no
se ajustó nada con el partido ni con etiquetas. En el partido encontró dos offsets fuertes, uno
por tiempo (145 y 139 ventanas), y un tercero a 8 frames del primero con 4 ventanas, que
suponemos un salto corto de la señal. Asignó el 59 % de los frames broadcast con un residuo
mediano de jugadores de 0,29 m; donde las dos cámaras ven la pelota, difieren 0,77 m en la mediana
y 5,5 m en el percentil 90. La pelota en frames evaluados pasa de 59,3 % a 63,0 % (+5.933 frames):
las dos cámaras pierden la pelota casi en los mismos momentos.

**Degradación.** El tracking DFL es limpio y el del desafío no. Para no aprender señales que en el
partido no existen, los partidos DFL se degradan con un perfil fijado a priori, sin medir el
tracking del desafío: 40 % de pelota ausente en huecos de 2 s de media, 8 % de ausencia por
jugador, ruido de 0,2 m en la pelota y de 0,05 m en los jugadores, y velocidades calculadas sólo
desde posiciones.

## 3. Modelo

**Features.** 99 columnas por frame, todas calculables con el tracking del desafío:

- señales base de geometría y control: distancias de cada equipo a la pelota, pelota quieta, fuera
  de cancha, calidad y ausencia de la observación, crudas y con medias centradas de 1 s y 4 s;
- memoria del último toque en radios de 1, 1,5 y 2,5 m: qué equipo estuvo cerca por última vez y
  hace cuánto;
- golpes: aceleración de la pelota y qué equipo estaba a menos de 3 m;
- presión: jugadores de cada equipo a menos de 5 y 10 m de la pelota;
- forma de equipo: dispersión, avance de cada equipo hacia el arco rival y velocidad de la pelota
  en la dirección de ataque.

**Clasificador.** Promedia las probabilidades de dos redes de 64 unidades tanh entrenadas con
semillas distintas, Adam, 12 épocas y clases balanceadas. Cada partido se usa con 3 degradaciones
distintas y también con local y visitante intercambiados, para aprender `H` y `A` de forma
simétrica.

**Modelos comparados.** Durante el desarrollo se evaluaron otras cuatro variantes con la misma
preparación y la misma decodificación. Este repositorio sólo contiene el modelo entregado.

| Modelo     | Descripción                                                 | Desarrollo DFL | Reserva DFL | Reserva LaLiga |
| ---------- | ----------------------------------------------------------- | -------------: | ----------: | -------------: |
| `mlp`      | Ensamble de 2 redes de una capa sobre 99 features por frame |          0,566 |       0,540 |      **0,434** |
| `tcn`      | Red convolucional temporal sobre las mismas 99 features     |          0,613 |           - |          0,418 |
| `best_mix` | Mezcla de 4 redes temporales por partido de desarrollo      |          0,637 |           - |          0,433 |
| `hybrid`   | 75 % LightGBM y 25 % red temporal sobre 322 features        |              - |           - |          0,371 |
| `mlp_xgb`  | Las 2 redes de `mlp` más XGBoost, con 4 degradaciones       |          0,552 |           - |              - |

spellF1 con el evaluador oficial. Desarrollo: media dejando un partido afuera, con 2 semillas.
`tcn` sigue a Borghesi et al. (2023), _Expert Systems with Applications_ 223, 119780. `best_mix` no
tiene una versión entrenada con todos los partidos. `hybrid` se entrenó con los partidos de
desarrollo, así que no se puede medir en ellos. `mlp_xgb` se descartó porque sus degradaciones se
eligieron con estadísticas sin etiquetas del tracking del desafío, y la organización indicó que los
parámetros se ajustan sólo con datos externos.

**Cuál se entrega.** `mlp` con la pelota completada desde el broadcast. En desarrollo `tcn` y
`best_mix` rinden más, pero esa ventaja no se trasladó a la reserva LaLiga, el único conjunto de
un estilo de juego cercano al del desafío. La diferencia con `best_mix` en esa reserva está dentro
del ruido. Como chequeo único con el candidato ya congelado y sin completar la pelota, `mlp`
obtuvo spellF1 0,439 en los 5 minutos públicos, contra 0,424 del mejor baseline. La versión con la
pelota completada no se puntuó: el modelo es el mismo y sólo cambia la entrada en el 3,7 % de los
frames evaluados.

## 4. Reglas de posesión implementadas

La verdad del desafío se deriva del tracking corregido: `D` cuando la pelota está muerta y, si no,
el equipo dueño. Las etiquetas DFL siguen la misma lógica, así que los modelos aprenden esas
convenciones de datos externos en lugar de reglas escritas a mano.

- **Pelota suelta y disputas:** la posesión sigue con el último equipo que la controló hasta que el
  otro la controla. Lo expresan la memoria del último toque y el costo de cambio de Viterbi, que
  penaliza alternancias cortas.
- **Reinicios:** son `D` hasta que la pelota vuelve a estar en juego. Las señales de pelota quieta
  y fuera de cancha alimentan al clasificador, y la transición vivo↔`D` tiene su propio costo.
- **Pelota ausente no implica juego detenido:** la ausencia es una feature más y el modelo aprende
  a sostener la posesión durante los huecos.
- **Tramo publicado `[110808, 113742)`:** siempre `D`.
- **Fuera de las mitades evaluadas:** siempre `X`. Cada mitad se decodifica por separado.

## 5. Limitaciones

- **Juego detenido.** En DFL el F1 de `D` es 0,97; en los 5 minutos públicos `mlp` recupera sólo el
  25 % de los frames `D` y fragmenta más (24 posesiones contra 17). O "pelota muerta" se define
  distinto en los dos proveedores, o el perfil de degradación se parece poco al tracking del
  desafío. Con 5 minutos no se puede distinguir.
- **Convención de posesión.** SkillCorner marca al jugador que controla la pelota y DFL al equipo
  dueño. En la reserva LaLiga, el mismo modelo híbrido pasa de 0,371 a 0,515 según la convención:
  pesa más que la arquitectura.
- **Selección optimista.** Las variantes se eligieron sobre los mismos 5 partidos de desarrollo,
  así que esos números son optimistas. Las dos reservas ya se consultaron.
- **Sin Smart Tagging.** Los eventos (laterales, faltas, goles) serían la mejor evidencia de juego
  detenido, pero no hay un equivalente con licencia para entrenar.
- **Relleno sin validar.** Ningún dato externo con licencia trae dos cámaras con etiquetas, así que
  el efecto de completar la pelota no se pudo medir. `mlp` se entrenó con 40 % de pelota ausente y
  recibe un 37 %; la pelota completada tiene el ruido de otra cámara.

**Con más tiempo:** eventos externos equivalentes al Smart Tagging para atacar el juego detenido;
más partidos externos con licencia, idealmente de LaLiga, que darían una reserva nueva; y una
decisión documentada sobre la convención de posesión del desafío.

## 6. Reproducción

Requisitos: Python 3.15 y [uv](https://docs.astral.sh/uv/). Los datos del desafío no están en el
repositorio: se descargan aparte a `data/` o se indican con `MTAH_DATA_DIR`. Hacen falta los ATD de
`tactical/` y de `broadcast/`.

```bash
uv sync --locked
uv run pitch-dynamics data
uv run pitch-dynamics predict
```

`predict` alinea el broadcast y completa la pelota táctica, entrena con los 7 partidos DFL,
etiqueta el partido, escribe `artifacts/predict/submission.csv` y `provenance.json` (commit,
versiones, plataforma y resumen de la alineación) y corre el validador oficial. Las semillas y los
hilos están fijos y la alineación es determinista, así que en la misma plataforma el archivo es
idéntico byte a byte. `uv run pitch-dynamics align` corre sólo la alineación y deja su reporte en
`artifacts/align/`.
