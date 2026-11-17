# Informe técnico

Pitch Dynamics · Equipo BOCABOCABOCA · More than a Hack 2027, Fase 1: posesión por frame

## 1. Resumen

El sistema asigna a cada uno de los 199.319 frames del video táctico una etiqueta `H` (posesión
local), `A` (posesión visitante), `D` (pelota muerta) o `X` (fuera de las mitades evaluadas).
Conforme a las reglas y a las aclaraciones de la organización, el modelo se entrena y se selecciona
exclusivamente con datos externos con licencia. Del partido del desafío se utilizan únicamente los
datos de tracking automático (ATD), como entrada en la etapa de predicción.

El pipeline consta de cuatro etapas:

1. **Completado de la pelota.** El ATD broadcast se alinea con la línea de tiempo táctica y su
   posición de pelota se utiliza en los frames tácticos donde la pelota no fue detectada.
2. **Features por frame.** Se calculan 99 variables de geometría, control y contexto a partir de las
   posiciones de jugadores y pelota.
3. **Clasificación.** Un ensamble de dos perceptrones multicapa estima la probabilidad de `H`, `A` y
   `D` en cada frame.
4. **Decodificación.** El algoritmo de Viterbi obtiene, para cada mitad por separado, la secuencia de
   etiquetas de máxima evidencia con un costo por cada cambio de estado.

## 2. Datos

| Fuente                                                                                        | Uso                                     | Licencia             |
| --------------------------------------------------------------------------------------------- | --------------------------------------- | -------------------- |
| DFL/IDSSE, 7 partidos de Bundesliga a 25 fps (Bassek et al., _Scientific Data_ 12, 195, 2025) | Entrenamiento, desarrollo y reserva DFL | CC-BY 4.0            |
| SkillCorner open data, 2 Clásicos de LaLiga a 10 fps                                          | Reserva LaLiga                          | MIT                  |
| ATD táctico del desafío                                                                       | Entrada en la predicción                | Licencia del desafío |
| ATD broadcast del desafío                                                                     | Completado de la pelota del ATD táctico | Licencia del desafío |

**Datos etiquetados.** El pipeline no utiliza etiquetas del partido del desafío ni etiquetas
manuales. Todas las etiquetas de entrenamiento y evaluación provienen de los proveedores:

| Conjunto                | Origen de la etiqueta                                                   | Volumen                                                  | Uso           |
| ----------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------- | ------------- |
| DFL/IDSSE, 7 partidos   | Proveedor: estado de la pelota y equipo en posesión en cada frame       | 1.002.644 frames: 293.931 `H`, 259.174 `A` y 449.539 `D` | Entrenamiento |
| SkillCorner, 2 Clásicos | Proveedor: equipo del jugador que controla la pelota, sin pelota muerta | 2 partidos                                               | Evaluación    |

El estado de la pelota y el equipo en posesión de DFL se traducen a `D`, `H` y `A` sin intervención
manual. La partición se fijó a partir de la lista de partidos, antes de procesar los datos:
desarrollo `J03WMX`, `J03WOH`, `J03WOY`, `J03WPY` y `J03WQQ`; reserva DFL `J03WN1` y `J03WR9`;
reserva LaLiga `skc_2841` y `skc_3442`.

**Verdad pública.** Los 5 minutos de verdad publicados por la organización no se utilizaron para
entrenar ni para seleccionar el modelo. En este pipeline se consultaron una única vez, con el
modelo ya congelado, para verificar su funcionamiento.

**Assets del desafío.** Se utilizan el ATD táctico y el ATD broadcast. La velocidad de pelota
informada por el ATD no es confiable, por lo que las velocidades se recalculan a partir de las
posiciones. El Smart Tagging no se utiliza: los datos externos de entrenamiento no tienen un
equivalente, y emplearlo sólo en la predicción introduciría una diferencia entre entrenamiento y
uso. La homografía y los videos no se utilizan.

## 3. Preprocesamiento

### 3.1 Degradación de los datos de entrenamiento

El tracking DFL es limpio, mientras que el ATD del desafío presenta pérdidas y ruido. Para que el
modelo no aprenda señales inexistentes en el partido, los partidos DFL se degradan con un perfil
fijado a partir de la documentación oficial, sin mediciones sobre el tracking del desafío:

- pelota ausente en el 40 % de los frames, en huecos de 2 s de duración media;
- ausencia del 8 % por jugador, en huecos de 1 s de duración media;
- ruido gaussiano de 0,2 m en la pelota y de 0,05 m en los jugadores;
- velocidades calculadas sólo a partir de posiciones.

Cada partido de entrenamiento se utiliza con tres realizaciones independientes de la degradación y
con los equipos local y visitante intercambiados, para que `H` y `A` se aprendan de forma simétrica.

### 3.2 Completado de la pelota desde el broadcast

Las dos grabaciones del partido tienen líneas de tiempo distintas y no se provee una tabla de
correspondencia. Durante una toma en vivo, el broadcast avanza a la misma tasa que el video táctico
(25 fps), de modo que `frame táctico = frame broadcast + desplazamiento`, con un desplazamiento
constante por tramo continuo de la grabación. Las repeticiones, la cámara lenta y los planos
cerrados no responden a ninguno de esos desplazamientos y se excluyen. La alineación procede en
cuatro pasos:

1. **Candidatos.** Para cada ventana de 30 s del broadcast, con avance de 15 s, se calcula la
   distancia cuadrática media entre las trayectorias de pelota de ambas grabaciones para todos los
   desplazamientos posibles, mediante correlaciones cruzadas por FFT. Se conservan los cinco
   desplazamientos de menor error.
2. **Verificación.** Un candidato se acepta si la mediana de la distancia de cada jugador del
   broadcast al jugador táctico más cercano del mismo equipo es inferior a 2 m. La verificación con
   jugadores es necesaria porque la proyección de una pelota en el aire sobre el plano de la cancha
   difiere entre cámaras.
3. **Votación.** Se retienen los desplazamientos aceptados por al menos cuatro ventanas.
4. **Asignación.** Cada frame del broadcast se asigna al desplazamiento cuyo residuo de jugadores
   cumple el umbral tanto en el propio frame como en la mediana de un segundo centrada en él. Los
   tramos de hasta un segundo sin jugadores suficientes se asignan cuando ambos extremos
   corresponden al mismo desplazamiento.

La posición de pelota del broadcast se incorpora únicamente en los frames tácticos sin pelota; las
detecciones tácticas existentes se conservan. Los umbrales se fijaron antes de procesar el partido
y el algoritmo se validó con datos sintéticos que incluyen repeticiones, planos cerrados y visión
parcial de la cancha. No se ajustó ningún parámetro con datos del partido.

En el partido, la alineación identificó dos desplazamientos principales, uno por tiempo, con 145 y
139 ventanas de soporte, y un tercero, a 8 frames del primero, con 4 ventanas. Se asignó el 59 % de
los frames del broadcast, con un residuo mediano de jugadores de 0,29 m. En los frames donde ambas
cámaras detectan la pelota, la diferencia entre posiciones es de 0,77 m en la mediana y de 5,5 m en
el percentil 90. La proporción de frames evaluados con pelota pasa de 59,3 % a 63,0 % (5.933 frames
adicionales).

## 4. Modelo

### 4.1 Features

Las 99 columnas por frame se calculan exclusivamente a partir de posiciones de jugadores y pelota:

- geometría y control: distancias de cada equipo a la pelota, pelota detenida, pelota fuera de la
  cancha, calidad y ausencia de la observación, en valor instantáneo y con medias centradas de 1 s
  y 4 s;
- memoria del último contacto en radios de 1, 1,5 y 2,5 m: equipo más cercano en el último contacto
  y tiempo transcurrido;
- golpes: aceleración de la pelota y equipo presente a menos de 3 m;
- presión: jugadores de cada equipo a menos de 5 y 10 m de la pelota;
- forma de los equipos: dispersión, avance hacia el arco rival y velocidad de la pelota en la
  dirección de ataque.

### 4.2 Clasificador y decodificación

El clasificador promedia las probabilidades de dos perceptrones con una capa oculta de 64 unidades
tanh, entrenados con semillas distintas mediante Adam durante 12 épocas, con lotes de 1.024 frames,
tasa de aprendizaje 0,003, regularización L2 de 0,0001 y pesos de clase balanceados. Se entrena con
uno de cada tres frames etiquetados.

La decodificación de Viterbi utiliza dos costos de transición, estimados como el logaritmo del
cociente entre las probabilidades de permanencia y de cambio en las etiquetas de entrenamiento: uno
para los cambios entre equipos y otro para las transiciones entre pelota viva y pelota muerta. Sin
esta etapa, la salida alterna entre equipos en cada disputa y genera posesiones espurias que
spellF1 penaliza.

### 4.3 Resultados

| Conjunto                                    | spellF1 |
| ------------------------------------------- | ------: |
| Desarrollo DFL, dejando un partido fuera    |   0,566 |
| Reserva DFL, 2 partidos                     |   0,540 |
| Reserva LaLiga, 2 Clásicos                  |   0,434 |
| Verdad pública de 5 minutos, sin completado |   0,439 |

Métrica calculada con el evaluador oficial. El valor de desarrollo es la media de los cinco
partidos, cada uno evaluado con un modelo entrenado sobre los otros cuatro y promediado sobre dos
semillas. Las reservas se consultaron una única vez, con el modelo congelado y entrenado sobre los
partidos de desarrollo (reserva DFL) o sobre los siete partidos DFL (reserva LaLiga). El mejor
baseline publicado obtiene 0,424 en la verdad pública.

## 5. Reglas de posesión implementadas

La verdad del desafío se deriva de tracking corregido: `D` cuando la pelota está muerta y, en otro
caso, el equipo poseedor. Las etiquetas DFL siguen la misma convención, de modo que el modelo la
aprende a partir de datos externos y no de reglas escritas a mano.

- **Pelota suelta y disputas.** La posesión se mantiene con el último equipo que controló la pelota
  hasta que el rival la controla. La memoria del último contacto y el costo de cambio de Viterbi
  penalizan las alternancias breves.
- **Reanudaciones.** El juego se considera detenido hasta que la pelota vuelve a estar en juego. Las
  señales de pelota detenida y fuera de la cancha alimentan al clasificador, y la transición entre
  pelota viva y muerta tiene un costo propio.
- **Pelota no detectada.** La ausencia de pelota no implica juego detenido. La ausencia es una
  variable más y el modelo aprende a sostener la posesión durante los huecos.
- **Tramo publicado `[110808, 113742)`.** Se etiqueta siempre como `D`.
- **Fuera de las mitades evaluadas.** Se etiqueta como `X`. Cada mitad se decodifica por separado.

## 6. Limitaciones

- **Pelota muerta.** En DFL el F1 de `D` es 0,97; en la verdad pública el modelo recupera el 25 % de
  los frames `D` y produce 24 posesiones frente a 17. La diferencia puede deberse a una definición
  distinta de pelota muerta entre proveedores o a una degradación de entrenamiento poco
  representativa del ATD del desafío; cinco minutos no permiten distinguir entre ambas causas.
- **Perfil de degradación.** Los parámetros de ruido y de duración de los huecos se fijaron sin
  mediciones sobre el partido y pueden diferir del ATD real.
- **Completado de la pelota.** No existen datos externos con licencia que combinen dos cámaras con
  etiquetas de posesión, por lo que el efecto del completado no pudo medirse. La pelota incorporada
  proviene de otra cámara, con un error distinto, y afecta al 3,7 % de los frames evaluados.
- **Convención de posesión en la reserva LaLiga.** SkillCorner identifica al jugador que controla la
  pelota y no marca pelota muerta, mientras que DFL y el desafío utilizan el equipo poseedor. El
  valor absoluto de esa reserva no es comparable con el del desafío.
- **Selección sobre desarrollo.** Los hiperparámetros se eligieron con los partidos de desarrollo,
  por lo que esa cifra es optimista.
- **Smart Tagging.** Los eventos del partido serían evidencia directa de pelota muerta, pero no
  existe un equivalente con licencia para el entrenamiento.

Como trabajo futuro se proponen: eventos externos equivalentes al Smart Tagging para detectar
pelota muerta, más partidos externos con licencia, en particular de LaLiga, y una definición
documentada de la convención de posesión del desafío.

## 7. Reproducción

Requisitos: Python 3.15 y [uv](https://docs.astral.sh/uv/). Los datos del desafío no forman parte
del repositorio. Los ATD de `tactical/` y `broadcast/` del bundle oficial deben ubicarse en `data/`
o en la carpeta indicada por `MTAH_DATA_DIR`.

```bash
uv sync --locked
uv run pitch-dynamics data
uv run pitch-dynamics predict
```

`predict` alinea el broadcast, completa la pelota táctica, entrena el modelo con los siete partidos
DFL, etiqueta el partido y escribe `artifacts/predict/submission.csv` junto con `provenance.json`
(commit, versiones, plataforma y resumen de la alineación). El archivo se valida con el validador
oficial. Las semillas y los hilos están fijos y la alineación es determinista, por lo que en una
misma plataforma el archivo resultante es idéntico byte a byte. `uv run pitch-dynamics align`
ejecuta sólo la alineación y escribe su reporte en `artifacts/align/`.
