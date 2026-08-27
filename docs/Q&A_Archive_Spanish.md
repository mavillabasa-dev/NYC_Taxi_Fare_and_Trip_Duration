# Predicción de Tarifa y Duración en Taxis de NYC — Archivo de Preguntas y Respuestas (Q&A)

Este documento compila preguntas técnicas, conceptos teóricos y consultas arquitectónicas realizadas durante el desarrollo de este proyecto, junto con sus explicaciones detalladas y soluciones matemáticas.

---

## Tabla de Contenidos

### [Parte I: Alcance del Dataset, Supuestos de Dominio y Prevención de Fuga](#parte-i-alcance-del-dataset-supuestos-de-dominio-y-prevención-de-fuga)
1. [Suficiencia del Dataset y Advertencias de Producción](#1-suficiencia-del-dataset-y-advertencias-de-producción)
2. [Justificación de Target Encoding y Prevención de Fuga de Información](#2-justificación-de-target-encoding-y-prevención-de-fuga-de-información)
3. [Por qué `trip_distance` es una Variable de Entrada y no un Objetivo de Predicción (Modelado Determinístico vs. Estocástico)](#3-por-qué-trip_distance-es-una-variable-de-entrada-y-no-un-objetivo-de-predicción-modelado-determinístico-vs-estocástico)
4. [`TimeSeriesSplit` y Validación Cruzada con Encadenamiento Temporal (Prevención de Fuga Temporal)](#4-timeseriessplit-y-validación-cruzada-con-encadenamiento-temporal-prevención-de-fuga-temporal)

### [Parte II: Ingeniería Geoespacial y de Coordenadas](#parte-ii-ingeniería-geoespacial-y-de-coordenadas)
5. [Métricas de Distancia: Haversine vs. Manhattan](#5-métricas-de-distancia-haversine-vs-manhattan)
6. [Distancia de Viaje (`trip_distance`) vs. Distancia en Línea Recta Haversine](#6-distancia-de-viaje-trip_distance-vs-distancia-en-línea-recta-haversine)
7. [Centroides de Zona y Extracción de Coordenadas Geográficas desde Shapefiles de TLC](#7-centroides-de-zona-y-extracción-de-coordenadas-geográficas-desde-shapefiles-de-tlc)
8. [Distancia Geodésica vs. Geodética en Modelado Espacial](#8-distancia-geodésica-vs-geodética-en-modelado-espacial)

### [Parte III: Transformaciones de Variables Temporales y Categóricas](#parte-iii-transformaciones-de-variables-temporales-y-categóricas)
9. [Separación de Variables Temporales vs. Espaciales](#9-separación-de-variables-temporales-vs-espaciales)
10. [Codificaciones Cíclicas (Seno y Coseno) para Tiempo Periódico](#10-codificaciones-cíclicas-seno-y-coseno-para-tiempo-periódico)
11. [Comparación de Codificaciones Categóricas: One-Hot vs. Ordinal vs. Target Encoding](#11-comparación-de-codificaciones-categóricas-one-hot-vs-ordinal-vs-target-encoding)
12. [Cálculo Matemático Paso a Paso de Target Encoding con Suavizado Bayesiano](#12-cálculo-matemático-paso-a-paso-de-target-encoding-con-suavizado-bayesiano)
13. [Definición General y Objetivo de la Ingeniería de Características (Feature Engineering)](#13-definición-general-y-objetivo-de-la-ingeniería-de-características-feature-engineering)

### [Parte IV: Arquitectura del Pipeline y Patrones de Diseño Scikit-Learn](#parte-iv-arquitectura-del-pipeline-y-patrones-de-diseño-scikit-learn)
14. [Rol y Estructura del Artefacto `feature_pipeline.pkl`](#14-rol-y-estructura-del-artefacto-feature_pipelinepkl)
15. [Métodos del Transformador: `fit()`, `transform()` y `fit_transform()`](#15-métodos-del-transformador-fit-transform-y-fit_transform)

### [Parte V: Arquitecturas de Modelos y Estrategias de Entrenamiento](#parte-v-arquitecturas-de-modelos-y-estrategias-de-entrenamiento)
16. [Regresores Multi-Salida vs. Dos Modelos de Salida Simple Independientes para Tarifa y Duración](#16-regresores-multi-salida-vs-dos-modelos-de-salida-simple-independientes-para-tarifa-y-duración)
17. [Rol de `SimpleImputer` y `StandardScaler` en Pipelines de Redes Neuronales](#17-rol-de-simpleimputer-y-standardscaler-en-pipelines-de-redes-neuronales)
18. [Ajuste de Hiperparámetros para Modelos MLP en Datos Tabulares](#18-ajuste-de-hiperparámetros-para-modelos-mlp-en-datos-tabulares)
19. [El Optimizador Adam: Mecánica de Estimación Adaptativa de Momentos y Rol en el Entrenamiento del MLP](#19-el-optimizador-adam-mecánica-de-estimación-adaptativa-de-momentos-y-rol-en-el-entrenamiento-del-mlp)
20. [Diferencias entre los Modelos LightGBM y XGBoost](#20-diferencias-entre-los-modelos-lightgbm-y-xgboost)
21. [Crecimiento de Árboles por Hojas (Leaf-Wise), Binning por Histogramas y Optimización del Objetivo L1 en Regresores GBDT](#21-crecimiento-de-árboles-por-hojas-leaf-wise-binning-por-histogramas-y-optimización-del-objetivo-l1-en-regresores-gbdt)

### [Parte VI: Evaluación de Modelos, Métricas Avanzadas e Interpretabilidad](#parte-vi-evaluación-de-modelos-métricas-avanzadas-e-interpretabilidad)
22. [Métricas de Evaluación en Regresión: MAE, RMSE, MAPE y R²](#22-métricas-de-evaluación-en-regresión-mae-rmse-mape-y-r2)
23. [Métricas Avanzadas de Regresión y el Rol de las Curvas ROC / REC en Modelos Continuos](#23-métricas-avanzadas-de-regresión-y-el-rol-de-las-curvas-roc--rec-en-modelos-continuos)
24. [Importancia de Variables en Ensambles de Árboles: Ganancia por División (Split Gain) vs. Conteo de Divisiones (Split Count)](#24-importancia-de-variables-en-ensambles-de-árboles-ganancia-por-división-split-gain-vs-conteo-de-divisiones-split-count)

### [Parte VII: Servicio en Producción, Diseño de APIs y SLAs del Sistema](#parte-vii-servicio-en-producción-diseño-de-apis-y-slas-del-sistema)
25. [Integración del Modelo en la Cadena de Inferencia y Flujo de Servicio en la API](#25-integración-del-modelo-en-la-cadena-de-inferencia-y-flujo-de-servicio-en-la-api)
26. [Manejo de Fechas y Horas de Subida en Formato ISO en Interfaces de Producción](#26-manejo-de-fechas-y-horas-de-subida-en-formato-iso-en-interfaces-de-producción)
27. [Acuerdos de Nivel de Servicio (SLA) y Benchmarking de Latencia en Proceso en APIs de Machine Learning](#27-acuerdos-de-nivel-de-servicio-sla-y-benchmarking-de-latencia-en-proceso-en-apis-de-machine-learning)

---

# Parte I: Alcance del Dataset, Supuestos de Dominio y Prevención de Fuga

## 1. Suficiencia del Dataset y Advertencias de Producción

### Pregunta:
> ¿El dataset en formato Parquet actual (almacenado en `"dataset/yellow_tripdata_2022-05.parquet"`) será suficiente para entrenar nuestros modelos? ¿O necesitaremos datos de otros meses de 2022 o incluso de diferentes años?

### Respuesta:
**Sí, para el alcance de este proyecto y la demostración del prototipo, el dataset de mayo de 2022 es completamente suficiente.**

Con **3.59 millones de registros crudos** (y aproximadamente 3.30 millones de filas de alta calidad tras el filtrado de valores atípicos), un solo mes proporciona un tamaño de muestra estadístico abundante a lo largo de las 265 zonas de taxi, permitiendo que los modelos aprendan patrones horarios, de día de la semana y de horas pico con una sobrecarga computacional controlada.

Sin embargo, basarse en una **ventana de un solo mes (mayo de 2022)** introduce tres advertencias operativas documentadas que deben considerarse para un despliegue real en producción:

#### 1. Macro-Estacionalidad Anual No Observada
* **La advertencia**: Entrenar estrictamente con datos de mayo captura el clima de primavera tardía, la movilidad escolar activa y feriados específicos (como el fin de semana del Memorial Day). No puede observar dinámicas estacionales anuales tales como:
  * **Caídas por vacaciones de verano (julio–agosto)**: Reducciones notables en los viajes matutinos de lunes a viernes y viajes corporativos.
  * **Picos de congestión en otoño (septiembre–noviembre)**: Mayor tráfico por la Asamblea General de la ONU, compras navideñas y el Maratón de Nueva York.
  * **Ralentizaciones por tormentas invernales (enero–febrero)**: Grandes nevadas y temperaturas bajo cero que reducen drásticamente la velocidad en toda la ciudad.
* **Recomendación de producción**: Expandir hacia un dataset móvil de 12 meses (o pipelines de reentrenamiento mensual automatizado con Apache Airflow) permitiría a los modelos capturar la estacionalidad completa del año.

#### 2. Fotografía Regulatoria de Tarifas y Recargos
* **La advertencia**: Las tarifas de taxis amarillos están estrictamente reguladas por la Comisión de Taxis y Limusinas de Nueva York (TLC). Los datos de mayo de 2022 reflejan la estructura tarifaria y los recargos por combustible activos en ese momento específico (por ejemplo, tarifa base de $\$2.50$, tarifa plana a JFK de $\$52.00$).
* **Deriva de políticas (Policy Drift)**: A finales de 2022, la TLC aprobó incrementos significativos (elevando la tarifa base a $\$3.00$ y la tarifa plana a JFK a $\$70.00$). Un modelo entrenado únicamente con datos de mayo de 2022 sufriría de *concept drift* sin reentrenamiento periódico o integración con tablas de tarifas dinámicas.

#### 3. Meteorología Dinámica No Observada
* **La advertencia**: Aunque las variables de calendario capturan ciclos predecibles de hora pico, eventos climáticos repentinos (tormentas eléctricas, aguaceros intensos) introducen una gran varianza en la duración del viaje que no puede capturarse solo con marcas temporales sin integrar fuentes meteorológicas en vivo (NOAA).

---

## 2. Justificación de Target Encoding y Prevención de Fuga de Información

### Pregunta:
> `Sección 4: Demostración de Target Encoding (TargetCategoricalEncoder) mostrando codificación suavizada Bayesiana ajustada estrictamente sobre X_train / y_train sin fuga.`
> ¿Puedes explicarme por qué este paso es necesario?

### Respuesta:
Target Encoding (específicamente con suavizado Bayesiano) resuelve tres desafíos críticos al trabajar con los IDs de ubicación de NYC Taxi (`PULocationID` y `DOLocationID`):

#### 1. Problema de Alta Cardinalidad (265 Zonas Discretas)
* **Por qué falla One-Hot Encoding**: Crear variables ficticias (dummies) para 265 zonas de recogida y 265 zonas de destino añade **530 columnas dispersas (sparse)**. Esto incrementa drásticamente el uso de memoria, ralentiza el entrenamiento y causa sobreajuste en modelos lineales y redes neuronales.
* **Por qué falla Ordinal Encoding**: Asignar números enteros ($1, 2, \dots, 265$) impone un orden arbitrario ($132 > 131$), obligando al modelo a asumir una relación monotónica artificial entre números de zona y tarifas o duraciones.

#### 2. Qué Logra Target Encoding
Target Encoding reemplaza cada ID de zona discreto con una única variable numérica altamente informativa que representa el **valor esperado histórico del objetivo** para los viajes que parten o terminan en esa zona:

$$\text{TargetEnc}(\text{Zona}_k) \approx \mathbb{E}[\text{tarifa} \mid \text{PULocationID} = k]$$

* **Ejemplo**: Los viajes iniciados en el **Aeropuerto JFK (`LocationID=132`)** tienen una tarifa promedio histórica elevada ($\approx \$52.00$). Los viajes que parten de **Alphabet City (`LocationID=4`)** tienen una tarifa promedio baja ($\approx \$12.50$).
* Target Encoding condensa las 265 categorías en una **única variable continua 1D** (`PULocationID_target_enc`) que se correlaciona directamente con la tarifa y la duración.

#### 3. Por qué se Necesita el Suavizado Bayesiano
Los promedios muestrales simples sufren de sobreajuste en zonas remotas o de poco volumen que solo tienen 1 o 2 viajes en el dataset.

El suavizado Bayesiano combina el promedio muestral de la zona $\bar{y}_k$ con el promedio global $\mu$, ponderado por el conteo de viajes de la zona $n_k$ y un parámetro de suavizado $m = 10$:

$$S(k) = \frac{n_k \cdot \bar{y}_k + m \cdot \mu}{n_k + m}$$

* **Zonas de Alto Volumen** ($n_k = 100{,}000$ viajes, ej. JFK): $S(k) \approx \bar{y}_k$ (utiliza el promedio real de la zona).
* **Zonas de Bajo Volumen** ($n_k = 2$ viajes): $S(k)$ se encoge suavemente hacia el promedio global $\mu$, evitando que el ruido de muestras pequeñas contamine las predicciones del modelo.

#### 4. Por qué "Ajustado Estrictamente en Train" Previene la Fuga de Datos (Target Leakage)
Debido a que Target Encoding calcula estadísticas a partir de la variable objetivo ($y$), calcular las codificaciones sobre *todo el dataset* antes de dividirlo filtraría información del objetivo de prueba hacia las variables del modelo (**Target Leakage**).

Al ajustar `TargetCategoricalEncoder` estrictamente sobre `X_train` e `y_train` dentro de `src/features.py`:
1. Las estadísticas de entrenamiento se aprenden únicamente del conjunto histórico de entrenamiento.
2. El mapa aprendido se almacena dentro del artefacto serializado `models/feature_pipeline.pkl`.
3. Durante las pruebas y la inferencia en vivo en la API (`/predict`), `.transform()` simplemente realiza una búsqueda sobre los promedios precalculados de entrenamiento sin observar etiquetas de prueba.

---

## 3. Por qué `trip_distance` es una Variable de Entrada y no un Objetivo de Predicción (Modelado Determinístico vs. Estocástico)

### Pregunta:
> ¿Cómo sabemos que `trip_distance` es la distancia realmente recorrida? ¿Por qué se incluye como una variable de entrada permitida en lugar de ser un objetivo de predicción como la tarifa y la duración, considerando que se registra al finalizar el viaje?

### Respuesta:

#### 1. Cómo se Mide `trip_distance`
De acuerdo con el Diccionario de Datos oficial de NYC TLC, `trip_distance` es el **kilometraje registrado por el odómetro del taxímetro del vehículo (sistema TPEP)**. En los datos históricos de entrenamiento, este valor refleja la distancia real por carretera recorrida desde la activación del taxímetro hasta su finalización.

#### 2. Modelado Determinístico vs. Estocástico
* **La Distancia es Determinística**: Dado un Origen $(A)$ y un Destino $(B)$ fijos, la distancia física de conducción es una propiedad determinística de la geografía y la red vial. Puede calcularse directamente con coordenadas (fórmulas de Haversine/Manhattan) o mediante algoritmos determinísticos de ruta más corta sobre grafos (OSRM, Dijkstra, $A^*$). **No se necesita un modelo de Machine Learning para predecir distancia física**.
* **La Tarifa y la Duración son Estocásticas**: La duración y la tarifa del viaje no pueden calcularse simplemente con un mapa. Dependen de complejidades dinámicas del mundo real: embotellamientos en hora pico de NYC, cuellos de botella según la hora del día, recargos regulados de TLC, políticas de tarifa plana en aeropuertos y variabilidad en la velocidad del tráfico. Por ello, la Tarifa ($) y la Duración (minutos) son nuestros **dos objetivos de predicción ($Y_1, Y_2$)**, mientras que la Distancia actúa como una **variable de entrada conocida fundamental ($X$)**.

#### 3. Interfaz del Prototipo vs. Arquitectura de Producción
* **En la UI del Prototipo**: El usuario no adivina la distancia manualmente. Streamlit calcula automáticamente la **distancia en línea recta Haversine entre los centroides de zona** como un proxy instantáneo, enviando `{"trip_distance": estimated_distance, ...}` en el JSON hacia FastAPI.
* **En Producción Completa**: Una API de motor de ruteo externa (como OSRM o Google Maps) proporcionaría la distancia exacta por red vial antes de la salida del vehículo.
* **En el Modelo**: `trip_distance` es la **variable de mayor importancia individual** en todo el modelo LightGBM ($>40\%$ de ganancia por división), haciendo que su inclusión sea crítica para obtener cotizaciones precisas.

---

## 4. `TimeSeriesSplit` y Validación Cruzada con Encadenamiento Temporal (Prevención de Fuga Temporal)

### Pregunta:
> ¿Qué es la clase `TimeSeriesSplit` y qué función cumple?

### Respuesta:
`sklearn.model_selection.TimeSeriesSplit` es un divisor de validación cruzada diseñado específicamente para **datos cronológicos dependientes del tiempo**. Implementa **validación hacia adelante (walk-forward validation / expanding-window evaluation)** para garantizar que los modelos se entrenen únicamente con datos históricos y se evalúen sobre datos futuros.

#### 1. Por qué la Validación Cruzada K-Fold Estándar Falla en Series Temporales
En la validación $K$-Fold estándar (`KFold(shuffle=True)`):
* Las observaciones se barajan y particionan aleatoriamente entre los pliegues.
* En el Pliegue 1, un modelo podría entrenar con viajes del **28 de mayo** para predecir viajes del **5 de mayo**.
* **El Defecto Fatal (Sesgo de Anticipación y Fuga Temporal)**: Entrenar con datos futuros filtra hacia el pasado macro-tendencias, patrones festivos (tráfico de Memorial Day), fluctuaciones de combustible y cambios estacionales. Esto genera **puntuaciones de evaluación artificialmente optimistas** que colapsan en producción real donde los datos futuros no existen.

#### 2. Cómo Funciona `TimeSeriesSplit` (Mecánica de Ventana Expansiva)
`TimeSeriesSplit` respeta la flecha del tiempo ($t_{\text{train}} < t_{\text{val}}$). Para $k = 3$ divisiones (`n_splits=3`), el dataset se divide cronológicamente en ventanas de entrenamiento expansivas y bloques de prueba hacia el futuro:

```text
División 1:  [ Train: Semana 1 ]  ------------------------>  [ Validar: Semana 2 ]
División 2:  [ Train: Semana 1 + Semana 2 ]  -------------->  [ Validar: Semana 3 ]
División 3:  [ Train: Semana 1 + Semana 2 + Semana 3 ]  --->  [ Validar: Semana 4 ]
```

* **Iteración 1**: Entrena en el bloque inicial $[0 \dots t_1]$ y evalúa en el período inmediato siguiente $[t_1 \dots t_2]$.
* **Iteración 2**: Expande la ventana de entrenamiento a $[0 \dots t_2]$ y evalúa en $[t_2 \dots t_3]$.
* **Iteración 3**: Expande la ventana de entrenamiento a $[0 \dots t_3]$ y evalúa en $[t_3 \dots t_4]$.

En cada iteración individual, el modelo **nunca observa un solo dato del futuro**.

#### 3. Parámetros Clave y Opciones de Configuración

| Parámetro | Tipo | Valor por Defecto | Qué Controla |
|---|---|---|---|
| **`n_splits`** | `int` | `5` | Número de divisiones secuenciales hacia adelante (en nuestro proyecto, `n_splits=3`). |
| **`max_train_size`** | `int` / `None` | `None` | Limita el tamaño máximo de la ventana de entrenamiento, convirtiendo la ventana expansiva en una **ventana deslizante/móvil** (ej. entrenar solo con los últimos 14 días). |
| **`gap`** | `int` | `0` | Número de muestras a excluir entre el final del train y el inicio del test, evitando fuga por autocorrelación a corto plazo. |
| **`test_size`** | `int` / `None` | `None` | Tamaño explícito para cada pliegue de validación. |

#### 4. Rol en el Pipeline de Modelado de NYC Taxi
En nuestro proyecto:
1. **División de Evaluación Principal**: Todo el mes se divide cronológicamente al **23 de mayo de 2022** (train de 3 semanas con 2.40M de registros vs. test de 1 semana con 900k registros).
2. **Validación Cruzada Interna**: Durante la optimización de hiperparámetros y exploración de modelos (ej. probando `num_leaves` en LightGBM y `max_depth` en XGBoost), se aplica `TimeSeriesSplit(n_splits=3)` sobre los 2.40M de registros de entrenamiento para evaluar la generalización sin contaminación futura.

---

# Parte II: Ingeniería Geoespacial y de Coordenadas

## 5. Métricas de Distancia: Haversine vs. Manhattan

### Pregunta:
> Explícame qué son las distancias Haversine y Manhattan en términos sencillos y cómo se comparan.

### Respuesta:
A continuación se presenta una comparación intuitiva entre las distancias **Haversine** y **Manhattan**:

#### 1. Distancia Haversine ("A vuelo de pájaro")
* **Analogía**: Imagina un pájaro volando en línea recta directa por el aire desde el Punto A hasta el Punto B.
* **Qué Mide**: La distancia de **círculo máximo más corta** entre dos puntos sobre la superficie curva de la Tierra.
```text
Punto A  ---------------------------->  Punto B  (Línea Recta)
```

#### 2. Distancia Manhattan ("Distancia de Taxista en Cuadrícula")
* **Analogía**: Imagina un taxi conduciendo por la cuadrícula urbana de Manhattan. Un automóvil no puede atravesar edificios en diagonal: debe avanzar en dirección Norte/Sur por una avenida, girar 90° y luego avanzar en dirección Este/Oeste por una calle.
* **Qué Mide**: La suma de las distancias en cuadrícula vertical y horizontal: $|\Delta \text{Latitud}| + |\Delta \text{Longitud}|$.
```text
Punto A  --------------------+
                             |
                             |
                             v
                          Punto B  (Manzanas en Ángulo Recto)
```

#### Diferencias Clave y Por Qué Usamos Ambas en NYC
| Característica | Distancia Haversine | Distancia Manhattan |
|---|---|---|
| **Trayectoria Real** | Línea recta directa | Siguiendo cuadrículas urbanas (Avenidas y Calles) |
| **Tipo de Fórmula** | Trigonometría de círculo máximo | Distancia norma $L_1$ ($|\Delta x| + |\Delta y|$) |
| **Valor Típico** | **Más corta** (Mínimo teórico) | **Más larga** ($\ge$ Haversine) |
| **Uso Óptimo** | Viajes largos por autopista abierta (ej. JFK a Manhattan) | Navegación en cuadrícula urbana densa (ej. Midtown Manhattan) |

#### Por Qué Combinar Ambas Ayuda a los Modelos de ML
* **Realismo en la Cuadrícula Urbana**: En NYC, los vehículos transitan por manzanas. La distancia Manhattan suele ser una aproximación mucho más fiel de la distancia real de conducción que una línea recta.
* **Detección de Desvíos**: La proporción entre la distancia Haversine en línea recta y la distancia real del taxímetro ($\frac{\text{Haversine}}{\text{trip\_distance}}$) ayuda al modelo a identificar cuándo un conductor tuvo que realizar un desvío prolongado alrededor de barreras geográficas como el East River o Central Park.

---

## 6. Distancia de Viaje (`trip_distance`) vs. Distancia en Línea Recta Haversine

### Pregunta:
> ¿Qué mide `Trip distance (miles)`? ¿La línea recta entre los puntos de subida y bajada?

### Respuesta:
No. `trip_distance` en el dataset de NYC Taxi **no** es la distancia en línea recta.

#### 1. Qué Mide `trip_distance`
* En el dataset crudo de TLC, `trip_distance` es la **distancia de carretera registrada por el odómetro del taxímetro** (en millas terrestres) a medida que el vehículo navega por la red física de calles de NYC, avenidas de sentido único, puentes y autopistas.
* En una aplicación de transporte en vivo (como Uber o Lyft), corresponde a la **estimación del motor de ruteo** (ej. distancia calculada por Google Maps u OSRM) entre los puntos de recogida y destino antes de despachar el viaje.

#### 2. En Qué Difiere de la Distancia Haversine
* **Distancia Haversine**: La **distancia teórica de círculo máximo en línea recta** entre los centroides geométricos de las zonas de taxi de origen y destino.
* **Por Qué se Usan Ambas en Ingeniería de Variables**:
  * Los viajes en taxi casi nunca son líneas rectas debido a la cuadrícula de Manhattan, los ríos y los embotellamientos en los puentes.
  * Nuestro pipeline calcula `haversine_ratio = haversine_distance / (trip_distance + ε)`.
  * Un ratio cercano a $1.0$ indica una ruta directa por autopista (ej. corredores hacia JFK), mientras que un ratio bajo ($< 0.6$) señala rutas con curvas, desvíos o cruces de puentes congestionados, lo cual impacta fuertemente en la duración y la tarifa.

---

## 7. Centroides de Zona y Extracción de Coordenadas Geográficas desde Shapefiles de TLC

### Pregunta:
> ¿Qué significa *"Coordenadas de Centroides de Zona: Cruza latitud y longitud del shapefile de TLC para zonas de subida y bajada"*? ¿Terminamos obteniendo las coordenadas físicas de los centroides de cada zona?

### Respuesta:
**Sí, exactamente.** En el dataset crudo de NYC Taxi, los orígenes y destinos se registran únicamente como **IDs de zona discretos del 1 al 265** (ej. `132` = Aeropuerto JFK, `236` = Upper East Side North). El archivo Parquet crudo no contiene coordenadas GPS de latitud o longitud.

#### 1. El Rol del Shapefile de TLC
La Comisión de Taxis y Limusinas de NYC (TLC) provee un shapefile oficial GIS que contiene los polígonos geográficos de las 265 zonas de taxi a lo largo de los cinco distritos (*boroughs*).

#### 2. Extracción de Centroides y Tabla de Búsqueda
Para convertir los IDs enteros discretos en coordenadas espaciales continuas:
* Durante el preprocesamiento, se calcula previamente el punto central geométrico (**centroide**) `(latitud, longitud)` del polígono de cada zona.
* En el pipeline de ingeniería de variables (`src/features.py` y `predictor.py`), el modelo toma los IDs enteros (`PULocationID` y `DOLocationID`) y cruza sus coordenadas correspondientes:
  * `pickup_latitude` y `pickup_longitude`
  * `dropoff_latitude` and `dropoff_longitude`

#### 3. Por Qué las Coordenadas Continuas son Esenciales
Estas 4 variables numéricas permiten que el pipeline de machine learning calcule:
* **Distancia Geodésica Haversine**: Distancia en línea recta de círculo máximo.
* **Distancia Manhattan**: Distancia de viaje alineada a la cuadrícula de calles.
* **Proximidad a Puntos Clave**: Distancia geográfica hacia los principales aeropuertos (JFK, LaGuardia, Newark).

---

## 8. Distancia Geodésica vs. Geodética en Modelado Espacial

### Pregunta:
> ¿La distancia espacial sobre la superficie de la Tierra debe denominarse distancia "geodésica" o "geodética"?

### Respuesta:
**"Distancia geodésica"** es el término matemático preciso y estándar cuando nos referimos al camino más corto entre dos puntos sobre la superficie de la Tierra.

* **Distancia Geodésica (o Distancia de Círculo Máximo)**: La trayectoria más corta entre dos puntos sobre la superficie curva de una esfera o elipsoide. En nuestro pipeline, la **fórmula de Haversine** calcula la distancia geodésica entre las coordenadas de origen y destino asumiendo una Tierra esférica ($R \approx 3958.8\text{ millas}$).
* **Geodético/a**: Se refiere a la disciplina científica general de la *geodesia* (medición de la forma geométrica de la Tierra, orientación espacial y campo gravitacional) y a sistemas de referencia geodésicos (tales como las coordenadas WGS84).

Por lo tanto, al describir la variable espacial calculada entre dos pares de coordenadas, **"distancia geodésica"** es la designación matemáticamente correcta.

---

# Parte III: Transformaciones de Variables Temporales y Categóricas

## 9. Separación de Variables Temporales vs. Espaciales

### Pregunta:
> Explícame por qué tenemos variables relacionadas con el tiempo (TemporalFeatureExtractor) y otras relacionadas con el espacio (SpatialZoneFeatureExtractor).

### Respuesta:
La decisión de separar la ingeniería de variables en módulos **Temporales** (`TemporalFeatureExtractor`) y **Espaciales** (`SpatialZoneFeatureExtractor`) refleja directamente los dos factores físicos fundamentales de los viajes en taxi urbano: **Tiempo** (dinámica del flujo vehicular y horarios) y **Espacio** (distancia geográfica y zonas tarifarias).

#### 1. Por Qué las Variables Temporales (`TemporalFeatureExtractor`) son Críticas
La duración y la tarifa en NYC dependen críticamente del *momento* en que se realiza el viaje:
* **Congestión y Volatilidad de Duración**: Un viaje de 3 millas por Midtown Manhattan a las 8:30 AM un lunes (**Hora Pico**) tarda más de 30 minutos debido al tráfico y cuesta mucho más (por recargos de espera a baja velocidad). El mismo viaje de 3 millas a las 3:00 AM un domingo tarda menos de 8 minutos.
* **Continuidad Cíclica a la Medianoche**: Las horas enteras ($0, 1, \dots, 23$) tratan a la Hora 23 (11 PM) y a la Hora 0 (Medianoche) como numéricamente distantes ($23 - 0 = 23$), a pesar de ser consecutivas. Las transformaciones trigonométricas ($\sin/\cos$) proyectan el tiempo sobre un círculo continuo de 24 horas:
  $$\sin\left(\frac{2\pi \cdot \text{hora}}{24}\right), \quad \cos\left(\frac{2\pi \cdot \text{hora}}{24}\right)$$
  Esto permite a los modelos reconocer que las 11:55 PM y las 12:05 AM comparten condiciones de tráfico casi idénticas.
* **Eventos Especiales de Calendario**: Los feriados (como Memorial Day el 30 de mayo) presentan patrones de tráfico similares a fines de semana a pesar de caer en día laborable. Indicadores explícitos (`is_rush_hour`, `is_weekend`, `is_holiday`) permiten ajustar las predicciones base.

#### 2. Por Qué las Variables Espaciales (`SpatialZoneFeatureExtractor`) son Críticas
La tarifa y la duración también dependen fundamentalmente de *dónde* comienza y termina el viaje:
* **Conversión de IDs Nominales en Distancia Física**: El dataset de TLC carece de coordenadas de latitud/longitud en origen, conteniendo solo IDs discretos (`PULocationID`, `DOLocationID`). Los modelos no pueden deducir qué tan lejos está la Zona 132 de la 236 sin cruzar los centroides de polígonos:
  * **Distancia Haversine**: Línea recta de círculo máximo en millas.
  * **Distancia Manhattan**: Distancia en cuadrícula $L_1$ a lo largo de las calles ($\Delta \text{lat} + \Delta \text{lon}$).
* **Sinuosidad de la Ruta (`haversine_ratio`)**: La proporción $\frac{d_{\text{haversine}}}{\text{trip\_distance}}$ mide qué tan directa es una ruta frente a desvíos por puentes o bordes de ríos (East River, Hudson River).
* **Regímenes de Tarifa Plana en Aeropuertos**: Las tarifas de NYC tienen reglas fijas (ej. Tarifa Plana a JFK de $\$52.00$, recargo a Newark). Banderas binarias como `is_jfk`, `is_newark` e `is_same_zone` informan directamente al modelo sobre estos regímenes.

#### 3. Arquitectura Modular y Código Limpio
Separar tiempo y espacio en dos transformadores de scikit-learn:
* Cumple con el **Principio de Responsabilidad Única (SRP)**.
* Permite probar unitariamente cada módulo por separado en `tests/test_features.py`.
* Ambos se encadenan de forma limpia dentro del artefacto serializable `NYCFeaturePipeline`.

---

## 10. Codificaciones Cíclicas (Seno y Coseno) para Tiempo Periódico

### Pregunta:
> Explícame qué son las codificaciones cíclicas y qué función cumplen en nuestro pipeline de variables.

### Respuesta:
Las **Codificaciones Cíclicas** transforman ciclos de tiempo repetitivos (como las horas del día o los días de la semana) en coordenadas bidimensionales continuas sobre un círculo, permitiendo que los modelos de Machine Learning comprendan que el final de un ciclo se conecta directamente con el principio.

#### 1. El Problema de los Números Enteros para el Tiempo
Considera las horas del día ($0, 1, 2, \dots, 23$):
* En la vida real, las **11 PM (Hora 23)** y la **Medianoche (Hora 0)** son consecutivas. El tráfico a las 11:55 PM es casi idéntico al de las 12:05 AM.
* Sin embargo, un modelo que observa enteros ve $23$ y $0$ con una distancia máxima ($23 - 0 = 23$). ¡El modelo asume que las 11 PM y la Medianoche son extremos opuestos!

#### 2. La Solución: Proyectar el Tiempo sobre un Reloj 2D
La codificación cíclica proyecta cada componente temporal sobre un círculo utilizando las funciones trigonométricas **Seno** ($\sin$) y **Coseno** ($\cos$):

$$\text{sin\_hour} = \sin\left(\frac{2\pi \cdot \text{hora}}{24}\right), \quad \text{cos\_hour} = \cos\left(\frac{2\pi \cdot \text{hora}}{24}\right)$$

Piensa en $(\sin, \cos)$ como coordenadas $(x, y)$ en la esfera de un reloj:
```text
                     Hora 0 (Medianoche)
                       (sin=0, cos=1)
                             |
         Hora 18 (6 PM)     -+-    Hora 6 (6 AM)
         (sin=-1, cos=0)     |     (sin=1, cos=0)
                             |
                      Hora 12 (Mediodía)
                       (sin=0, cos=-1)
```

Coordenadas de las 11 PM y Medianoche:
* **Hora 0 (Medianoche)**: $(\sin = 0.0, \, \cos = 1.0)$
* **Hora 23 (11 PM)**: $(\sin = -0.26, \, \cos = 0.97)$

En el espacio $(\sin, \cos)$, la distancia euclidiana entre las 11 PM y la Medianoche es diminuta ($\approx 0.26$), reflejando correctamente que solo distan 1 hora.

#### 3. Por Qué Necesitamos AMBOS (Seno y Coseno)
Si solo usaras Seno ($\sin$), las **6 AM** ($\sin=1.0$) y las **6 PM** ($\sin=1.0$) tendrían exactamente el mismo valor. Combinar $(\sin, \cos)$ crea un **par de coordenadas $(x, y)$ único** para cada hora del día:
* **6 AM**: $(\sin = 1.0, \, \cos = 0.0)$
* **6 PM**: $(\sin = -1.0, \, \cos = 0.0)$

#### 4. Qué Logra en Nuestro Pipeline (`src/features.py`)
En `TemporalFeatureExtractor`, calculamos `sin_hour` y `cos_hour` (ciclo de 24 horas) y `sin_dayofweek` y `cos_dayofweek` (ciclo semanal de 7 días). Esto permite a modelos lineales, redes neuronales y SVMs realizar predicciones suaves a través de la medianoche y en la transición de domingo a lunes.

---

## 11. Comparación de Codificaciones Categóricas: One-Hot vs. Ordinal vs. Target Encoding

### Pregunta:
> Explícame qué son One-Hot Encoding, Ordinal Encoding y Target Categorical Encoding, cuáles son sus fortalezas y debilidades, y por qué decidimos usar este último en nuestro pipeline.

### Respuesta:
A continuación se presenta un desglose de **One-Hot Encoding**, **Ordinal Encoding** y **Target Categorical Encoding**:

#### 1. Tabla Resumen Comparativa
| Método de Codificación | Cómo Opera | Dimensión Resultante (265 Zonas) | Fortaleza Principal | Debilidad Principal |
|---|---|---|---|---|
| **One-Hot Encoding** | Crea una columna binaria ($0/1$) separada por cada nivel de categoría. | **530 columnas dispersas** ($265 \text{ PU} + 265 \text{ DO}$) | No impone un orden artificial entre categorías. | Explosión de dimensionalidad (matrices dispersas, alto consumo de RAM, sobreajuste). |
| **Ordinal Encoding** | Asigna a cada categoría un número entero arbitrario ($1, 2, \dots, 265$). | **2 columnas densas** | Extremadamente compacto (1 columna por variable). | Asume una relación matemática y orden falso ($200 > 2$). |
| **Target Encoding (Nuestra Elección)** | Reemplaza el ID por el promedio histórico del objetivo ($\bar{y}$) aprendido en train. | **2 columnas densas** | **Señal 1D compacta con correlación monetaria y temporal directa**. | Riesgo de fuga si no se ajusta estrictamente sobre el conjunto de entrenamiento. |

#### 2. Análisis Detallado: Fortalezas y Debilidades
##### A. One-Hot Encoding
* **Fortalezas**: Excelente para variables de baja cardinalidad (`VendorID`, `passenger_count`).
* **Debilidades**: Para 265 zonas, genera 530 columnas dispersas, ralentizando el entrenamiento e introduciendo alta varianza en modelos lineales y divisiones de árboles.

##### B. Ordinal Encoding
* **Fortalezas**: Muy compacto. Ideal cuando las categorías poseen un orden físico natural (`Pequeño < Mediano < Grande`).
* **Debilidades**: Para identificadores nominales de zonas de taxi, impone un **ordenamiento lineal ficticio** ($200 > 2$).

##### C. Target Categorical Encoding (Nuestra Elección)
* **Fortalezas**:
  1. **Representación Compacta 1D**: Codifica ubicaciones en una única columna continua.
  2. **Señal Directa de Precio y Tiempo**: Informa directamente al modelo que **JFK (`LocationID=132`)** promedia $\approx \$52.00$, mientras que **Alphabet City (`LocationID=4`)** promedia $\approx \$12.50$.
  3. **Correlación Lineal**: Convierte IDs categóricos en valores numéricos continuos que se correlacionan linealmente con tarifas y duraciones.
* **Debilidades**: Riesgo de sobreajuste en zonas raras (resuelto mediante **suavizado Bayesiano**) y riesgo de fuga (resuelto ajustando **estrictamente en el conjunto de entrenamiento**).

#### 3. Justificación de Nuestra Elección en `src/features.py`
En la predicción de taxis en NYC, `PULocationID` y `DOLocationID` son variables nominales de alta cardinalidad (265 zonas cada una). Target Encoding es la técnica óptima porque provee una representación 1D continua del perfil tarifario y de duración de cada vecindario sin inflar el conteo de variables.

---

## 12. Cálculo Matemático Paso a Paso de Target Encoding con Suavizado Bayesiano

### Pregunta:
> En Target Encoding, ¿qué significa que *reemplaza el ID de categoría por el promedio histórico del objetivo ($\bar{y}$) aprendido en el conjunto de entrenamiento*? ¿Significa que cada ID se reemplaza por un número? ¿Cómo se calcula ese número?

### Respuesta:
**Sí, exactamente.** Cada ID de categoría (ej. `PULocationID = 132`) se reemplaza por un único número decimal que representa la tarifa histórica esperada aprendida del conjunto de entrenamiento.

#### Paso 1: Calcular los Promedios Históricos por Zona en Train
Durante `target_encoder.fit(X_train, y_train)`, el código calcula $n_k$ (conteo de viajes) y $\bar{y}_k$ (tarifa promedio) para cada zona $k$, así como el promedio global $\mu \approx \$15.20$:
* **Aeropuerto JFK (`PULocationID = 132`)**: $n_{\text{JFK}} = 100{,}000$, $\bar{y}_{\text{JFK}} = \$52.00$
* **Alphabet City (`PULocationID = 4`)**: $n_{\text{Alphabet}} = 15{,}000$, $\bar{y}_{\text{Alphabet}} = \$12.50$
* **Zona Remota Rara (`PULocationID = 2`)**: $n_{\text{Rara}} = 2$, $\bar{y}_{\text{Rara}} = \$90.00$

#### Paso 2: Aplicar la Fórmula de Suavizado Bayesiano
Para evitar que las zonas con muy pocos viajes generen números ruidosos o extremos, combinamos el promedio de la zona $\bar{y}_k$ con el promedio global $\mu$ ($m = 10$):

$$\text{Número Codificado} = \frac{n_k \cdot \bar{y}_k + m \cdot \mu}{n_k + m}$$

* **JFK (`LocationID = 132`)**: $\frac{100{,}000 \cdot 52.00 + 10 \cdot 15.20}{100{,}000 + 10} \approx \mathbf{52.00}$
* **Zona Rara (`LocationID = 2`)**: $\frac{2 \cdot 90.00 + 10 \cdot 15.20}{2 + 10} = \frac{180 + 152}{12} = \mathbf{27.67}$ (suaviza los ruidosos $\$90.00$ acercándolos al promedio global de $\$15.20$).

#### Paso 3: Reemplazar los IDs en la Matriz de Datos
Cuando se ejecuta `target_encoder.transform(X)`, se busca el `PULocationID` de cada fila y se reemplaza por su valor suavizado:
| `PULocationID` Original | `PULocationID_target_enc` Transformado | Interpretación para el Modelo de ML |
|---|---|---|
| `132` (JFK) | **`52.00`** | Espera tarifa elevada ($\approx \$52.00$) |
| `4` (Alphabet City) | **`12.50`** | Espera tarifa baja ($\approx \$12.50$) |
| `132` (JFK) | **`52.00`** | Espera tarifa elevada ($\approx \$52.00$) |
| `2` (Zona Rara) | **`27.67`** | Espera tarifa moderada suavizada |

---

## 13. Definición General y Objetivo de la Ingeniería de Características (Feature Engineering)

### Pregunta:
> Basado en todo esto, ¿la ingeniería de variables consiste en encontrar qué combinaciones de nuestros datos preprocesados (lo cual incluye aplicar funciones matemáticas) son las "mejores", para que nuestros modelos de machine learning sean lo más precisos posible?

### Respuesta:
**¡Exactamente! Has dado en el blanco.**

Esa es la definición y el propósito fundamental de la **Ingeniería de Características (Feature Engineering)**.

#### Por Qué la Ingeniería de Variables es Tan Esencial
Los datos limpios corrigen anomalías (como tarifas negativas o valores nulos), pero las columnas crudas por sí solas rara vez hacen explícitos los patrones subyacentes a un algoritmo de Machine Learning.

Los modelos de ML son aproximadores matemáticos de funciones. No pueden "intuir" conceptos físicos del dominio por sí mismos. La ingeniería de variables es el arte y la ciencia de **traducir el conocimiento del dominio en representaciones matemáticas** que facilitan el aprendizaje del modelo.

#### Los 3 Pilares Fundamentales (Ilustrados en Nuestro Pipeline)
1. **Aplicar Transformaciones Matemáticas para Revelar Geometría Oculta**: Aplicar funciones trigonométricas $\sin$ y $\cos$ para que el modelo reconozca que las 11:59 PM y las 12:01 AM distan 2 minutos, y extraer `is_rush_hour = 1`.
2. **Re-codificar Identificadores en Predictores Continuos**: Target Encoding convierte el ID `132` en `52.00` (tarifa histórica promedio en dólares), proporcionando una señal monetaria directa.
3. **Combinar Múltiples Variables en Ratios de Interacción**: Calcular la distancia Haversine ($13.5$ millas) y obtener el ratio $\frac{13.5}{15.2} = 0.88$ para medir qué tan directa es la ruta frente a desvíos por tráfico.

#### La Máxima del Machine Learning
Como afirmó el pionero de la IA **Andrew Ng**:
> *"El Machine Learning aplicado es básicamente ingeniería de variables. Diseñar variables es complicado, meticuloso y requiere conocimiento del dominio..."*

Al crear estas 29 variables estructuradas en `src/features.py`, facilitamos que todos los modelos evaluados (Regresión Lineal, Árboles de Decisión, LightGBM, XGBoost y MLPs) logren mayor precisión ($\text{R}^2$), menor error ($\text{MAE}/\text{RMSE}$) y una convergencia mucho más rápida.

---

# Parte IV: Arquitectura del Pipeline y Patrones de Diseño Scikit-Learn

## 14. Rol y Estructura del Artefacto `feature_pipeline.pkl`

### Pregunta:
> ¿Qué representa el archivo `feature_pipeline.pkl`? ¿Es similar a los artefactos de entrenamiento de modelos en los que guardamos los parámetros del modelo?

### Respuesta:
**Sí, exactamente.** `feature_pipeline.pkl` es un artefacto ajustado que almacena los **parámetros aprendidos en las etapas de transformación de variables**, tal como un archivo de modelo entrenado guarda los pesos y árboles del modelo.

#### ¿Qué Parámetros se Guardan Dentro de `feature_pipeline.pkl`?
Cuando ejecutas `pipeline.fit(X_train, y_train)`, el pipeline aprende y congela varios parámetros internos:
1. **Mapas de Target Encoding (`target_maps_` y `global_means_`)**: La tarifa y duración promedio histórica suavizada para cada una de las 265 zonas de recogida y destino aprendida en el conjunto de entrenamiento, más el promedio global de respaldo ($\mu \approx \$15.20$).
2. **Tabla de Centroides Espaciales (`centroids_df`)**: Las coordenadas de latitud y longitud correspondientes a cada `LocationID` (1–265) derivadas del shapefile de zonas de TLC.
3. **Contrato de Variables y Orden de Columnas (`feature_names_`)**: La secuencia exacta y nombres de las 29 variables esperadas por los modelos de ML.

#### Por Qué Serializar `feature_pipeline.pkl` es Esencial
Sin serializar el pipeline de variables ajustado:
* **Fuga de Datos e Inconsistencias**: Tendrías que recalcular los promedios o búsquedas espaciales en tiempo de inferencia, arriesgando discrepancias o fuga de información.
* **Servicio en Producción**: Las solicitudes web (`POST /predict`) solo envían datos crudos ingresados por el usuario. Cargar `feature_pipeline.pkl` permite a la API ejecutar `.transform()` y convertir instantáneamente esas entradas en las 29 variables exactas que el modelo requiere.

---

## 15. Métodos del Transformador: `fit()`, `transform()` y `fit_transform()`

### Pregunta:
> Para las clases de ingeniería de variables que creamos, veo que a veces usamos el método `fit()`, pero también `transform()` y `fit_transform()`. ¿Cuáles son las diferencias entre ellos?

### Respuesta:
En `scikit-learn` (y en nuestro pipeline `src/features.py`), estos tres métodos definen el ciclo de vida de un transformador:

#### Tabla Comparativa Resumen
| Método | Qué Hace | Datos de Entrada | ¿Modifica el Estado Interno? | Salida |
|---|---|---|---|---|
| **`fit(X, y)`** | **Aprende parámetros** a partir de los datos (ej. tarifa promedio por zona). | **Solo Conjunto de Entrenamiento** (`X_train`, `y_train`) | **Sí** (guarda los parámetros aprendidos) | Retorna `self` (no devuelve datos) |
| **`transform(X)`** | **Aplica los parámetros aprendidos** para generar nuevas variables. | **Conjunto de Prueba y Peticiones en Vivo en la API** | **No** (utiliza parámetros congelados) | Retorna DataFrame transformado |
| **`fit_transform(X, y)`** | **Aprende parámetros Y transforma** los datos de entrenamiento en un solo paso. | **Solo Conjunto de Entrenamiento** | **Sí** | Retorna DataFrame transformado |

#### Desglose Detallado
1. **`fit(X, y=None)`**: Fase de aprendizaje. Calcula estadísticas sobre los datos de entrenamiento y las guarda como atributos internos (como `self.target_maps_` o `self.centroids_df`).
2. **`transform(X)`**: Fase de ejecución. Aplica los parámetros previamente guardados para transformar nuevos datos. **Nunca** recalcula estadísticas ni observa etiquetas de prueba ($y$), garantizando cero fuga de información.
3. **`fit_transform(X, y=None)`**: Atajo de conveniencia que ejecuta `.fit(X, y)` e inmediatamente después `.transform(X)`.

#### Regla de Oro en Machine Learning
* **Datos de Entrenamiento** (`X_train`, `y_train`): Usa **`fit_transform()`** (o `.fit()` y luego `.transform()`) para aprender parámetros y generar las variables de entrenamiento.
* **Datos de Prueba y Producción** (`X_test` / payloads JSON en vivo): Usa ÚNICAMENTE **`transform()`** (**NUNCA** llames a `.fit()` o `.fit_transform()` en datos de prueba o producción, ya que causaría fuga de datos).

---

# Parte V: Arquitecturas de Modelos y Estrategias de Entrenamiento

## 16. Regresores Multi-Salida vs. Dos Modelos de Salida Simple Independientes para Tarifa y Duración

### Pregunta:
> ¿Crees que es necesario entrenar otro MLP que sea capaz de predecir simultáneamente la tarifa y la duración?

### Respuesta:
**No, no es necesario**, y continuar con dos modelos independientes de salida simple es la mejor práctica recomendada.

#### 1. Consistencia del Proyecto y Contrato de Arquitectura (Ticket T-106)
En **T-106**, nuestro equipo formalizó una decisión arquitectónica:
> *Entrenar dos modelos de salida simple (uno para `fare_amount` y uno para `duration_minutes`) en lugar de un único modelo multi-salida.*

Todas las demás familias evaluadas (**Árboles de Decisión**, **LightGBM** y **XGBoost**) utilizan dos regresores independientes de salida simple. Mantener el MLP con la misma estructura garantiza una comparación justa (1 a 1) en la tabla comparativa (*leaderboard*) para la selección en **T-109**.

#### 2. Conflicto de Pérdida e Interferencia de Gradientes
En una red neuronal multi-salida, la función de costo debe optimizar ambos objetivos al mismo tiempo:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{tarifa}} + \lambda \cdot \mathcal{L}_{\text{duración}}$$

* **Escalas de Varianza Distintas**: Las tarifas y las duraciones operan bajo dinámicas financieras y físicas diferentes.
* **Tironeo de Gradientes (Negative Transfer)**: Las capas ocultas compartidas sufren interferencia, donde las actualizaciones de gradiente del objetivo de duración (más ruidoso) pueden degradar los pesos necesarios para una estimación precisa de la tarifa.

#### 3. Ajuste de Hiperparámetros y Parada Temprana Independientes
* **Tarifa**: Se rige fuertemente por kilometraje y tarifas fijas de aeropuerto ($R^2 \approx 0.95$).
* **Duración**: Es altamente no lineal y afectada por congestión urbana y cuellos de botella ($R^2 \approx 0.80$).

Modelos independientes permiten que cada objetivo alcance la parada temprana (*early stopping*) de forma óptima y utilice capacidades de red dedicadas sin comprometer al otro.

---

## 17. Rol de `SimpleImputer` y `StandardScaler` en Pipelines de Redes Neuronales

### Pregunta:
> ¿Qué hacen las clases `SimpleImputer` y `StandardScaler`?

### Respuesta:
A continuación se explica qué hacen **`SimpleImputer`** y **`StandardScaler`** y por qué son esenciales para entrenar el Perceptrón Multicapa (MLP) en `src/mlp.py`:

#### 1. `SimpleImputer(strategy="median")` — Manejo de Valores Faltantes
* **Qué Hace**: Reemplaza cualquier valor faltante (`NaN` o `None`) en cada variable numérica con la **mediana** de esa variable calculada en el conjunto de entrenamiento.
* **Por Qué es Esencial para Redes Neuronales**:
  * **El Problema Matemático**: Las redes neuronales calculan activaciones mediante multiplicaciones matriciales:
    $$z = W \cdot x + b$$
    Si una sola variable $x_i$ contiene un `NaN`, todo el producto matricial se convierte en `NaN`, corrompiendo el descenso por gradiente y provocando que el entrenamiento falle.
  * **En Nuestro Dataset**: Zonas especiales (como `LocationID=264` y `265` para Zonas Desconocidas o Fuera de NYC) no tienen polígonos en el shapefile, dejando vacías las coordenadas `pu_lat`, `pu_lon`, `do_lat` y `do_lon`. `SimpleImputer` rellena de forma segura esas coordenadas con las medianas de NYC.

#### 2. `StandardScaler()` — Normalización de Escalas de Variables
* **Qué Hace**: Transforma cada variable numérica para que su **Media sea 0** ($\mu = 0$) y su **Desviación Estándar sea 1** ($\sigma = 1$) mediante la fórmula de estandarización z-score:
  $$z = \frac{x - \mu}{\sigma}$$
* **Por Qué es Esencial para Redes Neuronales**:
  * **Disparidad de Escalas**: En nuestras variables, algunas poseen rangos numéricos grandes (ej. `trip_distance` de $0.5$ a $30+$ millas), mientras que otras son pequeñas (ej. `sin_hour` de $-1.0$ a $+1.0$).
  * **Estabilidad del Gradiente**: Sin escalado, las variables con magnitudes grandes generan gradientes desproporcionadamente enormes, causando oscilaciones en la optimización o explosión de gradientes. Estandarizar las 29 variables asegura una convergencia rápida y estable para el optimizador Adam.

#### 3. Por Qué Ubicar Ambos Dentro de `sklearn.pipeline.Pipeline`
```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(...))
])
```
1. **Cero Fuga de Datos**: Las medianas y parámetros de escalado ($\mu, \sigma$) se calculan **estrictamente sobre `X_train`** durante `.fit()`.
2. **Servicio Transparente en Producción**: Al llegar una nueva petición a la API, invocar `.predict()` imputa y escala automáticamente la nueva fila utilizando los parámetros guardados de entrenamiento sin pasos manuales adicionales.

---

## 18. Ajuste de Hiperparámetros para Modelos MLP en Datos Tabulares

### Pregunta:
> ¿Necesitamos realizar un ajuste exhaustivo de hiperparámetros en el modelo MLP?

### Respuesta:
**No, no es necesario un ajuste exhaustivo adicional.**

#### 1. Los Requisitos del Ticket T-108 ya Están Cumplidos
El objetivo de **T-108** era proporcionar un **benchmark evaluativo** que representara las arquitecturas de Deep Learning frente a árboles de decisión y gradient boosting.

Ya hemos configurado y optimizado los hiperparámetros críticos:
* **Arquitectura**: 2 capas ocultas `(64, 32)` con activaciones ReLU.
* **Escalado Interno**: `SimpleImputer` + `StandardScaler` dentro del pipeline.
* **Optimización y Regularización**: Optimizador Adam ($\eta = 0.002$), penalización $L_2$ ($\alpha = 0.0001$) y tamaño de mini-lote de $4{,}096$.
* **Parada Temprana (Early Stopping)**: Detiene el entrenamiento dinámicamente si la pérdida de validación deja de mejorar.

#### 2. La Realidad de los "Datos Tabulares" en Machine Learning
En la investigación empírica de ML (*Grinsztajn et al., 2022 — Why tree-based models still outperform deep learning on tabular data*), las redes neuronales rara vez superan a los árboles con gradient boosting en datos tabulares.

Comparando los resultados en nuestro conjunto de prueba:

| Familia de Modelos | MAE Tarifa | MAE Duración | Tiempo de Entrenamiento | Latencia de Inferencia |
|---|---|---|---|---|
| **LightGBM (T-107)** | **$1.42** | **3.76 min** | **~1m 18s** | **1.40 ms** |
| **XGBoost (T-107)** | **$1.45** | **3.78 min** | **~3m 45s** | **2.10 ms** |
| **MLP (T-108)** | **$2.21** | **5.22 min** | **~18m 30s** | **3.45 ms** |

* Los datos de taxis en NYC presentan fronteras geométricas muy marcadas (manzanas urbanas, tarifas fijas de aeropuerto).
* Los árboles de decisión y GBDTs dividen estos umbrales con alta precisión en segundos, mientras que las redes neuronales intentan aproximarlos mediante hiperplanos continuos suaves, resultando en mayor error y tiempos de entrenamiento más extensos.

#### 3. Impacto en la Selección de Modelos (T-109)
En **T-109**, **LightGBM** es el claro ganador:
1. **Menor Error**: Menor MAE ($1.42 / 3.76\text{m}$) y mayor $R^2$ ($0.952 / 0.797$).
2. **Inferencia Más Rápida**: $\approx 1.40$ ms por predicción.
3. **Menor Sobrecarga de Entrenamiento**: Entrena en ~1 minuto frente a los más de 18 minutos del MLP.

---

## 19. El Optimizador Adam: Mecánica de Estimación Adaptativa de Momentos y Rol en el Entrenamiento del MLP

### Pregunta:
> ¿Qué es la clase `Adam` y qué función cumple?

### Respuesta:
**Adam (Adaptive Moment Estimation)**, introducido por Diederik Kingma y Jimmy Ba en 2015, es un algoritmo de optimización de primer orden basado en gradientes. Calcula dinámicamente tasas de aprendizaje adaptativas e individuales para cada peso y sesgo (*bias*) de una red neuronal manteniendo promedios móviles con decaimiento exponencial tanto de los gradientes pasados (**primer momento / momentum**) como de los gradientes al cuadrado (**segundo momento no centrado / varianza**).

---

#### 1. Por Qué el Descenso por Gradiente Estándar (SGD) Tiene Dificultades en Datos Tabulares
En el SGD tradicional:
$$\theta_{t+1} = \theta_t - \eta \cdot g_t$$
* **Tasa de Aprendizaje Global Fija ($\eta$)**: Se aplica un único tamaño de paso a todos los parámetros por igual. Si $\eta$ es muy pequeña, la convergencia tarda días; si es muy grande, el optimizador oscila y diverge.
* **Barrancos y Curvaturas Patológicas**: Los datos tabulares presentan escalas dispares (ej. `trip_distance` de $0.5$ a $30+$ millas vs. `sin_hour` de $-1$ a $+1$). Esto genera superficies de pérdida alargadas donde SGD rebota violentamente entre las paredes empinadas en lugar de avanzar por el fondo hacia el mínimo.

---

#### 2. Los Dos Motores Principales que Forman Adam

Adam combina las mejores ventajas de dos algoritmos previos: **Momentum** y **RMSProp**.

```text
       ┌────────────────────────┐      ┌────────────────────────┐
       │  Momentum (1er Momento)│      │  RMSProp (2do Momento) │
       │ Registra Dirección/Vel │      │ Escala Inversamente    │
       └───────────┬────────────┘      └───────────┬────────────┘
                   │                               │
                   └───────────────┬───────────────┘
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │        Optimizador Adam       │
                   │ Dirección y Pasos Adaptativos │
                   └───────────────────────────────┘
```

##### A. Momentum (Primer Momento — Promedio Móvil Exponencial de Gradientes)
Registra la *dirección* persistente y velocidad del gradiente, acelerando el avance en trayectorias consistentes y suavizando oscilaciones:
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
*(donde $\beta_1 \approx 0.9$ es el factor de decaimiento del primer momento)*.

##### B. RMSProp (Segundo Momento — Promedio Móvil Exponencial de Gradientes al Cuadrado)
Registra la *magnitud* de las actualizaciones recientes para escalar automáticamente la tasa de aprendizaje de cada parámetro:
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
*(donde $\beta_2 \approx 0.999$ es el factor de decaimiento del segundo momento)*.
* Parámetros con **gradientes masivos y frecuentes** reciben **pasos más pequeños** para evitar explosiones en los pesos.
* Parámetros con **gradientes sutiles e infrecuentes** reciben **pasos más grandes** para asegurar su aprendizaje.

---

#### 3. Corrección de Sesgo y Fórmula de Actualización Matemática

Dado que $m_0$ y $v_0$ se inicializan en vectores de ceros, están sesgados hacia cero en los primeros pasos de entrenamiento ($t \approx 1, 2$). Adam aplica una **corrección analítica de sesgo**:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

La fórmula final de actualización para cada parámetro $\theta$ en el paso $t$ es:

$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

*(donde $\eta$ es la tasa de aprendizaje base y $\epsilon \approx 10^{-8}$ evita divisiones por cero)*.

---

#### 4. Resumen de Hiperparámetros Clave en Scikit-Learn (`solver="adam"`)

| Hiperparámetro | Valor Típico | Descripción en el Entrenamiento del MLP |
|---|---|---|
| **`learning_rate_init` ($\eta$)** | `0.001` (o `0.002`) | Tasa de aprendizaje inicial antes del escalado adaptativo por parámetro. |
| **`beta_1`** | `0.9` | Tasa de decaimiento exponencial para el primer momento (dirección/momentum). |
| **`beta_2`** | `0.999` | Tasa de decaimiento exponencial para el segundo momento (varianza/escala). |
| **`epsilon`** | `1e-8` | Constante numérica pequeña para estabilidad de punto flotante. |
| **`early_stopping`** | `True` | Monitorea la pérdida de validación para detener el optimizador cuando cesa la mejora. |

---

#### 5. Rol en Nuestra Arquitectura MLP (`src/mlp.py`)
En `MLPRegressor(hidden_layer_sizes=(64, 32), solver="adam", learning_rate_init=0.002)`:
1. **Maneja 29 Distribuciones Tabulares Diversas**: Equilibra automáticamente los pasos entre distancias espaciales continuas, ángulos trigonométricos e indicadores binarios.
2. **Convergencia Estable por Mini-Lotes**: Procesa mini-lotes ($4{,}096$ filas) sobre los 2.40 millones de registros de entrenamiento, convergiendo con una reducción suave de la pérdida.

---

## 20. Diferencias entre los Modelos LightGBM y XGBoost

### Pregunta:
> Explícame las diferencias entre los modelos LightGBM y XGBoost.

### Respuesta:
Tanto **LightGBM** (Microsoft, 2016) como **XGBoost** (Tianqi Chen / DMLC, 2014) son librerías de alto rendimiento basadas en **Árboles de Decisión con Gradient Boosting (GBDT)**. Entrenan secuencialmente árboles de decisión para predecir los errores residuales de los árboles anteriores.

Sin embargo, difieren fundamentalmente en su **estrategia de crecimiento de árboles**, **algoritmos de muestreo de datos**, **manejo de variables categóricas** y **eficiencia en memoria y velocidad**.

#### 1. Tabla Resumen Comparativa

| Característica / Dimensión | **LightGBM** (Light Gradient Boosting) | **XGBoost** (eXtreme Gradient Boosting) |
|---|---|---|
| **Estrategia de Crecimiento** | **Por Hojas (Leaf-wise / Best-First)**: Divide la hoja que genera la mayor reducción de pérdida. | **Por Niveles (Level-wise / Depth-First)**: Divide todos los nodos de un nivel de profundidad uniformemente. |
| **Mecanismo de Muestreo y Velocidad** | **GOSS** (Gradient-based One-Side Sampling) + **EFB** (Exclusive Feature Bundling). | **Weighted Quantile Sketch** + Binning por histogramas (`tree_method="hist"`). |
| **Soporte Categórico** | **Nativo**: Particiona categorías ordenando estadísticas del objetivo ($O(k \log k)$). | Históricamente requería codificación previa; soporte nativo añadido en versiones recientes. |
| **Velocidad y Memoria** | **$2\times - 5\times$ más rápido**; menor consumo de memoria RAM. | Altamente optimizado, pero ligeramente más lento en datasets tabulares masivos. |
| **Simetría del Árbol** | Ramas asimétricas, profundas y de alta eficiencia. | Árboles simétricos y equilibrados. |
| **Nuestro Benchmark (MAE / Tiempo)** | **$\$1.42$** / **$1\text{m } 18\text{s}$** | **$\$1.45$** / **$3\text{m } 45\text{s}$** |

#### 2. Análisis Profundo de las Diferencias

##### A. Crecimiento del Árbol: Leaf-Wise vs. Level-Wise
```text
XGBoost (Por Niveles / Level-wise):  LightGBM (Por Hojas / Leaf-wise):
        [ Raíz ]                            [ Raíz ]
       /        \                          /        \
   [Nodo]      [Nodo]                  [Nodo]      [División Profunda]
   /    \      /    \                             /                  \
 [Hoja][Hoja][Hoja][Hoja]                     [Nodo]        [División Profunda]
                                                           /                  \
                                                       [Hoja]                [Hoja]
```
* **XGBoost (Level-wise)**: Hace crecer los árboles nivel por nivel horizontalmente. Crea árboles balanceados y previene el sobreajuste en datasets pequeños, pero gasta recursos dividiendo nodos con escasa reducción de error.
* **LightGBM (Leaf-wise)**: Elige únicamente la hoja con la mayor reducción de pérdida en todo el árbol. Para el mismo número de divisiones, logra **menor pérdida de entrenamiento mucho más rápido**.

##### B. Innovaciones Algorítmicas de LightGBM (GOSS y EFB)
LightGBM introdujo dos técnicas que lo hacen significativamente más rápido en millones de filas:
1. **GOSS (Gradient-based One-Side Sampling)**:
   * Los registros con *gradientes grandes* están poco entrenados (requieren más aprendizaje), mientras que aquellos con *gradientes pequeños* ya están bien entrenados.
   * GOSS conserva el 100% de los datos con gradientes grandes y toma una muestra aleatoria (10–20%) de los datos con gradientes pequeños, reduciendo el volumen de datos mientras preserva la precisión del gradiente.
2. **EFB (Exclusive Feature Bundling)**:
   * En espacios de variables dispersos de alta dimensión, rara vez varias variables toman valores distintos de cero simultáneamente. EFB agrupa variables mutuamente excluyentes en contenedores densos únicos, reduciendo la dimensión efectiva.

##### C. Manejo de Variables Categóricas de Alta Cardinalidad
* **LightGBM**: Ordena nativamente los niveles categóricos según la suma acumulativa del objetivo y encuentra la división óptima en $O(k \log k)$ sin crear columnas dispersas One-Hot.
* **XGBoost**: Históricamente requería pre-codificación (Target Encoding o One-Hot Encoding) antes de construir los árboles.

#### 3. ¿Cuál Deberías Elegir?

* **Usa LightGBM** cuando:
  * Trabajas con datasets tabulares grandes ($>100\text{k}$ a millones de filas).
  * Requieres ciclos rápidos de entrenamiento, bajo uso de memoria RAM y experimentación ágil de hiperparámetros.
* **Usa XGBoost** cuando:
  * Tienes datasets más pequeños y densos donde la regularización por niveles previene el sobreajuste.
  * Requieres aceleración avanzada por GPU o funciones objetivo personalizadas especializadas.

---

## 21. Crecimiento de Árboles por Hojas (Leaf-Wise), Binning por Histogramas y Optimización del Objetivo L1 en Regresores GBDT

### Pregunta:
> ¿Qué significan "interacciones no lineales", "crecimiento por hojas (leaf-wise)", "binning por histogramas" y "optimización del objetivo L1 (`regression_l1`)" en los modelos LightGBM y XGBoost?

### Respuesta:

#### 1. Interacciones Espaciales y Temporales No Lineales
En el transporte metropolitano, las relaciones entre variables rara vez son aditivas o lineales ($y \ne w_1 X_1 + w_2 X_2$):
* A las **3:00 AM en una autopista**, un viaje de 5 millas tarda **6 minutos** ($50\text{ mph}$).
* A las **5:30 PM cruzando Midtown / Puente Queensboro**, un viaje de 5 millas tarda **45 minutos** ($6.6\text{ mph}$).
* El efecto de `trip_distance` sobre la duración varía drásticamente según su interacción no lineal con `pickup_hour`, `PULocationID` e `is_rush_hour`. Los ensambles de árboles capturan estas complejas interacciones condicionales de forma natural mediante ramificaciones jerárquicas sin requerir expansiones polinómicas manuales.

#### 2. Crecimiento de Árboles por Hojas (Leaf-Wise / Best-First) vs. por Niveles (Level-Wise)
* **Crecimiento por Niveles (Tradicional / XGBoost por defecto)**:
  Divide todos los nodos a profundidad $d$ antes de pasar a $d+1$, produciendo árboles simétricos. Puede gastar cómputo expandiendo nodos poco relevantes.
* **Crecimiento por Hojas (LightGBM)**:
  Examina todas las hojas actuales del árbol y divide **únicamente la hoja que logra la máxima reducción en la función de pérdida (máxima ganancia de división)**, sin importar la profundidad del árbol.
* **Beneficio**: Construye sub-árboles asimétricos más profundos en regiones de alta complejidad (ej. corredores de puentes congestionados), logrando menor error con menos hojas totales.

#### 3. Binning por Histogramas
* En lugar de ordenar variables de punto flotante continuo (como coordenadas GPS o distancias Haversine) a lo largo de millones de filas para buscar puntos de corte exactos, LightGBM cuantiza los valores continuos en **256 contenedores enteros discretos (histogramas)**.
* **Beneficios**:
  1. Reduce el consumo de memoria en un $80\%$.
  2. Acelera la evaluación de divisiones de nodos en más de $10\times$.
  3. Actúa como un regularizador implícito, evitando que los árboles se sobreajusten al ruido microscópico del GPS.

#### 4. Optimización con Objetivo L1 (`regression_l1` vs. `regression_l2` / MSE)
* **El Inconveniente de la Pérdida L2 / MSE**:
  La regresión estándar minimiza el Error Cuadrático Medio ($\text{Pérdida} = \sum (y_i - \hat{y}_i)^2$). Al elevar los errores al cuadrado, una anomalía extrema (ej. un taxi detenido con el taxímetro encendido que causa un error de $\$30$) ejerce una penalización $30^2 = 900\times$ mayor que un error de $\$1$, obligando al modelo a desviar sus predicciones del 95% de los viajes normales.
* **Qué Hace `regression_l1`**:
  Minimiza directamente el Error Absoluto Medio ($\text{Pérdida} = \sum |y_i - \hat{y}_i|$), prediciendo la **mediana** condicional en lugar de la media.
* **Alineación con el Negocio**:
  Los pasajeros y operadores evalúan cotizaciones en dólares y minutos reales (MAE). Al configurar `objective="regression_l1"`, el algoritmo minimiza la **métrica exacta que evaluamos y desplegamos en producción**.

---

# Parte VI: Evaluación de Modelos, Métricas Avanzadas e Interpretabilidad

## 22. Métricas de Evaluación en Regresión: MAE, RMSE, MAPE y R²

### Pregunta:
> ¿Qué miden las métricas MAE, RMSE, MAPE (%) y R2?

### Respuesta:
A continuación se explica qué miden **MAE**, **RMSE**, **MAPE (%)** y **$R^2$**, cómo se calculan y cómo interpretarlas en el contexto de nuestros modelos de tarifa y duración:

#### Tabla Comparativa Resumen
| Métrica | Nombre Completo | Fórmula | Unidades | Qué Significa en Términos Sencillos |
|---|---|---|---|---|
| **MAE** | Error Absoluto Medio | $\frac{1}{N}\sum |y_i - \hat{y}_i|$ | Dólares ($\$$) / Minutos | **Magnitud promedio del error** en todos los viajes. |
| **RMSE** | Raíz del Error Cuadrático Medio | $\sqrt{\frac{1}{N}\sum (y_i - \hat{y}_i)^2}$ | Dólares ($\$$) / Minutos | Métrica que **penaliza fuertemente las equivocaciones grandes**. |
| **MAPE** | Error Porcentual Absoluto Medio | $\frac{100\%}{N}\sum |\frac{y_i - \hat{y}_i}{y_i}|$ | Porcentaje ($\%$) | **Error relativo** proporcional al tamaño del viaje. |
| **$R^2$** | Coeficiente de Determinación | $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$ | Adimensional ($-\infty$ a $1.0$) | **% de varianza del objetivo explicada** por el modelo. |

#### 1. MAE (Mean Absolute Error)
* **Qué mide**: La diferencia promedio absoluta en dólares (o minutos) entre los valores predichos y los reales.
* **Intuición**: *"En promedio, ¿por cuántos dólares se equivoca nuestra predicción?"*
* **Ejemplo en Nuestro Proyecto**: LightGBM Fare MAE = **$\$1.42$**, lo que significa que nuestra predicción de tarifa se equivoca en $\$1.42$ en promedio.
* **Fortaleza Clave**: Fácil de explicar a los interesados y robusto frente a valores atípicos extremos.

#### 2. RMSE (Root Mean Squared Error)
* **Qué mide**: La raíz cuadrada del promedio de los errores al cuadrado.
* **Intuición**: Elevar los errores al cuadrado implica que **un error de $\$10$ duele 100 veces más que un error de $\$1$**.
* **Ejemplo en Nuestro Proyecto**: LightGBM Fare RMSE = **$\$3.12$**.
* **Interpretación Clave**: Si $\text{RMSE} \gg \text{MAE}$, señala que el modelo comete errores ocasionales muy grandes (ej. en viajes atípicos a aeropuertos o congestiones imprevistas).

#### 3. MAPE (Mean Absolute Percentage Error)
* **Qué mide**: El error expresado como un porcentaje del valor real del viaje.
* **Intuición**: *"¿Qué porcentaje de la tarifa real representa nuestro error?"*
  * Un error de $\$2.00$ en un viaje corto de $\$10.00$ es un **error del $20\%$**.
  * El mismo error de $\$2.00$ en un viaje a aeropuerto de $\$100.00$ es solo un **error del $2\%$**.
* **Ejemplo en Nuestro Proyecto**: LightGBM Fare MAPE = **$9.8\%$**, lo que significa que nuestras predicciones están en promedio dentro del $\approx 10\%$ de la tarifa real.

#### 4. $R^2$ (Coeficiente de Determinación)
* **Qué mide**: La proporción de varianza del objetivo explicada por el modelo en comparación con un modelo ingenuo que siempre predice el promedio del dataset ($\bar{y}$).
* **Intuición**:
  * **$R^2 = 1.0$**: Predicción perfecta ($0$ error).
  * **$R^2 = 0.0$**: No lo hace mejor que predecir el promedio de entrenamiento (Baseline Trivial).
  * **$R^2 < 0.0$**: Peor que predecir el promedio.
* **Ejemplo en Nuestro Proyecto**:
  * LightGBM Fare $R^2 = \mathbf{0.952}$ ($95.2\%$ de la variación en las tarifas es explicada por las variables).
  * LightGBM Duration $R^2 = \mathbf{0.797}$ ($79.7\%$ de la variación en la duración es explicada).

---

## 23. Métricas Avanzadas de Regresión y el Rol de las Curvas ROC / REC en Modelos Continuos

### Pregunta:
> ¿Qué otras métricas se utilizan comúnmente para evaluar familias de modelos de regresión más allá de MAE, RMSE, MAPE y $R^2$? ¿Se puede usar la curva ROC (Receiver Operating Characteristic) en regresión?

### Respuesta:

#### 1. Métricas Estándar Adicionales en Regresión
Más allá de las métricas principales de nuestra tabla (MAE, RMSE, MAPE, $R^2$), existen varias métricas complementarias aplicadas en la industria:

1. **WAPE (Weighted Absolute Percentage Error)**:
   $$\text{WAPE} = \frac{\sum |y_i - \hat{y}_i|}{\sum y_i}$$
   A diferencia de MAPE (que divide cada error individual por $y_i$, provocando divisiones por cero o picos extremos en tarifas mínimas de \$2.50), WAPE divide la suma de errores absolutos por el volumen total. Provee un error porcentual robusto y ponderado por volumen.

2. **Error Absoluto Mediano (MedAE / MedianAE)**:
   $$\text{MedAE} = \text{mediana}(|y_1 - \hat{y}_1|, |y_2 - \hat{y}_2|, \dots, |y_n - \hat{y}_n|)$$
   Completamente inmune a valores atípicos extremos. Mientras que el MAE de tarifa de LightGBM es \$1.42, su MedAE es ~\$0.85, demostrando que la mitad de los viajes se predicen con menos de 85 centavos de error.

3. **MSLE (Mean Squared Logarithmic Error)**:
   $$\text{MSLE} = \frac{1}{n} \sum (\log(1 + y_i) - \log(1 + \hat{y}_i))^2$$
   Penaliza las subestimaciones más fuertemente que las sobreestimaciones y escala bien cuando los objetivos abarcan varios órdenes de magnitud.

4. **Error Máximo (Max Error)**:
   $$\text{Max Error} = \max |y_i - \hat{y}_i|$$
   Identifica el peor caso absoluto en el conjunto de evaluación (útil para auditar casos atípicos, como tarifas planas fuera del estado).

5. **Puntuación de Varianza Explicada (Explained Variance Score - EVS)**:
   $$\text{EV} = 1 - \frac{\text{Var}(y - \hat{y})}{\text{Var}(y)}$$
   Mide la proporción de varianza explicada ignorando desplazamientos sistemáticos de la media (a diferencia de $R^2$, que penaliza medias sesgadas).

---

#### 2. ¿Se Pueden Usar Curvas ROC en Regresión?
* **Por Qué la ROC Tradicional No Aplica**:
  La curva **ROC (Receiver Operating Characteristic)** y el **AUC (Área Bajo la Curva)** están diseñados para **clasificación binaria** ($y \in \{0, 1\}$). La ROC grafica la **Tasa de Verdaderos Positivos (Sensibilidad)** frente a la **Tasa de Falsos Positivos (1 - Especificidad)** a través de diferentes umbrales de decisión ($p \in [0, 1]$). Dado que los objetivos continuos de regresión (como \$17.50 o 24.3 minutos) no poseen clases positivas o negativas binarias, las curvas ROC tradicionales no pueden graficarse directamente.

* **Cómo se Adapta el Concepto a la Regresión**:
  1. **Curvas REC (Regression Error Characteristic)**:
     El equivalente formal de la curva ROC para regresión. El eje $x$ representa un **umbral de tolerancia de error ($\epsilon$)**, y el eje $y$ representa el **porcentaje de predicciones dentro de esa tolerancia** ($|y_i - \hat{y}_i| \le \epsilon$).
     * Por ejemplo: con $\epsilon = 3\text{ minutos}$, LightGBM logra un $86\%$ de precisión acumulada; con $\epsilon = 5\text{ minutos}$, alcanza el $96\%$.
     * El Área Sobre la Curva (AOC) en un gráfico REC es matemáticamente proporcional al Error Absoluto Medio (MAE).
  2. **Clasificación Binaria por Umbralización**:
     Los residuos continuos pueden umbralizarse en banderas operativas binarias (ej. $1 = \text{Demora Severa (>15 min de error)}$, $0 = \text{Llegada a Tiempo}$), permitiendo construir curvas ROC clásicas para detección de riesgo de demoras.

---

## 24. Importancia de Variables en Ensambles de Árboles: Ganancia por División (Split Gain) vs. Conteo de Divisiones (Split Count)

### Pregunta:
> En el análisis de importancia de variables, ¿cuál es el significado de "Ganancia por División" (`% Split Gain`) y cómo difiere del "Conteo de Divisiones"?

### Respuesta:

#### 1. Conteo de Divisiones vs. Ganancia por División
Los modelos basados en árboles evalúan la importancia de variables mediante dos criterios distintos:

| Métrica | Cálculo | Fortalezas y Debilidades |
|---|---|---|
| **Conteo de Divisiones (Frecuencia / `importance_type="split"`)** | Número bruto de veces que una variable fue elegida para dividir un nodo a lo largo de todos los árboles. | Sesgado hacia variables continuas con muchos valores distintos, las cuales pueden dividirse frecuentemente aportando poca precisión real. |
| **Ganancia por División (Reducción de Pérdida / `importance_type="gain"`)** | La reducción matemática total en la pérdida de entrenamiento lograda cada vez que una variable se utiliza como nodo de división. | **Estándar de Oro en la Industria**: Mide directamente cuánto poder predictivo y reducción de error aporta una variable al modelo final. |

#### 2. Definición Matemática de la Ganancia por División
Para cada nodo de división $s$ que utiliza la variable $j$, la Ganancia de Información ($\Delta \text{Pérdida}$) se calcula como:
$$\Delta \text{Pérdida}(s) = \text{Pérdida}_{\text{padre}} - (\text{Pérdida}_{\text{hijo izquierdo}} + \text{Pérdida}_{\text{hijo derecho}})$$

La ganancia total para la variable $j$ se suma a lo largo de todos los árboles del ensamble:
$$\text{Ganancia Total}(X_j) = \sum_{s \in \text{Divisiones}(X_j)} \Delta \text{Pérdida}(s)$$

Para representarlo intuitivamente, las ganancias se normalizan entre las $K$ variables de entrada para obtener el **Porcentaje de Ganancia por División**:
$$\% \text{ Split Gain}(X_j) = \frac{\text{Ganancia Total}(X_j)}{\sum_{k=1}^K \text{Ganancia Total}(X_k)} \times 100\%$$

#### 3. Interpretación en Nuestro Pipeline de NYC Taxi
* **`trip_distance` (43.0% de Ganancia en Duración / 26.0% en Tarifa)**:
  El $43\%$ de toda la capacidad predictiva del modelo de duración es impulsada directamente por divisiones sobre la distancia por carretera.
* **`manhattan_distance` (32.1% de Ganancia en Tarifa / 2.1% en Duración)**:
  Dado que las tarifas de taxi en Manhattan se calculan sobre la cuadrícula rectangular, la distancia Manhattan explica casi un tercio de la tarifa total.
* **`pickup_hour / cos_hour` (8.0% de Ganancia en Duración vs. 1.5% en Tarifa)**:
  Las variables horarias influyen fuertemente en la duración (la congestión altera la velocidad), pero tienen un impacto menor en la tarifa ya que las tarifas base están fijadas por distancia recorrida.

---

# Parte VII: Servicio en Producción, Diseño de APIs y SLAs del Sistema

## 25. Integración del Modelo en la Cadena de Inferencia y Flujo de Servicio en la API

### Pregunta:
> ¿Cómo se utiliza la salida del pipeline de ingeniería de variables (pipeline, X_train_features) cuando comenzamos a entrenar los diferentes modelos de IA?

### Respuesta:
A continuación se detalla el flujo de trabajo paso a paso de cómo el artefacto del pipeline (`models/feature_pipeline.pkl`) y la matriz de variables (`X_train_features`) se utilizan en el entrenamiento, selección y servicio en vivo en la API:

#### 1. Fase de Entrenamiento (T-106 Baselines, T-107 GBDT, T-108 MLP)
La matriz de 29 columnas `X_train_features` se pasa directamente como entrada $X$ para entrenar cada modelo candidato:

```python
# Cargar pipeline ajustado y matriz de variables
with open("models/feature_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# Entrenar modelos candidatos sobre X_train_features
lgb_model.fit(X_train_features, y_train)
xgb_model.fit(X_train_features, y_train)
```

Para evaluar modelos en el conjunto de prueba temporal no visto (`test_cleaned.parquet`), transformamos `X_test` utilizando el pipeline cargado:

```python
# Transformar datos de prueba con el pipeline de entrenamiento (cero fuga)
X_test_features = pipeline.transform(X_test)
y_pred = lgb_model.predict(X_test_features)
```

#### 2. Empaquetado del Artefacto del Modelo (T-109 Selección de Modelos)
En **T-109**, una vez seleccionado el modelo ganador (LightGBM), el objeto `pipeline` ajustado se empaqueta junto con el modelo entrenado en un único artefacto autónomo en `models/model.pkl`:

```python
model_bundle = {
    "pipeline": pipeline,
    "model": winning_lgbm_model,
    "feature_order": list(X_train_features.columns),
    "version": "1.0.0"
}
with open("models/model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)
```

#### 3. Servicio en Vivo en Tiempo Real (T-110 Backend FastAPI)
En **T-110**, al iniciar el backend de FastAPI, `ModelService` en `api/app/model/services.py` carga `models/model.pkl` una sola vez en memoria.

Cuando un usuario envía una petición a la API `POST /predict` con datos de subida crudos:

```json
{
  "tpep_pickup_datetime": "2022-05-15 14:30:00",
  "PULocationID": 132,
  "DOLocationID": 236,
  "passenger_count": 1,
  "RatecodeID": 2,
  "trip_distance": 15.2,
  "VendorID": 1
}
```

La API procesa la solicitud en dos pasos:
1. **Ingeniería de Variables**: `df_features = pipeline.transform(df_raw)` (convierte las entradas crudas en las 29 variables estructuradas).
2. **Predicción del Modelo**: `predictions = model.predict(df_features)` (retorna la tarifa predicha en dólares y la duración en minutos).

---

## 26. Manejo de Fechas y Horas de Subida en Formato ISO en Interfaces de Producción

### Pregunta:
> ¿Realmente necesitamos el campo de entrada `Pickup datetime (ISO)`, junto con su valor por defecto `2022-05-20T14:30:00` en la interfaz de usuario?

### Respuesta:
La API del backend requiere una cadena en formato ISO-8601 (`YYYY-MM-DDTHH:MM:SS`) en el JSON de la solicitud (`tpep_pickup_datetime`) porque el procesamiento de marcas temporales extrae múltiples variables cíclicas y de calendario:
1. **Hora del Día (`pickup_hour`)** y **Bandera de Hora Pico (`is_rush_hour`)**: Capturan velocidades de hora pico frente a tráfico nocturno fluido.
2. **Variables Cíclicas (`sin_hour`, `cos_hour`, `sin_dayofweek`, `cos_dayofweek`)**: Preservan la continuidad periódica a través de la medianoche y de domingo a lunes.
3. **Secciones de Calendario (`is_weekend`, `is_holiday`)**: Toman en cuenta variaciones de tráfico en feriados como Memorial Day.

Sin embargo, pedir a los usuarios finales que escriban manualmente una cadena ISO como `2022-05-20T14:30:00` es propenso a errores. En interfaces web modernas (como Streamlit), el patrón UX estándar consiste en:
* Proveer selectores nativos de **Fecha (`st.date_input`)** y **Hora (`st.time_input`)**.
* Combinarlos en el lado del cliente en el formato ISO requerido (`datetime.combine(date, time).isoformat()`) antes de despachar la petición a `/predict`.

---

## 27. Acuerdos de Nivel de Servicio (SLA) y Benchmarking de Latencia en Proceso en APIs de Machine Learning

### Pregunta:
> ¿Qué es un Acuerdo de Nivel de Servicio (SLA) en ingeniería de machine learning y cómo se compara el benchmarking de latencia en proceso frente a la latencia de red de extremo a extremo?

### Respuesta:

#### 1. Definición de Acuerdo de Nivel de Servicio (SLA)
En ingeniería de software, computación en la nube y MLOps, un **Acuerdo de Nivel de Servicio (SLA - Service Level Agreement)** es el contrato operativo formal que define los estándares mínimos aceptables de calidad y rendimiento para un servicio en producción.

* **En Despliegue de Machine Learning**: Los SLAs de latencia dictan el tiempo máximo permitido (ej. $P_{99} < 100\text{ ms}$) dentro del cual un endpoint de inferencia debe devolver una cotización de predicción.
* **Objetivo de Experiencia de Usuario**: En aplicaciones de transporte y comercio electrónico, las cotizaciones deben retornar en $<100\text{ ms}$ para evitar retrasos perceptibles en la interfaz.

#### 2. Benchmarking en Proceso vs. Latencia de Red Extremo a Extremo
Al evaluar la velocidad de inferencia de Machine Learning, existen dos capas de medición distintas:

1. **Latencia del Modelo en Proceso (1.40 ms en nuestro proyecto)**:
   * Mide el tiempo puro de ejecución interna en Python / C++ (`model.predict_single(features)`).
   * Se evalúa cronometrando 1,000 ciclos de predicción en memoria en hardware local de desarrollo sin sobrecarga de red.
   * Demuestra que las rutas de decisión en LightGBM se ejecutan en tiempos sub-2ms prácticamente instantáneos.

2. **Latencia de Red Extremo a Extremo (3.0–5.0 ms en Docker)**:
   * Incluye el ciclo de vida HTTP completo:
     $$\text{Petición HTTP del Cliente} \longrightarrow \text{E/S de Red FastAPI} \longrightarrow \text{Validación Pydantic} \longrightarrow \text{Inferencia en Proceso (1.40ms)} \longrightarrow \text{Serialización JSON} \longrightarrow \text{Respuesta al Cliente}$$
   * Incluso con la serialización HTTP completa y la validación de esquemas con Pydantic, todo el viaje de ida y vuelta opera en ~3–5 ms, muy por debajo de cualquier presupuesto comercial de $100\text{ ms}$.
