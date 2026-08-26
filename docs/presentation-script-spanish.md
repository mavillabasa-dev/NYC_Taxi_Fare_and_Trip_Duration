# Predicción de Tarifa y Duración en Taxis de NYC — Guión de Presentación (Español)

**Presentación Final Demo Day · 18 Diapositivas · 5 Presentadores · Tiempo Total: ~21:15 (incluyendo 2:30 de Demo en Vivo)**

---

## 1. Resumen de Presentadores y Asignación de Tiempos

| Presentador | Diapositivas Asignadas | Responsabilidades de Sección | Tiempo Total |
|---|---|---|---:|
| **Keyneth Lara** | Diapositivas 1–4 | Visión General del Proyecto y Requisitos | **3:55** |
| **William Vera** | Diapositivas 5–8 | Arquitectura, Limpieza de Datos e Ingeniería de Features | **5:00** |
| **Marcos Villabasa** | Diapositivas 9–13 | Exploración de Modelos, Leaderboard e Interpretabilidad | **6:30** |
| **Mauricio Mora** | Diapositiva 14 | Demostración en Vivo en Docker (FastAPI y Streamlit UI) | **2:30** |
| **Néstor Mamani** | Diapositivas 15–18 | Supuestos, Hoja de Ruta, Conclusiones y Preguntas (Q&A) | **3:20** |
| **Equipo Completo** | **Diapositivas 1–18** | **Historia Completa del Proyecto y Demostración en Vivo** | **~21:15** |

---

## 2. Orden de Ejecución de la Presentación

| # | Título de la Diapositiva | Sección | Presentador | Tiempo |
|---|---|---|---|---:|
| **01** | ¿Cuánto va a costar? ¿Cuánto va a tardar? | 1. Visión General | Keyneth Lara | 0:40 |
| **02** | El taxímetro responde demasiado tarde | 1. Visión General | Keyneth Lara | 0:50 |
| **03** | Escala del Dataset y Análisis Exploratorio (EDA) | 1. Visión General | Keyneth Lara | 1:15 |
| **04** | Requisitos de Ingeniería | 2. Requisitos | Keyneth Lara | 1:10 |
| **05** | Arquitectura del Sistema | 3. Alcance y Soluciones | William Vera | 1:20 |
| **06** | Limpieza de Datos y División Temporal | 3. Alcance y Soluciones | William Vera | 1:20 |
| **07** | Pipeline de Ingeniería de Características | 3. Alcance y Soluciones | William Vera | 1:20 |
| **08** | Contrato de Variables y Prevención de Fuga | 3. Alcance y Soluciones | William Vera | 1:00 |
| **09** | Estrategia de Exploración de Modelos | 4. Métricas y Resultados | Marcos Villabasa | 1:15 |
| **10** | Comparación Exhaustiva de Modelos (Leaderboard) | 4. Métricas y Resultados | Marcos Villabasa | 1:30 |
| **11** | Comparación Visual de Métricas (MAE y R²) | 4. Métricas y Resultados | Marcos Villabasa | 1:20 |
| **12** | Por qué Ganó LightGBM | 4. Métricas y Resultados | Marcos Villabasa | 1:10 |
| **13** | Importancia de Variables y Ganancia por Split | 4. Métricas y Resultados | Marcos Villabasa | 1:15 |
| **14** | **Demostración en Vivo del Prototipo** *(Demo)* | 5. Demostración | Mauricio Mora | **2:30** |
| **15** | Supuestos y Compensaciones de Ingeniería | 6. Conclusiones y Próximos Pasos | Néstor Mamani | 1:00 |
| **16** | Próximos Pasos y Hoja de Ruta | 6. Conclusiones y Próximos Pasos | Néstor Mamani | 1:00 |
| **17** | Puntos Clave del Proyecto | 6. Conclusiones y Próximos Pasos | Néstor Mamani | 1:00 |
| **18** | Agradecimiento y Preguntas (Q&A) | 6. Conclusiones y Próximos Pasos | Néstor Mamani | 0:20 |

---

## 3. Guión Detallado Diapositiva por Diapositiva

### Diapositiva 01: ¿Cuánto va a costar? ¿Cuánto va a tardar?
* **Presentador**: Keyneth Lara
* **Sección**: 1. Visión General
* **Tiempo**: 0:40

> "Buenas tardes a todos. Somos el equipo de ingeniería de NYC Taxi. Dos preguntas fundamentales definen la movilidad urbana: *¿Cuánto va a costar el viaje?* y *¿Cuánto va a tardar?* Durante décadas, los pasajeros de taxis amarillos tenían que esperar al final del viaje para saberlo. Hoy les presentamos un sistema de machine learning de punta a punta que responde ambas preguntas simultáneamente, con una latencia menor a 2 milisegundos, antes de que el pasajero suba al vehículo."

---

### Diapositiva 02: El taxímetro responde demasiado tarde
* **Presentador**: Keyneth Lara
* **Sección**: 1. Visión General
* **Tiempo**: 0:50

> "El taxímetro tradicional opera al momento de la bajada ($t_1$). Uno solo se entera de la tarifa final cuando el viaje termina. Sin embargo, las aplicaciones cambiaron las expectativas: los pasajeros exigen precios claros y por adelantado al momento de la subida ($t_0$). La comisión de taxis de Nueva York ahora permite tarifas por adelantado, pero implementarlo requiere un modelo predictivo exacto y en tiempo real. Toda nuestra arquitectura se enfoca en este segundo exacto: predecir usando únicamente lo que se sabe al subir."

---

### Diapositiva 03: Escala del Dataset y Análisis Exploratorio (EDA)
* **Presentador**: Keyneth Lara
* **Sección**: 1. Visión General
* **Tiempo**: 1:15

> "Para entrenar los modelos, procesamos el conjunto de datos oficial de NYC TLC de mayo de 2022: 3.59 millones de viajes en 265 zonas. Nuestro análisis exploratorio reveló tres características clave: primero, distribuciones asimétricas a la derecha con tarifas medias de $15.15 y duraciones de 16.5 minutos; segundo, patrones temporales claros en horas pico; y tercero, una gran concentración espacial en aeropuertos y Midtown. El EDA también identificó valores negativos, viajes sin pasajeros y columnas posteriores al viaje que requerían un filtrado riguroso."

---

### Diapositiva 04: Requisitos de Ingeniería
* **Presentador**: Keyneth Lara
* **Sección**: 2. Requisitos
* **Tiempo**: 1:10

> "Antes de escribir código, definimos requisitos estrictos. Funcionalmente, el prototipo debe entregar predicciones simultáneas para tarifa y duración, superando con holgura los baselines ingenuos de media y estableciendo un margen de error acotado para cotizaciones en tiempo real, garantizando cero fuga de datos. No funcionalmente, el prototipo de API debe cumplir un objetivo de latencia menor a 100 milisegundos, estar totalmente contenerizado con Docker Compose y ofrecer una interfaz interactiva junto a un endpoint REST. William les presentará nuestra arquitectura de sistema."

---

### Diapositiva 05: Arquitectura del Sistema
* **Presentador**: William Vera
* **Sección**: 3. Alcance y Soluciones
* **Tiempo**: 1:20

> "Gracias, Keyneth. Esta es nuestra arquitectura de sistema de punta a punta. A la izquierda está el pipeline offline dividido en tres etapas claras: Etapa 1 de Ingesta y Limpieza de Datos, Etapa 2 de Ingeniería de Características y Etapa 3 de Entrenamiento y Evaluación. La decisión de diseño clave es la frontera punteada: un solo artefacto serializado—`model.pkl` (4.2 MB)—cruza hacia el entorno Docker a la derecha. Allí, FastAPI sirve las predicciones vía REST y Streamlit actúa como interfaz interactiva comunicándose por HTTP, garantizando un despliegue ligero y desacoplado."

---

### Diapositiva 06: Limpieza de Datos y División Temporal
* **Presentador**: William Vera
* **Sección**: 3. Alcance y Soluciones
* **Tiempo**: 1:20

> "Nuestro pipeline de limpieza filtra 3.59 millones de filas a 3.30 millones de alta calidad, con un 92.04% de retención. Aplicamos límites físicos: pasajeros entre 1 y 9, duraciones de 1 a 300 minutos, distancias de 0.1 a 150 millas y tarifas de $2.50 a $500. Fundamentalmente, implementamos una división temporal cronológica al 23 de mayo de 2022. La división aleatoria tradicional filtra información del futuro al pasado; nuestra división de 3 semanas de entrenamiento y 1 semana de prueba simula un entorno real."

---

### Diapositiva 07: Pipeline de Ingeniería de Características
* **Presentador**: William Vera
* **Sección**: 3. Alcance y Soluciones
* **Tiempo**: 1:20

> "Pasando a las transformaciones de variables: a partir de 6 entradas crudas, nuestro pipeline genera 29 variables estructuradas en cuatro etapas. Primero, temporal: codificaciones cíclicas de seno y coseno para hora y día de la semana, más banderas de hora pico y feriados. Segundo, espacial: extraemos centroides del shapefile de TLC para calcular distancias Haversine, distancias Manhattan e indicadores de aeropuertos. Tercero, categórico: codificación por target con suavizado Bayesiano para zonas y tarifas, ajustada estrictamente sobre entrenamiento. Y cuarto, metadatos numéricos."

---

### Diapositiva 08: Contrato de Variables y Prevención de Fuga
* **Presentador**: William Vera
* **Sección**: 3. Alcance y Soluciones
* **Tiempo**: 1:00

> "Para eliminar cualquier fuga de datos, aplicamos un contrato de variables estricto. Solo se permiten seis variables de entrada al momento de inferencia: IDs de zona de origen y destino, fecha y hora de subida, cantidad de pasajeros, código de tarifa y distancia estimada. Ocho columnas posteriores al viaje—como hora de bajada, monto total, propina y peajes—están estrictamente prohibidas para prevenir fuga de target y sesgo temporal. Documentamos el supuesto clave de que la distancia representa una estimación inicial de GPS o motor de ruteo. Marcos les presentará nuestra estrategia de exploración de modelos."

---

### Diapositiva 09: Estrategia de Exploración de Modelos
* **Presentador**: Marcos Villabasa
* **Sección**: 4. Métricas y Resultados
* **Tiempo**: 1:15

> "Gracias, William. Evaluamos tres familias distintas de modelos. La Familia 1 estableció las líneas base: un regresor de Media Trivial y un Árbol de Decisión. La Familia 2 exploró árboles con gradient boosting: LightGBM y XGBoost con binning por histogramas. La Familia 3 implementó redes neuronales: un perceptrón multicapa con escalado estándar y parada temprana. Cada modelo se entrenó exactamente sobre los 2.40 millones de registros de entrenamiento y se evaluó en el conjunto de prueba temporal de 900k."

---

### Diapositiva 10: Comparación Exhaustiva de Modelos (Leaderboard)
* **Presentador**: Marcos Villabasa
* **Sección**: 4. Métricas y Resultados
* **Tiempo**: 1:30

> "Esta es nuestra tabla maestra evaluada sobre los 900,000 viajes del conjunto de prueba. La Media Trivial tuvo un MAE de $8.85 en tarifa y 9.27 minutos en duración, sin capacidad predictiva. Los árboles de decisión mejoraron a $1.46 y 3.94 minutos. LightGBM fue el claro ganador, logrando un MAE de $1.42 en tarifa y 3.76 minutos en duración, con un R² de 0.952 y 0.797 respectivamente. XGBoost quedó muy cerca con $1.45, mientras que la red MLP quedó atrás con $2.21. LightGBM superó a todos en cada métrica."

---

### Diapositiva 11: Comparación Visual de Métricas (MAE y R²)
* **Presentador**: Marcos Villabasa
* **Sección**: 4. Métricas y Resultados
* **Tiempo**: 1:20

> "Esta diapositiva presenta una comparación gráfica directa de nuestras dos métricas principales en todas las familias de modelos. A la izquierda, el gráfico de MAE compara el error en tarifa y duración lado a lado, mostrando cómo LightGBM logra el menor error en ambos objetivos: solo $1.42 en tarifa y 3.76 minutos en duración. A la derecha, el gráfico de R² ilustra la varianza explicada: los árboles de boosting capturan casi el 80% de la varianza en duración, superando holgadamente el 56.6% de la red neuronal, mientras explican más del 95% de la varianza en tarifas."

---

### Diapositiva 12: Por qué Ganó LightGBM
* **Presentador**: Marcos Villabasa
* **Sección**: 4. Métricas y Resultados
* **Tiempo**: 1:10

> "Tres factores concretos hicieron de LightGBM nuestra elección para producción. Primero, exactitud: su crecimiento por hojas y binning por histogramas capturan interacciones no lineales entre distritos mejor que otros modelos. Segundo, velocidad: 1.40 ms de ejecución en proceso en una laptop estándar permite calcular predicciones de forma instantánea con muy bajo uso de CPU. Y tercero, tamaño: el artefacto serializado pesa solo 4.2 MB, carga en 120 ms al iniciar el contenedor y no requiere GPUs."

---

### Diapositiva 13: Importancia de Variables y Ganancia por Split
* **Presentador**: Marcos Villabasa
* **Sección**: 4. Métricas y Resultados
* **Tiempo**: 1:15

> "Analizando la importancia de variables: las variables de distancia dominan el modelo, representando aproximadamente el 85% de la ganancia para tarifas y el 67% para duración. Específicamente, la distancia Manhattan es el principal predictor de tarifa con 32.1%, mientras que la distancia de viaje aporta el 43% en duración. Las variables horarias aportan un 8% adicional a la duración para capturar la congestión, pero menos del 2% a la tarifa. Fundamentalmente, pasamos la auditoría de fuga: ninguna variable supera el 65% de dominancia. Mauricio pasará ahora a realizar la demostración en vivo."

---

### Diapositiva 14: Demostración en Vivo del Prototipo (Demo)
* **Presentador**: Mauricio Mora
* **Sección**: 5. Demostración
* **Tiempo**: 2:30

> **[Transición a la Demo en Vivo]**: "Gracias, Marcos. Ahora comparto pantalla con nuestro despliegue en Docker para mostrar la experiencia completa tanto en el backend de FastAPI como en el dashboard interactivo de Streamlit."
> 
> **[1. Documentación FastAPI — Swagger UI]**: "Primero, vemos el servicio FastAPI en `http://localhost:8000/docs`. Ejecutamos el endpoint `/health` para confirmar que el contenedor está activo y que el artefacto de 4.2 MB está cargado en memoria. Luego, probamos el endpoint `/predict` con un viaje real: origen en Times Square (Zona 230), destino en Aeropuerto JFK (Zona 132), miércoles a las 6:30 PM, 1 pasajero y Tarifa 2 (JFK). Al hacer clic en 'Execute', la respuesta llega en solo 3 milisegundos: predice con exactitud la tarifa fija de $52.00 y estima 48 minutos de duración en hora pico. Si ingresamos datos inválidos, como una distancia negativa, la validación estricta con Pydantic rechaza la petición de inmediato con un error 422 estructurado."
> 
> **[2. Dashboard Geoespacial en Streamlit]**: "Ahora pasamos al dashboard interactivo en `http://localhost:8501`. En la pestaña de Estimador Individual, los usuarios pueden seleccionar cualquiera de las 265 zonas de taxi de NYC. Al seleccionar origen y destino, la interfaz traza la ruta geodésica en el mapa, calcula distancias Haversine y Manhattan, y actualiza los indicadores de tarifa y duración en tiempo real. En la pestaña de Mapa de Calor, al elegir un punto de origen, el sistema ejecuta inferencia batch sobre las 265 zonas simultáneamente, generando un mapa coroplético que visualiza gradientes de costo y tiempo en toda la ciudad."
> 
> **[Conclusión de la Demo]**: "Como han visto, los modelos LightGBM ofrecen predicciones en milisegundos, validación robusta y visualización geoespacial intuitiva en un contenedor totalmente autónomo. Néstor les presentará ahora los supuestos y la hoja de ruta."

---

### Diapositiva 15: Supuestos y Compensaciones de Ingeniería
* **Presentador**: Néstor Mamani
* **Sección**: 6. Conclusiones y Próximos Pasos
* **Tiempo**: 1:00

> "Gracias, Mauricio. Documentamos tres supuestos operativos y compensaciones de ingeniería. Primero, alcance temporal: los modelos se entrenaron exclusivamente con 3.5 millones de registros de mayo de 2022, capturando la primavera tardía pero sin observar estacionalidad anual como vacaciones de verano o tormentas de nieve en invierno. Segundo, clima dinámico: tormentas repentinas reducen la velocidad y no son observadas por variables de calendario sin telemetría meteorológica en vivo. Tercero, tráfico en tiempo real: aunque las tarifas de taxi son reguladas, incidentes viales imprevistos introducen varianza en la duración."

---

### Diapositiva 16: Próximos Pasos y Hoja de Ruta
* **Presentador**: Néstor Mamani
* **Sección**: 6. Conclusiones y Próximos Pasos
* **Tiempo**: 1:00

> "Nuestra hoja de ruta describe tres potenciales mejoras para producción. Primero, integrar un contenedor OSRM de ruteo en Docker Compose para calcular geometría de ruta y distancias reales antes de la subida. Segundo, implementar telemetría MLOps con Prometheus y Evidently AI para detectar proactivamente drift en datos, cambios estacionales y percentiles de latencia. Y tercero, establecer pipelines automatizados con Airflow para ingesta mensual de TLC y reentrenamiento champion-challenger."

---

### Diapositiva 17: Puntos Clave del Proyecto
* **Presentador**: Néstor Mamani
* **Sección**: 6. Conclusiones y Próximos Pasos
* **Tiempo**: 1:00

> "Para concluir, nuestro proyecto logró tres hitos fundamentales. Primero, ingeniería rigurosa: aplicando divisiones temporales estrictas y contratos de variables automatizados, garantizamos cero fuga de datos y métricas auténticas. Segundo, modelado de alto rendimiento: LightGBM logró un MAE de $1.42 en tarifa y 3.76 min en duración con latencia menor a 2 ms. Y tercero, arquitectura lista para producción: microservicios desacoplados y un artefacto autónomo listo para desplegar."

---

### Diapositiva 18: Agradecimiento y Preguntas (Q&A)
* **Presentador**: Néstor Mamani
* **Sección**: 6. Conclusiones y Próximos Pasos
* **Tiempo**: 0:20

> "El taxímetro tradicional responde demasiado tarde; nuestro sistema responde antes de subir al vehículo. De parte de Keyneth, William, Marcos, Mauricio y de mi parte: muchas gracias por su tiempo. ¡Quedamos abiertos a sus preguntas y comentarios!"
