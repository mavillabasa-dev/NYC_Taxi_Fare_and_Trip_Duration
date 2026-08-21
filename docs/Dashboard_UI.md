# Dashboard UI (T-111) — Documentación técnica

Este documento describe el dashboard de Streamlit implementado para T-111: predicción de
tarifa y duración de viajes de taxi en NYC, consumiendo la API de T-110.

> **Estado del proyecto en el momento de este documento:** T-105 a T-109 (feature
> engineering y modelling) todavía no están implementados. No existe un modelo entrenado
> real. Todo lo que se describe acá fue construido y probado contra un **modelo fixture**
> (ver sección "Qué falta integrar").

---

## 1. Arquitectura general

`ui/` es un árbol de Python **independiente** de `src/` y `api/` (mismo principio que ya
aplica el proyecto entre `src/` y `api/`: nunca se importan entre sí). Su única forma de
obtener predicciones es HTTP contra la API de T-110 — nunca carga un modelo directamente.

```
ui/
├── app.py                        # entrypoint de Streamlit
├── settings.py                   # configuración por variables de entorno
├── api_client.py                 # cliente HTTP hacia la API
├── zones.py                      # catálogo de zonas de NYC (nombres + coordenadas)
├── requirements.txt
├── Dockerfile
└── components/
    ├── prediction_form.py        # formulario + orquestación del resultado
    ├── trip_map.py                # mapa de puntos pickup/dropoff
    └── choropleth.py              # mapa choropleth de tarifas por zona
```

Los imports dentro de `ui/` son "planos" (`import api_client`, `from zones import ...`),
igual que hace `api/main.py` con `from settings import settings` — así el código funciona
sin importar si Streamlit corre con `ui/` como directorio actual o desde la raíz del repo.

---

## 2. Componentes, uno por uno

### `ui/settings.py`
Configuración centralizada, mismo patrón que `api/settings.py` (`dataclass` congelada +
`os.getenv`). Dos variables:

| Variable | Default | Para qué |
|---|---|---|
| `API_URL` | `http://localhost:8000` | URL base de la API de T-110. En Docker se pisa con `http://api:8000` (DNS interno de Compose). |
| `DATASET_DIR` | `../dataset` | Carpeta donde están `taxi_zone_centroids.csv` y `taxi_zones.geojson`. El default asume que corrés Streamlit con `ui/` como directorio actual. En Docker se pisa con `/app/dataset` (bind-mount). |

Se agregó `DATASET_DIR` porque `zones.py` y `choropleth.py` originalmente calculaban la
ruta a `dataset/` con `Path(__file__).resolve().parent.parent`, que se rompe apenas
Docker "aplana" la imagen (mismo problema que ya existía con `MODEL_PATH` en la API).

### `ui/api_client.py`
Único punto de contacto con la API. Dos funciones, **ninguna lanza excepción** — devuelven
un valor "de error" en vez de romper el script de Streamlit:

- `get_health() -> dict`: `GET /health`. Si no hay conexión, devuelve
  `{"status": "unreachable", "model_loaded": False, ...}`.
- `predict(payload: dict) -> tuple[int, dict]`: `POST /predict`. Devuelve
  `(status_code, body)`. `status_code == 0` significa "no se pudo conectar" (no es un
  código HTTP real, es un valor centinela nuestro).

### `ui/zones.py`
Carga `dataset/taxi_zone_centroids.csv` (generado por la ingesta de T-116,
`src/data_utils.py::derive_zone_centroids()`) y arma una columna `label` tipo
`"JFK Airport (Queens)"` para mostrar en los selectbox. Cacheado con `@st.cache_data` — se
lee el CSV una sola vez por sesión, no en cada rerun.

Dos zonas (`LocationID` 264 "Unknown" y 265 "Outside of NYC") no tienen coordenadas en el
shapefile original — el código las tolera con `fillna()` para el nombre, y los componentes
de mapa las descartan explícitamente (`dropna`) antes de graficar.

También vive acá `infer_ratecode(pu_location_id, do_location_id) -> int`: el `RatecodeID`
no es una preferencia del pasajero, es una consecuencia de qué zonas se eligen, así que se
infiere en vez de pedirse como input editable (ver detalle en la sección del formulario).
Regla: si el pickup o el dropoff es JFK (`LocationID 132`) → `2`; si es Newark
(`LocationID 1`) → `3`; en cualquier otro caso → `1` (estándar). LaGuardia (`138`) cae en
el caso general porque, a diferencia de JFK y Newark, no tiene tarifa especial en las
reglas reales de NYC TLC. Se pierde la posibilidad de setear tarifa negociada (`5`) o
grupal (`6`) — son casos raros que no se pueden inferir solo con la zona.

Por el mismo motivo (un PR review señaló que `trip_distance` tampoco debería ser un
número que el usuario tipea a mano) se agregó `estimate_trip_distance(pu_location_id,
do_location_id) -> float` + `haversine_miles(lat1, lon1, lat2, lon2) -> float`. Calcula la
distancia en **línea recta** entre los centroides de pickup y dropoff — es el único
cálculo de distancia que existe en el proyecto hasta ahora (antes vivía duplicado dentro
de `choropleth.py`, se centralizó acá). **Limitación real, no cosmética:** haversine
subestima sistemáticamente frente a una ruta real en auto (sobre todo en NYC, con ríos y
puentes de por medio) — es una mejora incremental sobre "el usuario adivina un número", no
un reemplazo de una API de ruteo real. Si alguna de las dos zonas no tiene coordenadas
conocidas (264/265), devuelve `DEFAULT_TRIP_DISTANCE_MILES` (`3.0`) como respaldo, porque
la API igual necesita un `trip_distance` numérico en el payload.

### `ui/components/prediction_form.py`
El corazón funcional del dashboard:

1. Carga las zonas (`zones.load_zones()`) y arma dos `st.selectbox` con nombres reales
   (Pickup/Dropoff), con defaults JFK Airport → Upper East Side North.
2. **Pickup zone y Dropoff zone están AFUERA del `st.form`**, a propósito: con ellas se
   calculan `infer_ratecode(pu_id, do_id)` y `estimate_trip_distance(pu_id, do_id)` (ambas
   de `zones.py`), mostrados en dos campos **deshabilitados** ("Rate code (auto)", "Trip
   distance (auto, miles)") con el mismo look que un input normal pero no editable. Los
   widgets dentro de un `st.form` no disparan rerun hasta el submit — sacar las zonas del
   form es lo que permite que estos campos se actualicen al instante apenas se cambia de
   zona, en vez de recién después de predecir. Si alguna zona no tiene coordenadas
   conocidas, aparece un `st.caption` aclarando que la distancia mostrada es un valor de
   respaldo, no una estimación real.
3. El resto de los inputs (fecha, pasajeros) sí están adentro de un `st.form` — evita que
   cada tecla dispare un rerun, solo el botón "Predict" lo hace.
4. Al submit, arma el payload exacto que espera `PredictionRequest`
   (`api/app/model/schema.py`) — usando `RatecodeID` y `trip_distance` ya calculados, no
   valores elegidos por el usuario — y llama `api_client.predict()`.
5. El resultado se guarda en `st.session_state["last_prediction"]` (ver sección Streamlit
   más abajo — es la parte más importante para entender el archivo).
6. `_render_result()` interpreta el `status_code` y muestra:
   - `200` → métricas + mapa de puntos + checkbox opcional para el choropleth.
   - `0` → error "no se pudo contactar la API".
   - `503` → error "modelo no cargado".
   - `422` → errores de validación formateados campo por campo (no el JSON crudo de
     Pydantic).
   - cualquier otro código → error genérico.

### `ui/components/trip_map.py`
Recibe dos `LocationID` y dibuja dos marcadores (verde=pickup, rojo=dropoff) con
`plotly.graph_objects.Scattermapbox`, usando las coordenadas de `zones.load_zones()`. Si
alguna de las dos zonas no tiene coordenadas conocidas, muestra un `st.info` en vez de
romper.

### `ui/components/choropleth.py`
Mapa de NYC coloreado por tarifa predicha. Para la zona de pickup elegida, llama
`/predict` contra las 263 zonas con geometría conocida (dropoff variable), usando
`zones.haversine_miles()` como proxy de `trip_distance` en cada fila — la misma función
que ahora usa también `prediction_form.py` para el campo "Trip distance (auto)" (antes
estaba duplicada acá). El `RatecodeID` de cada una de las 263 llamadas también se infiere
por fila con `zones.infer_ratecode(pu_location_id, do_row.LocationID)` — ninguno de los
dos es un valor fijo para todo el grid, porque el pickup puede combinarse con un dropoff
que sí es JFK o Newark en alguna fila puntual. Cacheado con `st.cache_data` por
`(pickup_zone, hora, pasajeros)`, y detrás de un checkbox para no disparar ~263 llamadas
HTTP en cada predicción simple.

### `ui/requirements.txt`
```
streamlit
requests
plotly
pandas
```
Deliberadamente mínimo — **no incluye `geopandas`** (necesaria para leer shapefiles), esa
dependencia vive solo en el `requirements.txt` de la raíz (offline, host) porque el
dashboard nunca lee el shapefile directamente, solo el `.geojson` ya convertido.

### `ui/Dockerfile`
Mismo patrón que `api/Dockerfile`: `python:3.11-slim`, instala requirements, `COPY . .`,
corre `streamlit run app.py --server.address=0.0.0.0 --server.port=8501`.

### `scripts/dev_fixture_model.py` y `scripts/shapefile_to_geojson.py`
No son parte de ningún ticket — son herramientas de desarrollo temporales, fuera de
`src/` y `api/`:

- **`dev_fixture_model.py`**: genera un `models/model.pkl` de mentira (siempre predice
  `fare=18.75`, `duration=27.5`) para poder desarrollar y demostrar el dashboard sin
  esperar a T-105-T-109. Usa `cloudpickle` (no `pickle`) porque así el archivo se puede
  cargar desde cualquier proceso sin que ese proceso necesite importar este script — el
  detalle está comentado en el propio archivo.
- **`shapefile_to_geojson.py`**: convierte el shapefile de zonas (ya extraído por la
  ingesta de T-116) a `dataset/taxi_zones.geojson`, que es lo único que `choropleth.py`
  necesita leer. Se corre una sola vez, en el host, con `geopandas`.

---

## 3. Cómo levantar el proyecto

### Local (dos terminales)

**Terminal 1 — API:**
```bash
source .venv/bin/activate
cd api
MODEL_PATH=../models/model.pkl uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
> El override de `MODEL_PATH` es necesario porque `.env` trae una ruta pensada para
> Docker (`models/model.pkl`, relativa a `/app`). Corriendo local con `api/` como
> directorio actual, esa misma ruta relativa apuntaría a `api/models/model.pkl`, que no
> existe.

**Terminal 2 — Dashboard:**
```bash
source .venv/bin/activate
cd ui
streamlit run app.py
```
Streamlit imprime una URL (`http://localhost:8501`) y abre el browser solo.

**Antes de la primera vez**, hace falta:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r api/requirements.txt -r ui/requirements.txt cloudpickle
cp .env.original .env
python scripts/dev_fixture_model.py          # genera el modelo fixture
python -m src.data_utils                     # descarga datos + genera taxi_zone_centroids.csv
python scripts/shapefile_to_geojson.py       # genera taxi_zones.geojson
```

### Docker

```bash
cp .env.original .env    # si no existe todavía
docker compose build
docker compose up
```

Esto levanta dos servicios: `api` (puerto 8000) y `dashboard` (puerto 8501), en la misma
red interna de Compose — el dashboard resuelve la API como `http://api:8000` (nombre de
servicio, no `localhost`).

⚠️ **Limitación conocida**: el modelo fixture está serializado con `cloudpickle`, que no
está en `api/requirements.txt` (a propósito — ese archivo debe quedar mínimo para el
modelo real de T-109, no para nuestra herramienta de desarrollo). Esto significa que hoy,
`docker compose up` levanta la API en estado `degraded` (modelo no cargado) salvo que se
instale `cloudpickle` manualmente dentro del contenedor (`docker compose exec api pip
install cloudpickle && docker compose restart api`). Es un problema **solo del modelo
fixture** — el modelo real de T-109 no debería tener este inconveniente si se serializa
con `pickle` estándar (lo normal para modelos de `scikit-learn`/`LightGBM`/`XGBoost`).

---

## 4. Streamlit para principiantes (lo que usamos de este framework)

Streamlit no se parece a React/Vue. No hay componentes con estado propio ni virtual DOM —
es un **script de Python que se vuelve a ejecutar completo, de arriba a abajo, cada vez
que el usuario interactúa con un widget** (un click, un selectbox, un checkbox). Esto
explica varias decisiones de diseño del código:

### `st.session_state` — el motivo por el que existe
Las variables normales de Python (`payload`, `status_code`, etc.) se **destruyen** en cada
rerun. Si guardáramos el resultado de la predicción en una variable común, desaparecería
apenas el usuario tocara *cualquier otro* widget (por ejemplo, el checkbox del
choropleth) — de hecho, esto pasó literalmente durante el desarrollo (ver
`prediction_form.py`, líneas 55-62): al tildar el checkbox del choropleth, todo el
resultado (métricas + mapa) desaparecía, porque el rerun disparado por el checkbox volvía
a poner `submitted = False`.

**Solución:** `st.session_state` es un diccionario que **persiste entre reruns**, para la
sesión del browser del usuario. Guardamos el resultado ahí una sola vez (cuando se
predice) y lo leemos en cada rerun subsiguiente, sin importar qué widget lo disparó.

### `st.cache_data` — evitar trabajo repetido
Como el script entero se re-ejecuta en cada interacción, leer un CSV de 265 filas o un
GeoJSON de 4.5 MB en cada rerun sería un desperdicio. `@st.cache_data` memoriza el
resultado de una función según sus argumentos — se usa en `zones.load_zones()`,
`choropleth._load_geojson()` y `choropleth._predict_grid()` (esta última, clave: evita
repetir ~263 llamadas HTTP si el usuario no cambió ni la zona de pickup ni la hora).

### `st.form` — agrupar inputs sin re-ejecutar en cada tecla
Sin `st.form`, cada widget (cada `number_input`, cada `selectbox`) dispararía su propio
rerun apenas cambia. `st.form` agrupa varios inputs y solo dispara un rerun cuando se
aprieta el botón de submit (`st.form_submit_button`) — necesario para no llamar a la API
en cada tecla que el usuario tipea.

**La otra cara de la moneda**, y por qué importa entender esto: si algo necesita
actualizarse *antes* del submit — como los campos "Rate code (auto)" y "Trip distance
(auto, miles)" de `prediction_form.py`, que tienen que reflejar la zona elegida al
instante, no recién después de predecir — esos widgets (y todo lo que dependa de ellos)
tienen que quedar **afuera** del `st.form`. Es el trade-off central de Streamlit: agrupar
en un form gana eficiencia
(menos reruns) pero pierde reactividad en vivo; sacar algo del form gana reactividad pero
dispara un rerun por cada cambio. Se eligió según qué necesitaba cada campo.

### Otros elementos usados
- `st.columns()` — layout en columnas (usado para las métricas y el formulario).
- `st.metric()` — número grande con label, para mostrar tarifa/duración predichas.
- `st.selectbox` / `st.number_input` / `st.text_input` / `st.checkbox` — widgets de input.
- `st.success` / `st.warning` / `st.error` / `st.info` — cajas de mensaje con color
  semántico.
- `st.progress()` — barra de progreso (usada durante las ~263 llamadas del choropleth).
- `st.plotly_chart()` — embebe una figura de Plotly (usado para ambos mapas).

---

## 5. Qué falta integrar

El dashboard está completo y probado en su plumbing (formulario, mapas, manejo de
errores, Docker), pero depende de piezas que **todavía no existen** en otras partes del
proyecto:

1. **Modelo real (T-105 a T-109)**: hoy todo corre contra un modelo fixture que siempre
   devuelve `fare=18.75`, `duration=27.5` sin importar el input. El choropleth se ve
   "plano" (un solo color) porque no hay variación real que mostrar todavía. Cuando
   `models/model.pkl` sea reemplazado por el artefacto real de T-109, **no hace falta
   tocar ningún código de `ui/`** — el contrato HTTP con la API no cambia.
2. **`cloudpickle` en el contenedor de la API**: el modelo fixture necesita esa librería
   para cargar dentro de Docker; el modelo real probablemente no la va a necesitar (ver
   sección 3). Si el modelo real también tuviera una dependencia de serialización
   especial, hay que agregarla a `api/requirements.txt` en T-109/T-114.
3. **Tabla de comparación de modelos**: la idea original incluía una sección de métricas
   (MAE/RMSE) comparando los modelos de T-106/107/108 — quedó fuera de esta
   implementación porque no es un criterio de aceptación formal de T-111 y esos números
   todavía no existen (dependen de T-109). Se puede agregar leyendo un artefacto estático
   (ej. `models/model_comparison.json`) el día que exista.
4. **`docker-compose.yml` tiene la key `version: "3.9"` obsoleta** — Compose la ignora con
   un warning; sacarla es tarea explícita de T-114, no se tocó acá.
5. **`Trip distance (auto)` usa haversine, no una API de ruteo real**: es una mejora
   incremental (antes era 100% manual) pero sigue siendo una subestimación sistemática de
   la distancia real en auto — sobre todo en NYC, con ríos y puentes de por medio. La
   solución de fondo sería integrar una API de ruteo (Google Maps, OSRM, etc.), fuera de
   alcance de este PR.
