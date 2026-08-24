# Dashboard UI (T-111) — Documentación técnica

Este documento describe el dashboard de Streamlit implementado para T-111: predicción de
tarifa y duración de viajes de taxi en NYC, consumiendo la API de T-110.

> **Estado:** revisado tras el cierre de T-105 a T-114. Cuando se escribió la primera
> versión de este documento no existía el pipeline de modelling; hoy sí, y
> `src/model_selection.py` produce un artefacto LightGBM real. **Nada de `ui/` cambió por
> eso**: su único contrato es HTTP contra la API de T-110. Para desarrollar sigue siendo
> válido levantar todo contra el modelo fixture de `scripts/dev_fixture_model.py`
> (ver sección 3).

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
| `DATASET_DIR` | autodetectado | Carpeta donde están `taxi_zone_centroids.csv` y `taxi_zones.geojson`. En Docker se pisa con `/app/dataset` (bind-mount). |

Se agregó `DATASET_DIR` porque `zones.py` y `choropleth.py` originalmente calculaban la
ruta a `dataset/` con `Path(__file__).resolve().parent.parent`, que se rompe apenas
Docker "aplana" la imagen (mismo problema que ya existía con `MODEL_PATH` en la API).

Cuando la variable no está seteada, `_get_default_dataset_dir()` prueba en orden:
`./dataset` (raíz del repo como directorio actual), luego `<repo>/dataset` calculado desde
`__file__`, y por último `../dataset` (Streamlit corriendo desde `ui/`). Así el mismo
código funciona local y en el contenedor sin configurar nada.

> ⚠️ El default se evalúa **al importar el módulo**, porque es el valor por defecto de un
> campo de `dataclass`. Depende del directorio de trabajo en ese instante y no se puede
> cambiar después con `monkeypatch`. Mismo patrón —y misma limitación— que
> `api/settings.py`.

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

El Taxi Zone Shapefile solo trae geometría para los `LocationID` **1 a 263**; las dos
zonas restantes (264 y 265) llegan sin coordenadas. El código las tolera con `fillna()`
para el nombre, y los componentes de mapa las descartan explícitamente (`dropna`) antes de
graficar.

> Esas dos zonas no son un detalle cosmético: el schema de la API acepta `1..265`, y
> durante un tiempo `predict_fast` las resolvía a `(0.0, 0.0)` y calculaba un viaje de
> ~5.400 millas desde el golfo de Guinea. Está corregido y cubierto por
> `tests/test_predictor_parity.py`, que compara las dos rutas de inferencia para cada uno
> de los 265 `LocationID`.

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
streamlit==1.41.1
requests==2.32.3
plotly==5.24.1
pandas==2.2.3
```
Deliberadamente mínimo — **no incluye `geopandas`** (necesaria para leer shapefiles), esa
dependencia vive solo en el `requirements.txt` de la raíz (offline, host) porque el
dashboard nunca lee el shapefile directamente, solo el `.geojson` ya convertido.

Las versiones están fijadas con `==`, igual que `api/requirements.txt`. A diferencia de
esos dos archivos, aquí no hace falta sincronizar nada con la raíz: el dashboard no
deserializa el modelo, así que su `pandas` no tiene por qué coincidir con el que produjo
`models/model.pkl`.

### `ui/Dockerfile`
Mismo patrón que `api/Dockerfile`: `python:3.11-slim`, instala requirements, `COPY . .`,
corre `streamlit run app.py --server.address=0.0.0.0 --server.port=8501`.

### `scripts/dev_fixture_model.py` y `scripts/shapefile_to_geojson.py`
No son parte de ningún ticket — son herramientas de desarrollo temporales, fuera de
`src/` y `api/`:

- **`dev_fixture_model.py`**: genera un `models/model.pkl` con estimadores
  `LinearRegression` baratos y centroides sintéticos, para poder desarrollar y demostrar
  el dashboard sin esperar al artefacto real. Importa la clase como
  `app.model.predictor` (no `api.app.model.predictor`): pickle graba la ruta del módulo,
  y dentro de la imagen el `COPY . .` desde `./api` aplana el árbol, así que solo existe
  `app.model.predictor`. Se niega a sobrescribir un artefacto existente salvo `--force`,
  porque escribe en la misma ruta que el modelo real.
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
pip install -r requirements-dev.txt -r ui/requirements.txt
cp .env.original .env
python scripts/dev_fixture_model.py          # genera el modelo fixture
python -m src.data_utils                     # descarga datos + genera taxi_zone_centroids.csv
python scripts/shapefile_to_geojson.py       # genera taxi_zones.geojson
```

En Windows / PowerShell:
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt -r ui\requirements.txt
Copy-Item .env.original .env
python scripts\dev_fixture_model.py
python -m src.data_utils
python scripts\shapefile_to_geojson.py
```

> `make install` y `make fixture` hacen los dos primeros pasos, pero **`make` no viene con
> Windows** — ahí hay que usar los comandos de arriba tal cual.

`dev_fixture_model.py` se niega a pisar un `models/model.pkl` existente: si ya entrenaste
el modelo real, o pasás `--force` a propósito, o escribís a otro lado con `--output`.

El dashboard necesita `taxi_zone_centroids.csv` para arrancar y ese archivo está
gitignoreado, así que **una checkout limpia no puede levantar el dashboard hasta correr la
ingesta**. `taxi_zones.geojson` sí está commiteado, pero no alcanza por sí solo.

### Docker

```bash
cp .env.original .env    # si no existe todavía
docker compose build
docker compose up
```

Esto levanta dos servicios: `api` (puerto 8000) y `dashboard` (puerto 8501), en la misma
red interna de Compose — el dashboard resuelve la API como `http://api:8000` (nombre de
servicio, no `localhost`). El `dashboard` espera con `depends_on: condition:
service_healthy`, así que no arranca hasta que el healthcheck de `GET /health` pase.

T-114 agregó un tercer servicio, `trainer`, detrás del perfil `train`: no se levanta con
`docker compose up` normal, solo con `docker compose --profile train up trainer`. Corre el
pipeline de entrenamiento completo en un contenedor efímero y deja el artefacto en
`./models` por bind-mount.

> **Nota histórica**: este documento describía antes una limitación por la que
> `docker compose up` dejaba la API en estado `degraded`, y sugería instalar
> `cloudpickle` dentro del contenedor. El diagnóstico era incorrecto: la causa real no
> era la librería de serialización sino la **ruta de módulo** grabada en el pickle. El
> fixture importaba la clase como `api.app.model.predictor`, y ese paquete no existe
> dentro de la imagen, así que la carga fallaba con `ModuleNotFoundError: No module
> named 'api'` — instalar `cloudpickle` no lo habría arreglado.
>
> Ya está corregido: el fixture importa `app.model.predictor` y serializa con `pickle`
> estándar, igual que `src/model_selection.export_production_model`. El artefacto pasa
> `tests/verify_isolation.py` y carga en el contenedor sin dependencias extra.

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
choropleth) — de hecho, esto pasó literalmente durante el desarrollo (el comentario que lo
explica sigue en `prediction_form.py`, junto a la escritura en `session_state`): al tildar
el checkbox del choropleth, todo el resultado (métricas + mapa) desaparecía, porque el
rerun disparado por el checkbox volvía a poner `submitted = False`.

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

El plumbing del dashboard (formulario, mapas, manejo de errores, Docker) está completo.
Cuando se escribió la primera versión de este documento, casi todo lo de abajo estaba
bloqueado por tickets sin terminar; la mayoría ya se cerró.

### Resuelto desde la primera versión

1. ~~**Modelo real (T-105 a T-109)**~~ — el pipeline existe y T-109 seleccionó LightGBM.
   Como se anticipó, **no hubo que tocar código de `ui/`**: el contrato HTTP no cambió.
   El artefacto sigue gitignoreado, así que cada quien lo genera local
   (`python -m scripts.run_training_pipeline`) o usa el fixture.
2. ~~**`cloudpickle` en el contenedor de la API**~~ — la causa real era la ruta de módulo
   grabada en el pickle, no la librería de serialización (ver sección 3). El fixture ya
   serializa con `pickle` estándar bajo `app.model.predictor` y carga en el contenedor sin
   dependencias adicionales.
3. ~~**`docker-compose.yml` con la key `version: "3.9"` obsoleta**~~ — la sacó T-114, y
   `tests/test_docker_config.py` ahora falla si alguien la reintroduce.

### Sigue pendiente

4. **Tabla de comparación de modelos**: los números de T-106/107/108 ya existen
   (`docs/T-109-model-selection.md`), pero el dashboard no los muestra. Nunca fue criterio
   de aceptación de T-111. Se puede agregar leyendo un artefacto estático — hoy habría que
   producirlo, porque `model_selection.py` devuelve el leaderboard en memoria sin
   escribirlo a disco.
5. **`Trip distance (auto)` usa haversine, no una API de ruteo real**: es una mejora
   incremental (antes era 100% manual) pero sigue siendo una subestimación sistemática de
   la distancia real en auto — sobre todo en NYC, con ríos y puentes de por medio. Pesa
   más ahora que hay un modelo real detrás: `trip_distance` es la feature más predictiva,
   el campo está **deshabilitado** y el usuario no puede corregirlo, así que el sesgo se
   traslada entero a la tarifa mostrada. Opciones: permitir sobrescribir el valor, aplicar
   un factor de rodeo (~1,3-1,4 para NYC), o integrar ruteo real (OSRM, Google Maps).
6. **`infer_ratecode` es más agresivo que las reglas reales de TLC**: marca `RatecodeID=2`
   (tarifa plana JFK) para *cualquier* trayecto que toque la zona 132, cuando en realidad
   la tarifa plana solo aplica a Manhattan ↔ JFK. Un JFK → Queens es tarifa estándar.
7. **`ui/` no tiene tests**: `infer_ratecode`, `haversine_miles` y
   `estimate_trip_distance` son funciones puras y perfectamente testeables sin levantar
   Streamlit.
8. **El choropleth hace ~263 peticiones HTTP secuenciales**: T-113 optimizó la latencia de
   petición individual, pero el caso de uso dominante del dashboard es este fan-out, donde
   manda el overhead HTTP. Un endpoint `POST /predict/batch` lo resolvería de raíz y
   aprovecharía que `transform_features` ya es vectorizado.
