# dev_fixture_model.py — Genera un models/model.pkl de mentira para desbloquear T-111
# mientras T-105-T-109 no producen un modelo real. Herramienta de desarrollo temporal,
# no es un entregable de ningún ticket: borrar o reemplazar cuando exista el modelo real.
#
# Usa cloudpickle en vez de pickle para serializar: cloudpickle guarda el código de la
# clase "por valor", no solo una referencia al módulo, así el resultado se puede cargar
# con pickle.load() normal desde cualquier proceso (p. ej. uvicorn corriendo desde api/)
# sin que ese proceso necesite poder importar este script.

from pathlib import Path

import cloudpickle

FEATURE_ORDER = [
    "PULocationID",
    "DOLocationID",
    "tpep_pickup_datetime",
    "passenger_count",
    "RatecodeID",
    "trip_distance",
]

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"


class FixtureModel:
    """Modelo de prueba: siempre predice los mismos valores fijos."""

    def predict(self, features):
        return [[18.75, 27.5] for _ in range(len(features))]


def main() -> None:
    bundle = {
        "model": FixtureModel(),
        "feature_order": FEATURE_ORDER,
        "version": "dev-fixture-0.1",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("wb") as f:
        cloudpickle.dump(bundle, f)
    print(f"Fixture model escrito en {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
