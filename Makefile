.PHONY: build run test down install fixture lint format

install:
	pip install -r requirements-dev.txt
	pre-commit install

lint:
	ruff check .
	black --check .

format:
	ruff check . --fix
	black .

fixture:
	python scripts/dev_fixture_model.py

build:
	docker compose build

run:
	docker compose up

test:
	pytest

down:
	docker compose down
