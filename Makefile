.PHONY: build run test down install fixture

install:
	pip install -r requirements-dev.txt

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
