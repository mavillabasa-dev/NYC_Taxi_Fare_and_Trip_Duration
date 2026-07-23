.PHONY: build run test down

build:
	docker compose build

run:
	docker compose up

test:
	pytest

down:
	docker compose down
