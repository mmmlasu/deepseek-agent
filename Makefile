.PHONY: install test lint typecheck check
install:
	python3 -m pip install -e '.[dev]'
test:
	pytest
lint:
	ruff check .
typecheck:
	mypy src
check: lint typecheck test
