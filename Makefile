.PHONY: lint format install-hooks prepush

# Lint the codebase using ruff
lint:
	poetry run ruff check src/ scripts/

# Automatically fix linting errors
format:
	poetry run ruff check src/ scripts/ --fix

test:
	poetry run pytest tests/ -v
