.PHONY: lint format install-hooks prepush

# Lint the codebase using ruff
lint:
	poetry run ruff check src/ scripts/

# Automatically fix linting errors
format:
	poetry run ruff check src/ scripts/ --fix

# Install a git pre-push hook to automatically lint before pushing
install-hooks:
	@echo "Installing pre-push git hook..."
	@echo '#!/bin/sh' > .git/hooks/pre-push
	@echo 'echo "Running linter before push..."' >> .git/hooks/pre-push
	@echo 'make lint' >> .git/hooks/pre-push
	@chmod +x .git/hooks/pre-push
	@echo "✅ Pre-push hook installed successfully!"

# Manual prepush command (can be run manually if hook isn't installed)
prepush: lint
