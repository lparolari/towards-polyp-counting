VENV_NAME = polypsense
VENV = $(shell conda info --base)/envs/$(VENV_NAME)/bin/
PACKAGE = polypsense

.PHONY: help
help:                  ## Show the help.
	@echo "Usage: make <target>"
	@echo ""
	@echo "Targets:"
	@grep -F "##" Makefile | grep -F -v fgrep


.PHONY: fmt
fmt:                   ## Format code using black & isort.
	@$(VENV)isort *.py ${PACKAGE}/
	@$(VENV)black *.py ${PACKAGE}/

.PHONY: test
test:                  ## Run tests.
	$(VENV)pytest -v -l --tb=short --maxfail=1 tests/	

.PHONY: test-cov
test-cov:              ## Run tests and generate coverage report.
	$(VENV)pytest -v --cov-config .coveragerc --cov=${PACKAGE} -l --tb=short --maxfail=1 tests/
	$(VENV)coverage html

.PHONY: virtualenv
virtualenv:            ## Create a virtual environment.
	@echo "creating virtualenv ..."
	@conda create -n $(VENV_NAME) python=3.10 -y

.PHONY: install
install:               ## Install dependencies.
	@echo "installing dependencies ..."
	@$(VENV)pip install -r requirements.txt
	@$(VENV)pip install -r requirements.dev.txt

.PHONY: precommit-install
precommit-install:     ## Install pre-commit hooks.
	@echo "installing pre-commit hooks ..."
	@$(VENV)pre-commit install

.PHONY: precommit-uninstall
precommit-uninstall:   ## Uninstall pre-commit hooks.
	@echo "uninstalling pre-commit hooks ..."
	@$(VENV)pre-commit uninstall

.PHONY: clean
clean:                 ## Clean unused files.
	@find ./ -name '*.pyc' -exec rm -f {} \;
	@find ./ -name '__pycache__' -exec rm -rf {} \;
	@find ./ -name 'Thumbs.db' -exec rm -f {} \;
	@find ./ -name '*~' -exec rm -f {} \;
	@rm -rf .cache
	@rm -rf .pytest_cache
	@rm -rf .mypy_cache
	@rm -rf build
	@rm -rf dist
	@rm -rf *.egg-info
	@rm -rf htmlcov
	@rm -rf .tox/
	@rm -rf docs/_build
	@rm -rf coverage.xml
	@rm -rf .coverage
