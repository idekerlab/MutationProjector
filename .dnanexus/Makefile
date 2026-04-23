.PHONY: clean clean-test clean-pyc clean-build docs help docker
.DEFAULT_GOAL := help

# 1. Find all source files in the platform directory
MD_SOURCES := $(wildcard MutationProjector/*.md)
SH_SOURCES := $(wildcard MutationProjector/src/*.sh)
PY_SOURCES := $(wildcard MutationProjector/test/*.py)

# 2. Convert source paths to build paths
MD_TARGETS := $(patsubst MutationProjector/%.md, build/MutationProjector/%.md, $(MD_SOURCES))
SH_TARGETS := $(patsubst MutationProjector/src/%.sh, build/MutationProjector/src/%.sh, $(SH_SOURCES))
PY_TARGETS := $(patsubst MutationProjector/test/%.py, build/MutationProjector/test/%.py, $(PY_SOURCES))

# Other file dependencies files
DEPS := $(wildcard docker/*.*) conf.yml


define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

clean-build: ## remove build artifacts
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +

clean-pyc: ## remove Python file artifacts
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test: ## remove test and coverage artifacts
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

docker: ## Builds MutationProjector Docker Image
	@cd .. ;\
	cv=`grep 'version' .dnanexus/conf.yml | sed "s/^.*: *\"//" | sed "s/\".*//"`; \
	docker buildx build --platform linux/amd64 -t digitaltumors/mutationprojector:$$cv -f .dnanexus/docker/Dockerfile .; \

