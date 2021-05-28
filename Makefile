.DEFAULT_GOAL := help

IMAGE   := dail
VERSION = $(shell make --no-print-directory version)

COLOR=\033[0;32m
NC=\033[0m

OBJECTS := $$(find . -type f \( -name *.py -not -path "./dail/envs/reacher/*" -not -path "./dail/envs/adroit/*" -not -path "./dail/environments/*" \))

###### Development

.PHONY: update-requirements
update-requirements: ## Update requirements files
	@poetry export --output requirements.txt

.PHONY: -format
format: ## Format code
	@printf "\n${COLOR}=== Formatting code...${NC}\n"
	-poetry run pyupgrade --py36-plus $(OBJECTS)
	poetry run isort --settings-path pyproject.toml $(OBJECTS)
	poetry run black --config pyproject.toml ./

.PHONY: test
test: test-lint test-unit test-safety ## Run all tests by decreasing priority

.PHONY: test-lint
test-lint: ## Run code linting tests
	@printf "\n${COLOR}=== Running code linting tests...${NC}\n"
	poetry run isort --settings-path pyproject.toml $(OBJECTS)
	poetry run black --config pyproject.toml --diff --check ./
	poetry run darglint -v 2 $(OBJECTS)
	poetry run flake8 --config setup.cfg $(OBJECTS)
	poetry run pylint --rcfile pyproject.toml $(OBJECTS)
	poetry run mypy --config-file setup.cfg $(OBJECTS)

.PHONY: test-unit
test-unit: ## Run unit tests
	@printf "\n${COLOR}=== Running unit tests...${NC}\n"
	@poetry run pytest

.PHONY: test-safety
test-safety: ## Run dependencies safety tests
	@printf "\n${COLOR}=== Running dependencies safety tests...${NC}\n"
	poetry check
	poetry run pip check
	poetry run safety check --full-report

.PHONY: build-dev
build-dev: ## Build an image for development and testing
	@printf "\n${COLOR}=== Building $(IMAGE):dev image...${NC}\n"
	@docker build \
		-t $(IMAGE):dev \
		-f Dockerfile .

.PHONY: shell
shell: ## Launch a shell inside the development image
	@printf "\n${COLOR}=== Launching a shell inside the local $(IMAGE):dev image...${NC}\n"
	docker run -ti --rm \
		--entrypoint /bin/bash \
		-u $$(id -u):$$(id -g) \
		$(IMAGE):dev \
		$(CMD)

###### Deployment

.PHONY: build
build: clean update-requirements ## Build and save an image ready to run the grader in production
	@printf "\n${COLOR}=== Building $(IMAGE):$(VERSION) image...${NC}\n"
	@docker build --compress \
		-t $(IMAGE):$(VERSION) \
		-f Dockerfile .
	@printf "\n${COLOR}=== Saving $(IMAGE):$(VERSION) image...${NC}\n"
	@mkdir -p dist
	@docker save \
		-o dist/$(IMAGE)_$(VERSION).tar \
		$(IMAGE):$(VERSION)

###### Clean up

.PHONY: clean-local
clean: clean-tmp clean-images ## Clean temporary files, directories and remove all the built images

.PHONY: clean-tmp
clean-tmp: ## Clean temporary files and directories
	@printf "\n${COLOR}=== Cleaning temporary files and directories...${NC}\n"
	@rm -rf dist
	@-find . -regex "^.*\(__pycache__\|\.pytest_cache\|\.coverage\(\|\..*\)\)" -exec rm -rf {} +

.PHONY: clean-images
clean-images: ## Remove all the built images
	@printf "\n${COLOR}=== Removing all the built $(IMAGE):* images...${NC}\n"
	@-docker rmi --force $$(docker images -q $(IMAGE) | uniq)

###### Additional commands

.PHONY: version
version: ## Print the current version
	@cat pyproject.toml | grep -o '\([0-9]\+\.\?\)\{3\}' | head -1

ESCAPE = 
.PHONY: help
help: ## Print this help
	@grep -E '^([a-zA-Z_-]+:.*?## .*|######* .+)$$' Makefile \
		| sed 's/######* \(.*\)/\n        $(ESCAPE)[1;31m\1$(ESCAPE)[0m/g' \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[33m%-20s\033[0m %s\n", $$1, $$2}'
