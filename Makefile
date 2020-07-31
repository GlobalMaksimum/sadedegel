# The binary to build (just the basename).
MODULE := sadedegel

# Where to push the docker image.
REGISTRY ?= docker.pkg.github.com/globalmaksimum/teknofest

IMAGE := $(REGISTRY)/$(MODULE)

# This version-strategy uses git tags to set the version string
TAG := $(shell git describe --tags --always --dirty)

BLUE='\033[0;34m'
NC='\033[0m' # No Color

test:
	@python -c "import nltk; nltk.download('punkt')"
	@pytest

whl:
	@python3 setup.py sdist bdist_wheel

build-prod:
	@echo "\n${BLUE}Building Production image with labels:\n"
	@echo "name: $(MODULE)"
	@echo "version: $(VERSION)${NC}\n"
	@sed                                     \
	    -e 's|{NAME}|$(MODULE)|g'            \
	    -e 's|{VERSION}|$(VERSION)|g'        \
	    prod.Dockerfile | docker build -t $(IMAGE):$(VERSION) -f- .

push: build-prod
	@echo "\n${BLUE}Pushing image to GitHub Docker Registry...${NC}\n"
	@docker push $(IMAGE):$(VERSION)

login:
	@docker login docker.pkg.github.com -u $(USER)

build-dev:
	@echo "\n${BLUE}Building Development image with labels:\n"
	@echo "name: $(MODULE)"
	@echo "version: $(TAG)${NC}\n"
	@sed                                 \
	    -e 's|{NAME}|$(MODULE)|g'        \
	    -e 's|{VERSION}|$(TAG)|g'        \
	    dev.Dockerfile | docker build -t $(IMAGE):$(TAG) -f- .

shell: build-dev
	@echo "\n${BLUE}Launching a shell in the containerized build environment...${NC}\n"
		@docker run                                                 \
			-ti                                                     \
			--rm                                                    \
			--entrypoint /bin/bash                                  \
			-u $$(id -u):$$(id -g)                                  \
			$(IMAGE):$(TAG)										    \
			$(CMD)

lint:
	@echo "\n${BLUE}Running Pylint against source and test files...${NC}\n"
	@pylint --rcfile=setup.cfg **/*.py
	@echo "\n${BLUE}Running Flake8 against source and test files...${NC}\n"
	@flake8  **/*.py
	@echo "\n${BLUE}Running Bandit against source files...${NC}\n"
	@bandit -r --ini setup.cfg

clean:
	rm -rf .pytest_cache .coverage .pytest_cache coverage.xml build dist sadedegel.egg-info sadedegel/ml/model/sbd.test.pickle
