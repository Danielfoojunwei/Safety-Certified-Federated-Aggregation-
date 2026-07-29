# =============================================================================
# SCA -- developer and reviewer entry points.
#
# The README points readers at these targets, so every one of them must run.
# `make help` lists them.
#
# Deliberately NOT set anywhere in this file: PYTHONHASHSEED. Reproducibility
# is a property of the code (sca.utils.seeding.stable_hash), not of the way you
# launch it. Pinning the hash seed here would hide audit finding F9 rather than
# fix it, and `make repro-check` explicitly *unsets* it in every child process
# so the check stays able to fail.
# =============================================================================

PYTHON        ?= python3
PIP           ?= $(PYTHON) -m pip
PYTEST        ?= $(PYTHON) -m pytest

# Coverage floor. Measured, not aspirational: the fast suite reports 78% line
# coverage over sca/ at the time this floor was set (see `make coverage`), with
# the legacy pre-rebuild entry points omitted via [tool.coverage.run] in
# pyproject.toml. The floor sits a few points below the measured value so that
# ordinary refactoring does not produce spurious red, and is meant to be raised
# as sca/experiments/metrics.py (currently 0%) gains tests. Do not lower it to
# make a build pass.
COVERAGE_FLOOR ?= 75

# Reproducibility check: how many independent processes to compare.
REPRO_REPEATS ?= 3
REPRO_DIR     ?= .repro

# Experiment drivers. Override on the command line if the module names change:
#   make experiments EXPERIMENTS_CMD="python -m sca.experiments.my_driver"
EXPERIMENTS_MODULE ?= sca.experiments.run_all
FIGURES_MODULE     ?= sca.experiments.make_figures
EXPERIMENTS_CMD    ?= $(PYTHON) -m $(EXPERIMENTS_MODULE)
FIGURES_CMD        ?= $(PYTHON) -m $(FIGURES_MODULE)
SMOKE_FLAGS        ?= --smoke

# Keep CPU runs from oversubscribing a CI runner and from producing
# thread-count-dependent float reductions.
export OMP_NUM_THREADS ?= 1
export MKL_NUM_THREADS ?= 1
export TOKENIZERS_PARALLELISM ?= false

.DEFAULT_GOAL := help
.PHONY: help install install-dev test test-fast test-slow coverage repro-check \
        experiments smoke paper-figures lint clean check ci

# -----------------------------------------------------------------------------
help:
	@echo "SCA make targets"
	@echo ""
	@echo "  install        pip install -e '.[dev]'"
	@echo "  test-fast      fast suite (skips 'slow' and 'network' markers)"
	@echo "  test           full suite, including slow tests"
	@echo "  coverage       full-suite coverage, fails under $(COVERAGE_FLOOR)%"
	@echo "  repro-check    run $(REPRO_REPEATS) fresh processes and byte-diff"
	@echo "                 their outputs -- the check that catches F9"
	@echo "  smoke          fast end-to-end run of the experiment driver"
	@echo "  experiments    full experiment sweep -> results/"
	@echo "  paper-figures  regenerate every figure/table -> results/"
	@echo "  check          test-fast + coverage + repro-check (pre-push gate)"
	@echo "  clean          remove caches, coverage data, and .repro/"
	@echo ""
	@echo "  results/ is tracked in git on purpose; see .gitignore for why."

# -----------------------------------------------------------------------------
install:
	$(PIP) install -e '.[dev]'
	@$(PYTHON) -c "import sca; print('sca', sca.__version__, 'installed')"

install-dev: install

# -----------------------------------------------------------------------------
# Tests.
#
# 'slow' and 'network' are opt-in via conftest (--runslow / --runnetwork), and
# skipped tests are always reported with a reason because pyproject sets -ra.
# Nothing here silences a failure.
# -----------------------------------------------------------------------------
test-fast:
	$(PYTEST) -q

test:
	$(PYTEST) -q --runslow

test-slow:
	$(PYTEST) -q --runslow -m slow

coverage:
	$(PYTEST) --runslow \
	    --cov=sca --cov-branch \
	    --cov-report=term-missing \
	    --cov-report=xml:coverage.xml \
	    --cov-report=html:htmlcov \
	    --cov-fail-under=$(COVERAGE_FLOOR)
	@echo "HTML report: htmlcov/index.html"

# -----------------------------------------------------------------------------
# Reproducibility. THE job that would have caught F9.
#
# Runs the built-in probe (sca.utils.reprocheck) in $(REPRO_REPEATS) fresh
# interpreters with PYTHONHASHSEED UNSET -- i.e. with hash randomisation
# ACTIVE -- and byte-compares everything each run wrote. If a driver module is
# present it is wrapped too, with SCA_RESULTS_DIR pointed at a per-run scratch
# directory, so real experiment artifacts are diffed as well.
# -----------------------------------------------------------------------------
repro-check:
	@if $(PYTHON) -c "import importlib.util,sys; \
	    sys.exit(0 if importlib.util.find_spec('$(EXPERIMENTS_MODULE)') else 1)" \
	    2>/dev/null; then \
	    echo ">> repro-check: probe + $(EXPERIMENTS_MODULE) $(SMOKE_FLAGS)"; \
	    SCA_DETERMINISTIC=1 $(PYTHON) -m sca.utils.reprocheck check \
	        --repeats $(REPRO_REPEATS) --clean --workdir $(REPRO_DIR) \
	        --cmd "$(EXPERIMENTS_CMD) $(SMOKE_FLAGS)"; \
	else \
	    echo ">> repro-check: probe only ($(EXPERIMENTS_MODULE) not importable)"; \
	    $(PYTHON) -m sca.utils.reprocheck check \
	        --repeats $(REPRO_REPEATS) --clean --workdir $(REPRO_DIR); \
	fi

# Strictly stronger variant: force DIFFERENT explicit PYTHONHASHSEED values.
repro-check-strict:
	$(PYTHON) -m sca.utils.reprocheck check \
	    --repeats $(REPRO_REPEATS) --clean --vary-hashseed --workdir $(REPRO_DIR)

# -----------------------------------------------------------------------------
# Experiments.
#
# These fail loudly with an actionable message rather than silently doing
# nothing when the driver module is absent. A "successful" no-op experiment
# target is how stale results survive.
# -----------------------------------------------------------------------------
define _require_module
	@$(PYTHON) -c "import importlib.util, sys; \
	spec = importlib.util.find_spec('$(1)'); \
	sys.exit(0) if spec else (sys.stderr.write( \
	  '\nERROR: module $(1) is not importable.\n' \
	  'The experiment driver has not landed yet, or was renamed.\n' \
	  'Point this target at the real one, e.g.\n' \
	  '    make $(2) $(3)=\"python -m sca.experiments.<driver>\"\n\n'), sys.exit(1))"
endef

smoke:
	$(call _require_module,$(EXPERIMENTS_MODULE),smoke,EXPERIMENTS_CMD)
	$(EXPERIMENTS_CMD) $(SMOKE_FLAGS)

experiments:
	$(call _require_module,$(EXPERIMENTS_MODULE),experiments,EXPERIMENTS_CMD)
	$(EXPERIMENTS_CMD)
	@echo "results written to results/ (tracked in git -- commit the diff)"

paper-figures:
	$(call _require_module,$(FIGURES_MODULE),paper-figures,FIGURES_CMD)
	$(FIGURES_CMD)
	@echo "figures written to results/figures/"

# -----------------------------------------------------------------------------
lint:
	@$(PYTHON) -m ruff check sca tests 2>/dev/null || \
	    echo "(ruff not installed; skipping -- lint is advisory, not a gate)"
	@$(PYTHON) -m compileall -q sca tests >/dev/null && echo "syntax ok"

# Pre-push gate. Same three things CI runs on every push.
check: test-fast repro-check
	$(PYTEST) -q --cov=sca --cov-branch --cov-report=term \
	    --cov-fail-under=$(COVERAGE_FLOOR)

ci: check

# -----------------------------------------------------------------------------
clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov $(REPRO_DIR)
	rm -f .coverage .coverage.* coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} +
	@echo "clean. results/ was NOT touched -- it is a tracked artifact."
