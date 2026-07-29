"""Shared pytest fixtures and marker policy for the SCA test suite.

Design notes
------------
**Determinism is enforced, not assumed.**  ``_deterministic_ambient_state`` is
an autouse fixture that reseeds every ambient RNG before *each* test.  Without
it, test order determines the global RNG state a test inherits, so running
``pytest -k something`` gives different numbers than a full run, and a flaky
statistical test looks fine locally and fails in CI.  Library code should not
be consuming ambient randomness at all, but the fixture makes any leak
reproducible instead of intermittent.

**The suite must not silently depend on ``PYTHONHASHSEED``.**  Session setup
records whether it is set and exposes it via the ``pythonhashseed_is_set``
fixture.  Cross-process reproducibility tests explicitly *unset* it in the
child environment (see :func:`child_python`), because setting it would mask
exactly the defect they exist to catch (audit finding F9).

**Markers.**  ``slow``, ``montecarlo``, ``network``, and ``subprocess`` are
declared in ``pyproject.toml`` under ``--strict-markers``, so a typo is an
error rather than a silently-unfiltered test.  CI runs
``-m 'not slow and not network'`` on every push and the full suite nightly and
on demand.  ``--runslow`` / ``--runnetwork`` opt back in locally.

Nothing here weakens an assertion or auto-skips a failing test.  The only
skipping is for tests that genuinely cannot run in the current environment
(no network), and those report a reason via ``-ra``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from sca.utils.paths import repo_root
from sca.utils.seeding import set_global_seed, stable_rng

# --------------------------------------------------------------------------
# The canonical seed for the whole suite.  Tests that need a *different* seed
# should derive one with stable_rng(SEED, "<test name>") rather than inventing
# a magic number, so two tests never accidentally share a stream.
# --------------------------------------------------------------------------
SEED = 20260729

REPO_ROOT = repo_root()


# ==========================================================================
# CLI options and marker policy
# ==========================================================================
def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked 'slow' (full-size runs, minutes)",
    )
    parser.addoption(
        "--runnetwork",
        action="store_true",
        default=False,
        help="run tests marked 'network' (require the HuggingFace Hub)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Deselect opt-in markers unless explicitly requested.

    Deliberately implemented as a *skip with a visible reason* rather than a
    silent deselection: with ``-ra`` (set in ``pyproject.toml``) the summary
    always states how many tests were held back and why, so nobody can mistake
    a fast run for a full run.
    """
    skip_slow = pytest.mark.skip(
        reason="slow: pass --runslow (or `make test`) to include"
    )
    skip_net = pytest.mark.skip(
        reason="network: pass --runnetwork to include (needs HuggingFace Hub)"
    )
    run_slow = config.getoption("--runslow")
    run_net = config.getoption("--runnetwork")

    # `-m ...` is an explicit user request; never override it.
    marker_expr = config.getoption("markexpr", default="") or ""

    for item in items:
        if "slow" in item.keywords and not run_slow and "slow" not in marker_expr:
            item.add_marker(skip_slow)
        if "network" in item.keywords and not run_net and "network" not in marker_expr:
            item.add_marker(skip_net)


# ==========================================================================
# Ambient determinism
# ==========================================================================
@pytest.fixture(autouse=True)
def _deterministic_ambient_state():
    """Reseed every ambient RNG before each test.

    Autouse and unconditional.  This makes a single test's result independent
    of which tests ran before it, so ``pytest -k name`` reproduces the full-run
    outcome exactly.
    """
    set_global_seed(SEED)
    yield


@pytest.fixture
def seed() -> int:
    """The canonical suite seed."""
    return SEED


@pytest.fixture
def rng(request: pytest.FixtureRequest) -> np.random.Generator:
    """A per-test independent RNG stream, keyed by the test's node id.

    Two tests never share a stream, and adding a draw in one test cannot shift
    another test's numbers -- the failure mode that makes statistical tests
    mysteriously flaky after an unrelated edit.
    """
    return stable_rng(SEED, "test", request.node.nodeid)


@pytest.fixture
def pythonhashseed_is_set() -> bool:
    """Whether the ambient interpreter was launched with ``PYTHONHASHSEED``.

    Correctness must not depend on it.  Available so a test can assert that it
    passes in *both* states.
    """
    return bool(os.environ.get("PYTHONHASHSEED"))


# ==========================================================================
# Subprocess helpers -- the only way to test cross-process reproducibility
# ==========================================================================
@pytest.fixture
def child_env() -> dict:
    """Environment for a child interpreter with ``PYTHONHASHSEED`` UNSET.

    Unsetting is the whole point: hash randomisation must be *active* in the
    child, otherwise the reproducibility test passes vacuously while the bug is
    live.
    """
    env = dict(os.environ)
    env.pop("PYTHONHASHSEED", None)
    env["PYTHONPATH"] = str(REPO_ROOT)
    # Keep child runs cheap and quiet on a CPU-only box.
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


@pytest.fixture
def child_python(child_env: dict):
    """Return ``run(code, *, hash_seed=None, timeout=300) -> str`` (stdout).

    Runs ``code`` in a fresh interpreter and returns its stripped stdout,
    raising with the child's stderr on a non-zero exit.  ``hash_seed`` sets
    ``PYTHONHASHSEED`` explicitly, for tests that assert results are the same
    under *different* hash seeds.
    """

    def run(code: str, *, hash_seed: str | None = None, timeout: int = 300) -> str:
        env = dict(child_env)
        if hash_seed is not None:
            env["PYTHONHASHSEED"] = hash_seed
        proc = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(code)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(REPO_ROOT),
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise AssertionError(
                "child interpreter failed "
                f"(rc={proc.returncode})\n--- stdout ---\n{proc.stdout}\n"
                f"--- stderr ---\n{proc.stderr}"
            )
        return proc.stdout.strip()

    return run


@pytest.fixture
def repo_root_path() -> Path:
    """Absolute path of the repository root."""
    return REPO_ROOT


@pytest.fixture
def results_tmpdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect :func:`sca.utils.paths.results_dir` into a temp directory.

    Any test that exercises an experiment driver should use this, so a test run
    can never overwrite the committed ``results/`` tree.
    """
    target = tmp_path / "results"
    target.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SCA_RESULTS_DIR", str(target))
    return target


@pytest.fixture(scope="session")
def hub_available() -> bool:
    """Whether the HuggingFace Hub is reachable in this environment.

    Used by ``network``-marked tests to report an honest skip reason instead of
    a confusing connection traceback.  Never used to skip a test that could
    have run.
    """
    try:
        from huggingface_hub import HfApi

        HfApi().dataset_info("PKU-Alignment/PKU-SafeRLHF")
        return True
    except Exception:
        return False
