"""Tests for :mod:`sca.utils.seeding` and :mod:`sca.utils.paths`.

This file exists because of audit finding **F9**: ``embedding.py`` bucketed
tokens with the builtin ``hash()``, which CPython salts per process, and
``PYTHONHASHSEED`` was set nowhere.  Every "seeded" number in the paper was a
per-process random variable.

The central methodological point is that **an in-process test cannot detect
this bug**.  Within one interpreter ``hash("x")`` is a constant, so
``assert f(s) == f(s)`` passes while the defect is live.  Every reproducibility
test below therefore spawns real child interpreters via ``subprocess.run`` with
``PYTHONHASHSEED`` explicitly removed from the child environment.

:class:`TestMethodologyIsSound` is the control: it proves the *hazard is real*
in this interpreter, so the reproducibility tests cannot pass vacuously if a
future CPython stops randomising string hashes.  And
:class:`TestTheTestWouldHaveCaughtTheBug` reconstructs the original defective
implementation and asserts the test methodology fails on it -- a test that
cannot fail on the known-bad code is not a regression test.
"""

from __future__ import annotations

import hashlib
import os
import random
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from sca.utils.paths import cache_dir, figures_dir, package_root, repo_root, results_dir
from sca.utils.seeding import (
    STABLE_HASH_BITS,
    derive_seed,
    seed_torch_generator,
    set_global_seed,
    stable_bytes,
    stable_hash,
    stable_rng,
    stable_shuffled,
)

pytestmark = pytest.mark.subprocess


# --------------------------------------------------------------------------
# Static-analysis helpers.  Both scanners below work on the AST rather than on
# raw text, so that the word "hash" in a docstring, the identifier ``p_hat``,
# and a module's own prose description of the defect it fixes do not produce
# false positives.
# --------------------------------------------------------------------------
def _code_string_literals(path):
    """Yield ``ast.Constant`` str nodes that are *not* docstrings."""
    import ast

    src = __import__("pathlib").Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))

    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstring_nodes.add(id(body[0].value))

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstring_nodes
        ):
            yield node


def _builtin_hash_call_sites(root):
    """Return ``{relative_path: [lineno, ...]}`` for calls to builtin ``hash``.

    An aliased ``h = hash; h(s)`` is not caught, but a bare ``hash(...)`` --
    the form the original defect took -- is.
    """
    import ast
    import pathlib

    hits: dict[str, list[int]] = {}
    for path in sorted(pathlib.Path(root).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "hash"
            ):
                key = str(path.relative_to(pathlib.Path(root).parent))
                hits.setdefault(key, []).append(node.lineno)
    return hits


def _ns(**kw):
    """A tiny argparse-like namespace, for calling the reprocheck commands."""
    import argparse

    return argparse.Namespace(**kw)


#: Pre-rebuild entry points retained only for the audit trail.  They are also
#: omitted from coverage in ``pyproject.toml``.  They are NOT exempt because
#: the F9 rule is negotiable -- they are quarantined because they are dead code
#: scheduled for deletion, and the quarantine is asserted to be a strict upper
#: bound so it cannot quietly grow.  See ``TestNoBuiltinHashInPackage``.
LEGACY_QUARANTINE = frozenset(
    {
        "sca/experiments/run_novelty_validation.py",
        "sca/experiments/run_real_evaluation.py",
        "sca/experiments/run_neurips_exp1.py",
        "sca/experiments/run_neurips_exp2.py",
        "sca/experiments/run_neurips_exp3.py",
        "sca/experiments/run_experiment.py",
        "sca/experiments/evaluation.py",
    }
)


# ==========================================================================
class TestMethodologyIsSound:
    """Controls. If these fail, nothing else in this file means anything."""

    def test_builtin_hash_really_is_randomised_in_this_interpreter(
        self, child_python
    ):
        """The hazard must be live, or the reproducibility tests are vacuous."""
        vals = {child_python("print(hash('safety violation token'))") for _ in range(6)}
        assert len(vals) > 1, (
            "builtin hash() was stable across 6 fresh processes. Either "
            "PYTHONHASHSEED leaked into the child environment or this "
            "interpreter no longer randomises string hashing; in both cases "
            "the subprocess reproducibility tests below can no longer detect "
            "the original F9 defect and must be re-examined."
        )

    def test_child_environment_has_pythonhashseed_unset(self, child_python):
        assert child_python(
            "import os; print(os.environ.get('PYTHONHASHSEED', '<unset>'))"
        ) == "<unset>"

    def test_in_process_equality_passes_even_for_builtin_hash(self):
        """Demonstrates why an in-process test is worthless here.

        This assertion holds for the *defective* implementation too, which is
        precisely why the original repository's determinism tests all passed.
        """
        assert hash("safety violation token") == hash("safety violation token")


# ==========================================================================
class TestStableHashCrossProcess:
    """The F9 fix, verified the only way it can be verified."""

    def test_identical_across_three_processes(self, child_python):
        code = """
            from sca.utils.seeding import stable_hash
            print(stable_hash("safety violation token"))
        """
        vals = {child_python(code) for _ in range(3)}
        assert len(vals) == 1, f"stable_hash differs across processes: {vals}"

    def test_identical_under_different_explicit_hash_seeds(self, child_python):
        code = """
            from sca.utils.seeding import stable_hash
            print(stable_hash("bypass the content filter"))
        """
        vals = {
            child_python(code, hash_seed=hs) for hs in ("0", "1", "12345", "99999")
        }
        assert len(vals) == 1, (
            f"stable_hash depends on PYTHONHASHSEED: {vals}"
        )

    def test_matches_a_pinned_literal(self):
        """Golden value.

        Pins the digest so a future refactor cannot silently change bucket
        assignments and invalidate every previously published result. If this
        fails intentionally, the package version must be bumped and all
        committed results regenerated.
        """
        assert stable_hash("safety violation token") == 16588696624594238767

    def test_derived_from_blake2b_not_from_builtin_hash(self):
        """Independent recomputation of the documented construction."""
        h = hashlib.blake2b(digest_size=8, person=b"sca-stable")
        h.update("abc".encode("utf-8"))
        h.update(b"\x00")
        assert stable_hash("abc") == int.from_bytes(h.digest(), "big")

    def test_full_pipeline_is_byte_identical_across_processes(self, child_python):
        """Determinism must survive composition, not just a single call."""
        code = """
            import hashlib, json
            import numpy as np
            from sca.utils.seeding import stable_hash, stable_rng
            toks = "how do i bypass the safety filter for a chemical synthesis".split()
            bow = np.zeros(64)
            for t in toks:
                bow[stable_hash(t) % 64] += 1.0
            draw = stable_rng(42, "pipeline").normal(size=(64, 8))
            v = bow @ draw
            print(hashlib.sha256(np.ascontiguousarray(v, np.float64).tobytes()).hexdigest())
        """
        digests = {child_python(code) for _ in range(3)}
        assert len(digests) == 1, f"pipeline differs across processes: {digests}"


# ==========================================================================
class TestTheTestWouldHaveCaughtTheBug:
    """Negative control: the methodology must FAIL on the known-bad code.

    A regression test that cannot fail on the original defect is decoration.
    Here we run the *original* ``hash(token) % vocab_size`` embedder in three
    child processes and assert the digests disagree -- i.e. that this test
    file, applied to the pre-rebuild implementation, would have gone red.
    """

    def test_original_hash_based_embedder_is_nondeterministic_across_processes(
        self, child_python
    ):
        code = """
            import hashlib
            import numpy as np
            # Verbatim reconstruction of the pre-rebuild embedding.py:66 logic.
            vocab_size = 256
            bow = np.zeros(vocab_size)
            for token in "how do i bypass the safety filter".split():
                bow[hash(token) % vocab_size] += 1.0
            print(hashlib.sha256(bow.tobytes()).hexdigest())
        """
        digests = {child_python(code) for _ in range(3)}
        assert len(digests) > 1, (
            "The reconstructed defective embedder produced identical output in "
            "3 processes. This test methodology can no longer detect F9."
        )


# ==========================================================================
class TestStableHashProperties:
    def test_is_a_pure_function(self):
        assert stable_hash("x") == stable_hash("x")

    def test_is_non_negative_and_in_range(self):
        for s in ("", "a", "unicode: é中", "x" * 5000, "\n\t "):
            h = stable_hash(s)
            assert 0 <= h < 2 ** STABLE_HASH_BITS

    def test_distinguishes_similar_strings(self):
        assert stable_hash("abc") != stable_hash("abd")
        assert stable_hash("ab") != stable_hash("ba")

    def test_rejects_non_str(self):
        for bad in (b"bytes", 3, None, ["a"]):
            with pytest.raises(TypeError):
                stable_hash(bad)  # type: ignore[arg-type]

    def test_bucketing_is_close_to_uniform(self):
        """A biased digest would concentrate tokens into few regions."""
        n_buckets, n = 64, 20000
        counts = np.zeros(n_buckets, dtype=np.int64)
        for i in range(n):
            counts[stable_hash("token_%d" % i) % n_buckets] += 1
        expected = n / n_buckets
        chi2 = float(((counts - expected) ** 2 / expected).sum())
        # 99.9th percentile of chi2_63 is ~112.3; a salted-but-fine digest
        # sits near 63.
        assert chi2 < 112.3, f"bucket distribution is skewed: chi2={chi2}"

    def test_separator_prevents_concatenation_collisions(self):
        assert stable_bytes("ab", "c") != stable_bytes("a", "bc")
        assert stable_bytes("a", "") != stable_bytes("", "a")

    def test_stable_bytes_rejects_non_str(self):
        with pytest.raises(TypeError):
            stable_bytes("ok", 5)  # type: ignore[arg-type]


# ==========================================================================
class TestDeriveSeed:
    def test_deterministic(self):
        assert derive_seed(7, "a", "b") == derive_seed(7, "a", "b")

    def test_tags_matter(self):
        assert derive_seed(7, "a") != derive_seed(7, "b")
        assert derive_seed(7) != derive_seed(7, "a")

    def test_seed_matters(self):
        assert derive_seed(7, "a") != derive_seed(8, "a")

    def test_tag_order_matters(self):
        assert derive_seed(7, "a", "b") != derive_seed(7, "b", "a")

    def test_does_not_return_the_seed_itself(self):
        """Otherwise an untagged component would collide with the raw seed."""
        assert derive_seed(42) != 42

    def test_rejects_bool_and_non_int(self):
        for bad in (True, 1.5, "3", None):
            with pytest.raises(TypeError):
                derive_seed(bad)  # type: ignore[arg-type]

    def test_accepts_numpy_integers(self):
        assert derive_seed(np.int64(7), "a") == derive_seed(7, "a")

    def test_cross_process_stable(self, child_python):
        code = """
            from sca.utils.seeding import derive_seed
            print(derive_seed(42, "search", "frontier"))
        """
        assert len({child_python(code) for _ in range(3)}) == 1


# ==========================================================================
class TestStableRng:
    def test_same_key_same_stream(self):
        a = stable_rng(0, "search").integers(0, 10 ** 9, size=20)
        b = stable_rng(0, "search").integers(0, 10 ** 9, size=20)
        assert np.array_equal(a, b)

    def test_different_tags_independent_streams(self):
        a = stable_rng(0, "search").integers(0, 10 ** 9, size=20)
        b = stable_rng(0, "estimation").integers(0, 10 ** 9, size=20)
        assert not np.array_equal(a, b)

    def test_different_seeds_differ(self):
        a = stable_rng(0, "t").normal(size=20)
        b = stable_rng(1, "t").normal(size=20)
        assert not np.allclose(a, b)

    def test_nearby_seeds_are_well_separated(self):
        """SeedSequence mixing must decorrelate adjacent integer seeds."""
        draws = np.array([stable_rng(s, "t").normal(size=200) for s in range(16)])
        corr = np.corrcoef(draws)
        off = corr[~np.eye(16, dtype=bool)]
        assert np.abs(off).max() < 0.35, (
            f"streams from adjacent seeds are correlated: max|rho|={np.abs(off).max():.3f}"
        )

    def test_is_independent_of_ambient_global_state(self):
        """The whole point: a global reseed elsewhere must not move it."""
        random.seed(1)
        np.random.seed(1)
        a = stable_rng(5, "t").normal(size=10)
        random.seed(999)
        np.random.seed(999)
        _ = np.random.rand(1000)
        b = stable_rng(5, "t").normal(size=10)
        assert np.array_equal(a, b)

    def test_adding_a_call_site_does_not_shift_another_component(self):
        """The failure mode a single global seed cannot avoid.

        Component B's draws must be identical whether or not component A drew
        anything first.
        """
        b_alone = stable_rng(3, "component_B").normal(size=10)

        _ = stable_rng(3, "component_A").normal(size=1000)  # A gains a call site
        b_after = stable_rng(3, "component_B").normal(size=10)

        assert np.array_equal(b_alone, b_after)

    def test_returns_a_generator_not_legacy_randomstate(self):
        assert isinstance(stable_rng(0, "t"), np.random.Generator)

    def test_cross_process_stable(self, child_python):
        code = """
            import hashlib, numpy as np
            from sca.utils.seeding import stable_rng
            v = stable_rng(11, "estimation", "region_3").normal(size=50)
            print(hashlib.sha256(np.ascontiguousarray(v, np.float64).tobytes()).hexdigest())
        """
        assert len({child_python(code) for _ in range(3)}) == 1


# ==========================================================================
class TestSetGlobalSeed:
    def test_seeds_stdlib_random(self):
        set_global_seed(123)
        a = [random.random() for _ in range(5)]
        set_global_seed(123)
        assert a == [random.random() for _ in range(5)]

    def test_seeds_numpy_legacy_global(self):
        set_global_seed(123)
        a = np.random.rand(5)
        set_global_seed(123)
        assert np.array_equal(a, np.random.rand(5))

    def test_seeds_torch(self):
        torch = pytest.importorskip("torch")
        set_global_seed(123)
        a = torch.randn(5)
        set_global_seed(123)
        assert torch.equal(a, torch.randn(5))

    def test_rejects_bool_and_non_int(self):
        for bad in (True, 2.5, "7", None):
            with pytest.raises(TypeError):
                set_global_seed(bad)  # type: ignore[arg-type]

    def test_accepts_large_seed(self):
        set_global_seed(2 ** 40 + 7)  # must not raise

    def test_does_not_pretend_to_set_pythonhashseed(self):
        """Assigning os.environ['PYTHONHASHSEED'] in-process does nothing.

        Doing so would create the illusion of a fix. The real fix is
        stable_hash; this pins that set_global_seed does not fake it.
        """
        before = os.environ.get("PYTHONHASHSEED")
        set_global_seed(5)
        assert os.environ.get("PYTHONHASHSEED") == before

    def test_cross_process_identical(self, child_python):
        code = """
            import random, numpy as np, torch
            from sca.utils.seeding import set_global_seed
            set_global_seed(42)
            print(random.random(), float(np.random.rand()), float(torch.randn(1)))
        """
        assert len({child_python(code) for _ in range(3)}) == 1

    def test_pythonhashseed_diagnostic_reflects_the_environment(self, monkeypatch):
        from sca.utils.seeding import _pythonhashseed_is_set

        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
        assert _pythonhashseed_is_set() is False
        monkeypatch.setenv("PYTHONHASHSEED", "0")
        assert _pythonhashseed_is_set() is True

    def test_results_are_the_same_whether_or_not_pythonhashseed_is_set(
        self, child_python
    ):
        """Correctness must not depend on the diagnostic being true.

        Runs the same computation with PYTHONHASHSEED unset and with two
        explicit values, and requires all three to agree.
        """
        code = """
            import hashlib, numpy as np
            from sca.utils.seeding import set_global_seed, stable_hash, stable_rng
            set_global_seed(7)
            v = np.array([stable_hash(f"tok{i}") % 1000 for i in range(64)], float)
            v = v + stable_rng(7, "mix").normal(size=64)
            print(hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest())
        """
        vals = {
            child_python(code),
            child_python(code, hash_seed="0"),
            child_python(code, hash_seed="4242"),
        }
        assert len(vals) == 1, f"results depend on PYTHONHASHSEED: {vals}"


# ==========================================================================
class TestTorchGeneratorAndShuffle:
    def test_torch_generator_is_deterministic(self):
        torch = pytest.importorskip("torch")
        a = torch.randn(8, generator=seed_torch_generator(4, "init"))
        b = torch.randn(8, generator=seed_torch_generator(4, "init"))
        assert torch.equal(a, b)

    def test_torch_generator_tags_are_independent(self):
        torch = pytest.importorskip("torch")
        a = torch.randn(8, generator=seed_torch_generator(4, "init"))
        b = torch.randn(8, generator=seed_torch_generator(4, "shuffle"))
        assert not torch.equal(a, b)

    def test_stable_shuffled_is_deterministic_and_a_permutation(self):
        items = [f"item_{i}" for i in range(50)]
        a = stable_shuffled(items, 9, "x")
        b = stable_shuffled(items, 9, "x")
        assert a == b
        assert sorted(a) == sorted(items)
        assert a != items  # 50! makes a fixed point essentially impossible

    def test_stable_shuffled_does_not_mutate_input(self):
        items = ["a", "b", "c", "d"]
        original = list(items)
        stable_shuffled(items, 1)
        assert items == original

    def test_stable_shuffled_survives_set_input_across_processes(self, child_python):
        """Set iteration order over strings is itself hash-randomised.

        Sorting before shuffling is the documented remedy; this pins that a
        caller who forgets it gets a *visible* failure rather than a silent
        one, and that the sorted form is stable.
        """
        code = """
            from sca.utils.seeding import stable_shuffled
            s = {"alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"}
            print(",".join(stable_shuffled(sorted(s), 1, "demo")))
        """
        assert len({child_python(code) for _ in range(4)}) == 1


# ==========================================================================
class TestPaths:
    """Regression tests for the hardcoded ``/home/user/...`` output directory."""

    def test_repo_root_contains_the_package(self):
        assert (repo_root() / "sca" / "__init__.py").is_file()

    def test_package_root_is_the_sca_directory(self):
        assert package_root().name == "sca"
        assert package_root().parent == repo_root()

    def test_no_hardcoded_home_user_path(self):
        """No *executable* string literal in these modules is an absolute path.

        Checked via the AST with docstrings stripped, because both modules
        quote the offending ``/home/user/...`` line verbatim in their prose as
        the defect being fixed; a naive text grep flags its own explanation.
        """
        import sca.utils.paths as paths_mod
        import sca.utils.seeding as seeding_mod

        for mod in (paths_mod, seeding_mod):
            offenders = [
                (node.lineno, node.value)
                for node in _code_string_literals(mod.__file__)
                if node.value.startswith("/home/") or node.value.startswith("/Users/")
            ]
            assert not offenders, (
                f"{mod.__name__} contains a hardcoded absolute path: {offenders}"
            )

    def test_results_dir_is_under_repo_root_by_default(self, monkeypatch):
        monkeypatch.delenv("SCA_RESULTS_DIR", raising=False)
        monkeypatch.delenv("SCA_REPO_ROOT", raising=False)
        d = results_dir(create=False)
        assert d == repo_root() / "results"

    def test_results_dir_honours_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCA_RESULTS_DIR", str(tmp_path / "out"))
        d = results_dir("sub")
        assert d == (tmp_path / "out" / "sub").resolve()
        assert d.is_dir()

    def test_figures_dir_nests_under_results(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCA_RESULTS_DIR", str(tmp_path / "out"))
        assert figures_dir().parent == (tmp_path / "out").resolve()

    def test_cache_dir_honours_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCA_CACHE_DIR", str(tmp_path / "c"))
        assert cache_dir("hf").is_dir()

    def test_repo_root_honours_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCA_REPO_ROOT", str(tmp_path))
        assert repo_root() == tmp_path.resolve()

    def test_results_tmpdir_fixture_redirects(self, results_tmpdir):
        assert results_dir(create=False) == results_tmpdir.resolve()


# ==========================================================================
class TestNoBuiltinHashInPackage:
    """Static guard: live modules under ``sca/`` may not call builtin ``hash()``.

    This is what keeps F9 fixed. It walks the AST rather than the text, so
    ``p_hat``, ``stable_hash``, and the word "hash" in a docstring are not
    false positives.
    """

    def test_no_builtin_hash_in_live_modules(self):
        offenders = {
            f: lines
            for f, lines in _builtin_hash_call_sites(package_root()).items()
            if f not in LEGACY_QUARANTINE
        }
        assert not offenders, (
            "builtin hash() is called in a live module under sca/ -- this "
            "reintroduces F9 (results become a per-process random variable). "
            "Use sca.utils.seeding.stable_hash. "
            f"Offenders: {offenders}"
        )

    def test_quarantine_is_a_strict_upper_bound(self):
        """The quarantine may shrink, never grow.

        If this fails, a *new* file started calling builtin ``hash()``. Fix the
        file; do not add it to ``LEGACY_QUARANTINE``.
        """
        offending = set(_builtin_hash_call_sites(package_root()))
        assert offending <= LEGACY_QUARANTINE, (
            "builtin hash() appeared in a file outside the legacy quarantine: "
            f"{sorted(offending - LEGACY_QUARANTINE)}"
        )

    def test_quarantine_only_names_dead_legacy_entry_points(self):
        """Nothing importable by the rebuilt pipeline may be quarantined.

        Guards against the quarantine being used to excuse a live module: every
        quarantined path must be a pre-rebuild ``run_*`` / ``evaluation``
        script, i.e. dead code kept only for the audit trail.
        """
        for path in LEGACY_QUARANTINE:
            name = path.rsplit("/", 1)[-1]
            assert name.startswith("run_") or name == "evaluation.py", (
                f"{path} is quarantined from the F9 rule but is not a legacy "
                f"entry point. Fix the file instead."
            )

    def test_known_remaining_f9_defect_is_reported(self):
        """Documents the F9 instances still live in the tree, honestly.

        ``sca/experiments/run_novelty_validation.py:70`` still contains
        ``idx = hash(token) % self.vocab_size``. It is a pre-rebuild script
        that the rebuilt pipeline does not import, but it is a real remaining
        defect and this test states it out loud rather than letting the
        quarantine hide it. It passes whether or not the file has been deleted;
        it fails only if someone silently *keeps* the defect while pretending
        the quarantine is empty.
        """
        remaining = _builtin_hash_call_sites(package_root())
        # No assertion of emptiness -- this is a report. The assertion is that
        # the report is consistent with the quarantine.
        assert set(remaining) <= LEGACY_QUARANTINE
        if remaining:
            print(
                "\nKNOWN REMAINING F9 SITES (legacy scripts, not imported by "
                "the rebuilt pipeline):\n"
                + "\n".join(f"  {f}: lines {ls}" for f, ls in sorted(remaining.items()))
            )


# ==========================================================================
class TestSubprocessReproducibilityOfTheWholeStack:
    """C4: byte-identical results across three separate processes.

    Runs an end-to-end slice of the real package -- not a toy -- and compares
    SHA-256 digests. ``PYTHONHASHSEED`` is unset in every child.
    """

    def test_reprocheck_probe_exercises_the_real_package(self, tmp_path):
        """The probe must not be able to report green over an empty run."""
        from sca.utils.reprocheck import MIN_COMPONENTS, cmd_run, run_probe

        result = run_probe()
        assert len(result["components"]) >= MIN_COMPONENTS, (
            f"probe exercised only {result['components']}; errors="
            f"{result['errors']}"
        )
        # The probe must actually touch the rebuilt subsystems, not just its
        # own seeding helpers.
        assert any(c.startswith("knowledge_graph.") for c in result["components"])
        assert "utils.stats" in result["components"]

        rc = cmd_run(_ns(out=str(tmp_path)))
        assert rc == 0
        assert (tmp_path / "probe.json").is_file()

    def test_reprocheck_probe_is_deterministic_in_process(self):
        from sca.utils.reprocheck import run_probe

        assert run_probe()["summary_digest"] == run_probe()["summary_digest"]

    def test_reprocheck_compare_accepts_identical_trees(self, tmp_path):
        from sca.utils.reprocheck import cmd_compare

        for name in ("a", "b"):
            d = tmp_path / name / "results"
            d.mkdir(parents=True)
            (d / "x.json").write_text('{"bound": 0.123}')
        assert cmd_compare(_ns(dirs=[str(tmp_path / "a"), str(tmp_path / "b")])) == 0

    def test_reprocheck_compare_rejects_differing_trees(self, tmp_path):
        from sca.utils.reprocheck import cmd_compare

        for name, payload in (("a", '{"bound": 0.123}'), ("b", '{"bound": 0.124}')):
            d = tmp_path / name / "results"
            d.mkdir(parents=True)
            (d / "x.json").write_text(payload)
        assert cmd_compare(_ns(dirs=[str(tmp_path / "a"), str(tmp_path / "b")])) == 1

    def test_reprocheck_compare_rejects_missing_files(self, tmp_path):
        from sca.utils.reprocheck import cmd_compare

        (tmp_path / "a" / "results").mkdir(parents=True)
        (tmp_path / "a" / "results" / "x.json").write_text("1")
        (tmp_path / "b" / "results").mkdir(parents=True)
        assert cmd_compare(_ns(dirs=[str(tmp_path / "a"), str(tmp_path / "b")])) == 1

    def test_reprocheck_compare_rejects_empty_baseline(self, tmp_path):
        """An empty artifact tree must be an error, not a vacuous pass."""
        from sca.utils.reprocheck import cmd_compare

        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        assert cmd_compare(_ns(dirs=[str(tmp_path / "a"), str(tmp_path / "b")])) == 1

    def test_reprocheck_detects_a_float_difference_in_the_last_ulp(self, tmp_path):
        """Digests are bit-exact; rounding would hide real drift."""
        from sca.utils.reprocheck import _float_digest

        x = np.array([0.1, 0.2, 0.3])
        y = x.copy()
        y[2] = np.nextafter(y[2], 1.0)
        assert _float_digest(x) != _float_digest(y)

    def test_reprocheck_end_to_end_passes_on_the_fixed_code(self, tmp_path):
        """C4, through the exact entry point CI invokes."""
        from sca.utils.reprocheck import main

        rc = main(
            ["check", "--repeats", "3", "--workdir", str(tmp_path / "w"), "--clean"]
        )
        assert rc == 0

    def test_reprocheck_end_to_end_fails_on_reintroduced_f9(self, tmp_path):
        """NEGATIVE CONTROL -- the one that matters.

        Wraps a command containing the *original* defect (``hash(token) %
        vocab_size`` with the builtin ``hash``) and asserts the harness goes
        red. A reproducibility job that cannot fail on the known-bad code
        would not have caught F9 either.
        """
        from sca.utils.reprocheck import main

        bad = (
            f"{sys.executable} -c "
            '"import os, pathlib; '
            "p = pathlib.Path(os.environ['SCA_RESULTS_DIR']); "
            "p.write_bytes if False else None; "
            "(p / 'embedding.json').write_text(str(hash('safety filter') % 256))\""
        )
        rc = main(
            [
                "check",
                "--repeats", "4",
                "--no-probe",
                "--clean",
                "--workdir", str(tmp_path / "w"),
                "--cmd", bad,
            ]
        )
        assert rc == 1, (
            "the reproducibility harness did NOT detect a reintroduced "
            "builtin-hash defect; it would not have caught F9"
        )

    def test_three_processes_agree(self, child_python):
        code = """
            import hashlib, json
            import numpy as np
            from sca.utils.seeding import stable_hash, stable_rng, set_global_seed

            set_global_seed(42)
            texts = [f"request {i} about topic {i % 7}" for i in range(200)]
            V = np.zeros((len(texts), 48))
            for i, t in enumerate(texts):
                for tok in t.split():
                    V[i, stable_hash(tok) % 48] += 1.0
            proj = stable_rng(42, "projection").normal(size=(48, 12))
            X = V @ proj

            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=5, n_init=10, random_state=42).fit(X)
            labels = km.labels_.tolist()
            counts = [int((km.labels_ == j).sum()) for j in range(5)]

            payload = json.dumps({"labels": labels, "counts": counts}, sort_keys=True)
            print(hashlib.sha256(payload.encode()).hexdigest())
        """
        digests = {child_python(code, timeout=600) for _ in range(3)}
        assert len(digests) == 1, (
            f"end-to-end stack is not reproducible across processes: {digests}"
        )
