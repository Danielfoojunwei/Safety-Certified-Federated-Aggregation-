"""Cross-process reproducibility check -- the job that would have caught F9.

Audit finding F9: ``embedding.py`` bucketed tokens with the builtin ``hash()``,
which CPython salts per process, and ``PYTHONHASHSEED`` was never set.  Every
"seeded" result was a per-process random variable; a re-run missed 14 of 14
checked verifier values.  No test caught it, because the only test that *can*
catch it must compare **separate processes**, and every test in the old repo
ran in one.

Usage
-----
Run the built-in probe once and write its artifacts to a directory::

    python -m sca.utils.reprocheck run --out /tmp/r1

Compare two or more artifact directories byte-for-byte::

    python -m sca.utils.reprocheck compare /tmp/r1 /tmp/r2 /tmp/r3

Do the whole thing -- N fresh interpreters, ``PYTHONHASHSEED`` unset in each,
then compare (this is what ``make repro-check`` and the ``reproducibility`` CI
job invoke)::

    python -m sca.utils.reprocheck check --repeats 3

Wrap a real experiment driver instead of (in addition to) the built-in probe::

    python -m sca.utils.reprocheck check --repeats 3 \
        --cmd "python -m sca.experiments.run_all --smoke"

The wrapped command is run with ``SCA_RESULTS_DIR`` pointed at a per-repeat
scratch directory, so whatever it writes into ``results/`` is captured and
diffed automatically.

Design notes
------------
*Hash randomisation is left ON in the children.*  Setting ``PYTHONHASHSEED``
would make the check pass vacuously while the defect is live -- that is the
entire lesson of F9.  ``--vary-hashseed`` additionally forces *different*
explicit seeds, which is a strictly stronger probe.

*The probe cannot pass vacuously.*  It exercises the rebuilt package modules
and records which ones participated.  If fewer than
:data:`MIN_COMPONENTS` participate, it exits non-zero rather than reporting a
green check over an empty computation.

*Floats are compared bit-exactly*, via the raw IEEE-754 bytes, not via a
rounded string.  Rounding to 6 decimals is how a reproducibility check quietly
stops detecting anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from sca.utils.paths import repo_root
from sca.utils.seeding import derive_seed, set_global_seed, stable_hash, stable_rng

#: The probe must exercise at least this many real package components.  A probe
#: that silently degrades to "numpy still works" is not a reproducibility check.
MIN_COMPONENTS = 3

PROBE_SEED = 20260729


# ==========================================================================
# Digest helpers
# ==========================================================================
def _float_digest(a: Any) -> str:
    """Bit-exact digest of an array of floats.

    Uses the raw little-endian float64 bytes, so a difference in the last ulp
    is visible.  Comparing ``round(x, 6)`` would hide exactly the kind of
    drift a reproducibility check exists to find.
    """
    arr = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _json_digest(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digests(root: Path) -> dict[str, str]:
    """Map every file under ``root`` to its SHA-256, keyed by relative path."""
    root = Path(root)
    if not root.is_dir():
        return {}
    return {
        str(p.relative_to(root)): file_digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


# ==========================================================================
# The probe
# ==========================================================================
def run_probe() -> dict[str, Any]:
    """Exercise the rebuilt package and return a dict of digests.

    Every component is imported defensively so the probe still runs while the
    rebuild is in flight, but the participating set is recorded and the caller
    enforces :data:`MIN_COMPONENTS`.  A component that raises is recorded as an
    error, which is itself a reproducibility-relevant fact (an import that
    succeeds in one process and fails in another is a bug too).
    """
    set_global_seed(PROBE_SEED)
    out: dict[str, Any] = {"components": [], "digests": {}, "errors": {}}

    def record(name: str, fn) -> None:
        try:
            out["digests"][name] = fn()
            out["components"].append(name)
        except Exception as exc:  # noqa: BLE001 - recorded, then surfaced
            out["errors"][name] = f"{type(exc).__name__}: {exc}"

    corpus = [
        f"how do i {verb} the {noun} for topic {i % 11}"
        for i, (verb, noun) in enumerate(
            [("bypass", "safety filter"), ("evade", "content policy"),
             ("explain", "chemistry"), ("summarise", "article"),
             ("translate", "passage")] * 60
        )
    ]

    # --- 1. seeding: the F9 fix itself -----------------------------------
    record(
        "seeding.stable_hash",
        lambda: _json_digest([stable_hash(t) for t in corpus[:50]]),
    )
    record(
        "seeding.stable_rng",
        lambda: _float_digest(stable_rng(PROBE_SEED, "probe").normal(size=256)),
    )
    record(
        "seeding.derive_seed",
        lambda: _json_digest([derive_seed(PROBE_SEED, f"tag{i}") for i in range(20)]),
    )

    # --- 2. embeddings ----------------------------------------------------
    def _embedding():
        from sca.knowledge_graph.embedding import HashedBagOfWordsEmbedder

        emb = HashedBagOfWordsEmbedder(embed_dim=32, seed=PROBE_SEED)
        vecs = emb.embed_batch([{"prompt": t} for t in corpus])
        return _float_digest(vecs)

    record("knowledge_graph.embedding", _embedding)

    # --- 3. region partition (embed -> k-means -> weights) ----------------
    def _partition():
        from sca.knowledge_graph.embedding import HashedBagOfWordsEmbedder
        from sca.knowledge_graph.regions import RegionPartition

        emb = HashedBagOfWordsEmbedder(embed_dim=32, seed=PROBE_SEED)
        vecs = np.asarray(emb.embed_batch([{"prompt": t} for t in corpus]))
        part = RegionPartition(k=6, seed=PROBE_SEED).fit(vecs)
        part.stratify(vecs)
        weights = part.empirical_weights()
        assign = [int(part.assign(v)) for v in vecs]
        return _json_digest(
            {
                "weights": {str(k): _float_digest([v]) for k, v in sorted(weights.items())},
                "assign": assign,
            }
        )

    record("knowledge_graph.regions", _partition)

    # --- 4. statistics / acceptance rule ----------------------------------
    def _stats():
        from sca.utils.stats import RegionStat, check_acceptance, optimal_allocation

        stats = [
            RegionStat(j, w, n, v)
            for j, (w, n, v) in enumerate(
                [(0.35, 200, 9), (0.25, 150, 21), (0.20, 90, 0), (0.20, 300, 44)]
            )
        ]
        res = check_acceptance(stats, epsilon=0.2, delta=0.05, k_total=6, budget_cap=500)
        alloc = optimal_allocation([0.35, 0.25, 0.20, 0.20], 1000, k_total=4)
        return _json_digest(
            {
                "bound": _float_digest([res.bound]),
                "accepted": bool(res.accepted),
                "vacuous": bool(res.vacuous),
                "ucb": _float_digest([r.ucb for r in res.per_region]),
                "alloc": list(map(int, alloc)),
            }
        )

    record("utils.stats", _stats)

    # --- 5. Merkle commitment ---------------------------------------------
    def _crypto():
        import sca.utils.crypto as crypto

        for fname in ("merkle_root", "build_merkle_tree", "MerkleTree"):
            obj = getattr(crypto, fname, None)
            if obj is None:
                continue
            leaves = [f"leaf-{i}".encode() for i in range(7)]
            try:
                r = obj(leaves)
            except Exception:
                r = obj([b.decode() for b in leaves])
            root = getattr(r, "root", r)
            return hashlib.sha256(
                root if isinstance(root, bytes) else str(root).encode()
            ).hexdigest()
        raise AttributeError("no recognised Merkle entry point in sca.utils.crypto")

    record("utils.crypto", _crypto)

    # --- 6. torch determinism ---------------------------------------------
    def _torch():
        import torch

        from sca.utils.seeding import seed_torch_generator

        g = seed_torch_generator(PROBE_SEED, "probe.torch")
        x = torch.randn(64, 16, generator=g)
        w = torch.randn(16, 4, generator=g)
        return _float_digest((x @ w).detach().numpy())

    record("torch", _torch)

    out["components"] = sorted(out["components"])
    out["summary_digest"] = _json_digest(out["digests"])
    return out


# ==========================================================================
# Commands
# ==========================================================================
def cmd_run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_probe()
    n = len(result["components"])
    (out_dir / "probe.json").write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(f"components exercised ({n}): {', '.join(result['components'])}")
    if result["errors"]:
        print("components that FAILED to run:", file=sys.stderr)
        for k, v in sorted(result["errors"].items()):
            print(f"  {k}: {v}", file=sys.stderr)
    print(f"summary_digest {result['summary_digest']}")

    if n < MIN_COMPONENTS:
        print(
            f"\nFATAL: only {n} component(s) participated (minimum "
            f"{MIN_COMPONENTS}). A reproducibility check over an almost-empty "
            f"computation is worse than none, because it reports green. "
            f"Fix the import errors above.",
            file=sys.stderr,
        )
        return 2
    return 0


def _compare_dirs(dirs: list[Path]) -> tuple[bool, list[str]]:
    """Return ``(ok, messages)`` for a byte-for-byte comparison of ``dirs``."""
    base, *rest = dirs
    base_tree = tree_digests(base)
    msgs: list[str] = []
    ok = True

    if not base_tree:
        return False, [f"FATAL: {base} contains no files; nothing was compared."]

    for other in rest:
        other_tree = tree_digests(other)
        only_base = sorted(set(base_tree) - set(other_tree))
        only_other = sorted(set(other_tree) - set(base_tree))
        differing = sorted(
            f for f in set(base_tree) & set(other_tree)
            if base_tree[f] != other_tree[f]
        )
        if only_base or only_other or differing:
            ok = False
            msgs.append(f"--- {base.name} vs {other.name} ---")
            for f in only_base:
                msgs.append(f"  only in {base.name}: {f}")
            for f in only_other:
                msgs.append(f"  only in {other.name}: {f}")
            for f in differing:
                msgs.append(
                    f"  DIFFERS: {f}\n"
                    f"      {base.name}: {base_tree[f]}\n"
                    f"      {other.name}: {other_tree[f]}"
                )
    return ok, msgs


def cmd_compare(args: argparse.Namespace) -> int:
    dirs = [Path(d).resolve() for d in args.dirs]
    if len(dirs) < 2:
        print("compare needs at least two directories", file=sys.stderr)
        return 2
    ok, msgs = _compare_dirs(dirs)
    n_files = len(tree_digests(dirs[0]))
    if ok:
        print(
            f"REPRODUCIBLE: {len(dirs)} runs agree byte-for-byte "
            f"over {n_files} file(s)."
        )
        return 0
    print(
        "NOT REPRODUCIBLE: runs disagree. This is the F9 failure mode "
        "(results are a per-process random variable).",
        file=sys.stderr,
    )
    for m in msgs:
        print(m, file=sys.stderr)
    return 1


def cmd_check(args: argparse.Namespace) -> int:
    """Run N fresh interpreters and compare their artifacts."""
    workdir = Path(args.workdir).resolve() if args.workdir else Path(
        tempfile.mkdtemp(prefix="sca-repro-")
    )
    if workdir.exists() and args.clean:
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    root = repo_root()
    run_dirs: list[Path] = []

    for i in range(args.repeats):
        rd = workdir / f"run{i + 1}"
        if rd.exists():
            shutil.rmtree(rd)
        (rd / "results").mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        # THE POINT: hash randomisation stays ON. Setting PYTHONHASHSEED here
        # would make this check pass while the defect is live.
        env.pop("PYTHONHASHSEED", None)
        if args.vary_hashseed:
            env["PYTHONHASHSEED"] = str(1000 + i * 7919)
        env["PYTHONPATH"] = str(root)
        env["SCA_RESULTS_DIR"] = str(rd / "results")
        env["SCA_CACHE_DIR"] = str(workdir / "cache")  # shared, so downloads
        env.setdefault("OMP_NUM_THREADS", "1")         # are not re-diffed
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")

        # Built-in probe.
        if not args.no_probe:
            proc = subprocess.run(
                [sys.executable, "-m", "sca.utils.reprocheck", "run", "--out", str(rd)],
                capture_output=True, text=True, env=env, cwd=str(root),
                timeout=args.timeout,
            )
            sys.stdout.write(f"[run{i + 1}] probe: {proc.stdout}")
            if proc.returncode != 0:
                sys.stderr.write(proc.stderr)
                return proc.returncode

        # Optional wrapped experiment driver.
        if args.cmd:
            proc = subprocess.run(
                args.cmd, shell=True, capture_output=True, text=True,
                env=env, cwd=str(root), timeout=args.timeout,
            )
            (rd / "cmd_stdout.txt").write_text(proc.stdout, encoding="utf-8")
            if proc.returncode != 0:
                sys.stderr.write(
                    f"[run{i + 1}] wrapped command failed (rc={proc.returncode})\n"
                    f"{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}\n"
                )
                return proc.returncode
            print(f"[run{i + 1}] wrapped command ok")

        run_dirs.append(rd)

    args.dirs = [str(d) for d in run_dirs]
    rc = cmd_compare(args)
    if rc == 0:
        mode = (
            "with DIFFERENT explicit PYTHONHASHSEED values"
            if args.vary_hashseed
            else "with PYTHONHASHSEED unset (hash randomisation ACTIVE)"
        )
        print(f"  ({args.repeats} separate processes, {mode})")
    print(f"artifacts: {workdir}")
    return rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m sca.utils.reprocheck",
        description="Cross-process reproducibility check (audit finding F9).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run the probe once and write artifacts")
    r.add_argument("--out", required=True)
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("compare", help="byte-compare two or more artifact dirs")
    c.add_argument("dirs", nargs="+")
    c.set_defaults(func=cmd_compare)

    k = sub.add_parser("check", help="run N fresh interpreters, then compare")
    k.add_argument("--repeats", type=int, default=3)
    k.add_argument("--workdir", default=None)
    k.add_argument("--cmd", default=None,
                   help="extra command to run per repeat, with SCA_RESULTS_DIR set")
    k.add_argument("--no-probe", action="store_true",
                   help="skip the built-in probe (only meaningful with --cmd)")
    k.add_argument("--vary-hashseed", action="store_true",
                   help="force DIFFERENT explicit PYTHONHASHSEED values per run")
    k.add_argument("--clean", action="store_true")
    k.add_argument("--timeout", type=int, default=3600)
    k.set_defaults(func=cmd_check)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
