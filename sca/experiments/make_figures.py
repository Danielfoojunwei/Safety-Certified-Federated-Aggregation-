"""``python -m sca.experiments.make_figures`` -- render results/ into tables.

Criterion **C6**: every number that appears in the README must be regenerated
by a committed script writing into ``results/``.  This module reads the JSON
that :mod:`sca.experiments.run_all` wrote and emits Markdown tables plus a
``results/README_NUMBERS.md`` fragment.  Nothing here computes a statistic; it
only formats what the runner already recorded, so a table can never disagree
with the artifact it came from.

``matplotlib`` is not installed in this environment and is deliberately not a
dependency: the deliverables are tables of numbers with confidence intervals,
and a PNG would be one more artifact that can drift from the JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Sequence

from sca.utils.paths import results_dir

__all__ = ["render_verifier_table", "render_fl_table", "main"]


def _load(name: str) -> dict | None:
    path = results_dir() / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _ci(d: dict) -> str:
    return f"{d['mean']:.4f} [{d['lo']:.4f}, {d['hi']:.4f}]"


def render_verifier_table(payload: dict) -> str:
    s = payload["summary"]
    h = payload["hypotheses"]
    rows = []
    for arm in sorted(s["bound"], key=lambda a: s["bound"][a]["mean"]):
        rows.append([
            arm,
            _ci(s["bound"][arm]),
            f"{s['weighted_p_hat'][arm]['mean']:.4f}",
            f"{s['total_violations_raw'][arm]['mean']:.1f}",
            f"{s['distinct_violations_dedup_depth0'][arm]['mean']:.1f}",
            f"{s['raw_amplification'][arm]['mean']:.2f}x",
            f"{s['dedup_amplification'][arm]['mean']:.2f}x",
            s["bound"][arm]["n_seeds"],
        ])
    body = _table(
        ["arm", "certified bound (95% CI)", "weighted p_hat",
         "violations RAW", "violations DEDUP", "ampl. RAW", "ampl. DEDUP",
         "seeds"],
        rows,
    )
    lines = ["## Verifier arm", "", body, "", "### H2 -- does search-guided "
             "allocation tighten the certified bound?", ""]
    for key in ("H2_search_guided_vs_w23", "H2_search_guided_vs_proportional",
                "H2_search_guided_vs_uniform"):
        v = h.get(key, {})
        if not v.get("available"):
            continue
        verdict = "SUPPORTED" if v["a_is_tighter"] else "**NOT SUPPORTED**"
        lines.append(
            f"- `{v['arm_a']}` vs `{v['arm_b']}`: "
            f"{v['mean_difference_a_minus_b']:+.4f} (lower is tighter) -> "
            f"{verdict}, p = {v['p_value']:.4f} "
            f"(minimum attainable p = {v['min_attainable_p']:.4f})"
        )
    amp = h["F8_amplification"]
    lines += ["", "### F8 -- recursive amplification, raw vs deduplicated", "",
              f"- null mutator's RAW score >= real operators': "
              f"{amp['null_mutator_raw_beats_real']}",
              f"- identity mutator's RAW score >= real operators': "
              f"{amp['identity_mutator_raw_beats_real']}",
              "", "Raw amplification re-counts one failure many times. The "
              "deduplicated column counts distinct depth-0 ancestors and is "
              "the only one that may be quoted.", ""]
    return "\n".join(lines)


def render_fl_table(payload: dict) -> str:
    s = payload["summary"]
    h = payload["hypotheses"]
    rows = [[arm, _ci(s[arm]), s[arm]["n_seeds"]] for arm in sorted(s)]
    lines = ["## Federated arm -- held-out test accuracy", "",
             "The gate never saw this split. `always_reject_gate` must equal "
             "`frozen_pretrained` and `always_accept_gate` must equal "
             "`no_gate`; both are asserted at the level of parameter bytes "
             "during the run.", "",
             _table(["arm", "held-out accuracy (95% CI)", "seeds"], rows), ""]

    c1 = h["C1_clean_fedavg_beats_frozen"]
    lines += ["### C1 -- clean FedAvg must beat the frozen checkpoint", "",
              f"- passed on {c1['n_seeds_passing']}/{c1['n_seeds']} seeds", ""]

    h3 = h["H3_gate_improves_quality_under_attack"]
    nd = h3.get("n_where_gate_is_the_do_nothing_baseline", 0)
    lines += ["### H3 -- does the certified gate improve held-out quality "
              "under attack?", "",
              f"- the gate helped in {h3['n_where_gate_helps']}/"
              f"{h3['n_configurations']} configurations",
              f"- the gate was the **do-nothing baseline** (gated model "
              f"byte-identical to the frozen checkpoint on every seed) in "
              f"{nd}/{h3['n_configurations']} configurations", ""]
    if h3.get("caveat"):
        lines += ["> " + h3["caveat"], ""]
    rows = []
    for name, v in sorted(h3["per_configuration"].items()):
        rows.append([
            name.replace("|certified_gate", ""),
            f"{v['gate_mean']:.4f}", f"{v['no_gate_mean']:.4f}",
            f"{v['difference']:+.4f}",
            "yes" if v["gate_helps"] else "no",
            f"{v.get('n_seeds_gated_model_is_the_frozen_checkpoint', 0)}"
            f"/{v['n_seeds']}",
            f"{v['p_value']:.4f}", v["n_seeds"],
        ])
    lines += [_table(["aggregator | attack", "gate", "no gate", "difference",
                      "gate helps", "seeds where gated == frozen", "p",
                      "seeds"], rows), ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sca.experiments.make_figures")
    ap.add_argument("--smoke", action="store_true",
                    help="render the smoke_*.json artifacts instead")
    args = ap.parse_args(argv)
    prefix = "smoke_" if args.smoke else ""

    parts: list[str] = [
        f"<!-- generated by `python -m sca.experiments.make_figures"
        f"{' --smoke' if args.smoke else ''}`; do not edit by hand -->",
        "",
        "# Regenerated results",
        "",
    ]
    found = 0
    ver = _load(f"{prefix}verifier_arm.json")
    if ver is not None:
        parts.append(render_verifier_table(ver))
        found += 1
    fl = _load(f"{prefix}fl_arm.json")
    if fl is not None:
        parts.append(render_fl_table(fl))
        found += 1
    manifest = _load(f"{prefix}run_manifest.json")
    if manifest is not None:
        g = manifest.get("git", {})
        v = manifest.get("versions", {})
        parts += ["## Provenance", "",
                  f"- generated at {manifest.get('generated_at_utc')}",
                  f"- git commit `{g.get('commit')}` on `{g.get('branch')}` "
                  f"(dirty={g.get('dirty')})",
                  f"- python {v.get('python')}, numpy {v.get('numpy')}, "
                  f"torch {v.get('torch')}, scikit-learn {v.get('sklearn')}",
                  ""]
        found += 1

    if found == 0:
        print(
            f"ERROR: no {prefix}*.json artifacts in {results_dir()}.\n"
            f"Run `python -m sca.experiments.run_all"
            f"{' --smoke' if args.smoke else ''}` first.",
            file=sys.stderr,
        )
        return 1

    text = "\n".join(parts)
    out = results_dir() / f"{prefix}README_NUMBERS.md"
    out.write_text(text)
    print(text)
    print(f"\nwrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
