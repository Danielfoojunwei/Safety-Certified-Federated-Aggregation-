"""Experiment framework: data splits, attacks, baselines, metrics, drivers.

:mod:`sca.experiments.data` owns the three-way split -- disjoint client pools,
a server verification pool the gate may see, and a held-out test set the gate
never sees.  ``SplitBundle.assert_disjoint()`` is the guard against audit
finding F1, where every reported accuracy was a training accuracy.

Benchmark suites must load real data or say plainly that they are synthetic.
The pre-rebuild suites called ``_generate_synthetic()`` while their docstrings
cited real venues and real dataset sizes (audit finding F12).

Experiment outputs go to :func:`sca.utils.paths.results_dir`, never to a
hardcoded absolute path.
"""

__all__: list[str] = []
