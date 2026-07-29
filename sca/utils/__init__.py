"""Cross-cutting utilities.

* :mod:`sca.utils.seeding` -- process-independent hashing and RNG derivation.
  Import ``stable_hash`` from here rather than calling the builtin ``hash()``
  on a string anywhere in the package (audit finding F9).
* :mod:`sca.utils.paths` -- repository-relative path resolution.  Replaces the
  hardcoded ``/home/user/...`` output directories in the legacy entry points.
* :mod:`sca.utils.stats` -- anytime-valid concentration bounds, the acceptance
  rule, and budget allocation.
* :mod:`sca.utils.crypto` -- Merkle commitments over verification transcripts.

Nothing is re-exported eagerly: ``sca.utils.stats`` imports scipy, and a bare
``import sca.utils`` should not pay for it.
"""

__all__: list[str] = []
