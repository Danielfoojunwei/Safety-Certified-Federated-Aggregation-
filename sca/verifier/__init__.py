"""RLM verifier: recursive search, then i.i.d. estimation, then a certificate.

The three stages are deliberately separated, because using the same adaptively
generated samples both to *search* for failures and to *estimate* the violation
rate is what made the original Theorem 1 false (audit finding F3):

Stage A (search)
    Mutation recursion and MKG frontier exploration over the search pool.
    Recorded to ``search_*`` counters.  Its only output is a per-region
    suspicion score used to choose the estimation allocation.  It never touches
    the bound.
Stage B (estimation)
    For each region, draw ``m_j`` fresh i.i.d. samples from ``D | R_j`` out of a
    *disjoint* estimation pool.  Recorded to ``estimation_*`` counters.
Stage C (certificate)
    Compute the bound from ``estimation_stats`` alone.

Violation counts are reported both raw and deduplicated to distinct depth-0
ancestors (audit finding F8).

* :mod:`sca.verifier.rlm_verifier` -- the three-stage driver.
* :mod:`sca.verifier.mutations` -- mutation operators, including the mandatory
  NULL (whitespace) and IDENTITY controls.
* :mod:`sca.verifier.safety_predicate` -- the safety oracle.  It must not read
  the ground-truth label of the set it is scored on (audit finding F2).
"""

__all__: list[str] = []
