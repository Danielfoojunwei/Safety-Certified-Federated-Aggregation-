"""Federated learning components: clients, server, and aggregation rules.

The aggregation gate must actually be able to change the model.  In the
pre-rebuild code the server aggregated into a ``deepcopy`` and "rolled back" by
restoring the untouched original, so rejection was a strict no-op and the
headline Byzantine number was just the frozen pretrained checkpoint (audit
finding F5).  Two sanity checks now pin this and belong in every results table:

* ``always-reject`` must numerically equal ``frozen pretrained``;
* ``always-accept`` must numerically equal ``no gate``.

Client data is split non-IID by Dirichlet, not by sorting on the label and
chunking -- contiguous label chunks make every client single-class, which
collapses Krum to a constant predictor even with zero adversaries
(audit finding F7).
"""

__all__: list[str] = []
