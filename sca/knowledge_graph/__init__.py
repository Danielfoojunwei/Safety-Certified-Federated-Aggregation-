"""Model Knowledge Graph (MKG): coverage structure over interaction regions.

* :mod:`sca.knowledge_graph.embedding` -- interaction embedders.  The hashed
  bag-of-words embedder uses :func:`sca.utils.seeding.stable_hash`, not the
  builtin ``hash()`` (audit finding F9).
* :mod:`sca.knowledge_graph.regions` -- the region partition and the
  :class:`~sca.knowledge_graph.regions.Region` type, whose *estimation* and
  *search* counters are kept strictly separate (audit finding F3).
* :mod:`sca.knowledge_graph.mkg` -- the graph itself.  ``tau`` must be
  calibrated rather than hardcoded to 1.0, or the graph degenerates to the
  complete graph and "MKG-guided" becomes indistinguishable from blind search
  (audit finding F4).

Nothing is re-exported eagerly; these modules import scikit-learn and networkx.
"""

__all__: list[str] = []
