"""
Spike A — Embedding model evaluation for Georgian/English support.

Run AFTER installing sentence-transformers:
    pip install sentence-transformers

What this measures (on 30 Georgian-English pairs):
- Whether the model's retrieved paragraph matches the gold title (hit-rate)
- Embedding speed on CPU (sentences/sec)
- Peak RAM usage
- Vector dimension

Results are printed to stdout.  Copy the output into docs/DECISIONS.md
and STOP to let the owner choose the model.

Candidate models to test (verify language support on the model card first):
  A. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2  (dim=384)
  B. sentence-transformers/paraphrase-multilingual-mpnet-base-v2   (dim=768)
  C. intfloat/multilingual-e5-small                                (dim=384)

NOTE: Georgian (kat / ka) coverage must be verified on each model card.
      If a model card does not list Georgian, mark it UNVERIFIED.
"""

# TODO: implement after the owner has Python + sentence-transformers installed
# and the HotpotQA subset manifest is ready (run build_subset.py first).
raise NotImplementedError("Spike A not yet implemented — see docstring for instructions.")
