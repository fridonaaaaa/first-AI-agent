# fusion.py — explained

## What it does
Takes two or more ranked lists (e.g. dense search results + BM25 results) and
merges them into a single ranked list. A document that appears near the top of
multiple lists gets a higher combined score than one that only appears in one.

## The formula

```
rrf_score(d) = Σ  1 / (k + rank_i(d))
                i
```

| Symbol | Meaning |
|---|---|
| rank_i(d) | Position of document d in list i — 1-indexed; if absent, no contribution |
| k | Smoothing constant (default 60) |
| Σ | Sum over all input ranked lists |

## Worked example (k=60)

Two input lists:
```
Dense: [A(rank 1), B(rank 2), C(rank 3)]
BM25:  [B(rank 1), C(rank 2), D(rank 3)]
```

RRF scores:
```
A = 1/(60+1)                      = 0.01639  ← dense only, rank 1
B = 1/(60+2) + 1/(60+1)           = 0.03252  ← in both lists
C = 1/(60+3) + 1/(60+2)           = 0.03200  ← in both lists
D = 1/(60+3)                      = 0.01587  ← BM25 only, rank 3
```

Final order: **B > C > A > D**

Notice: B was rank 2 in dense but rank 1 in BM25 — appearing in both lists
boosted it above A which was rank 1 in dense but absent from BM25.

## Why original scores are ignored

Dense and BM25 scores live on completely different scales (cosine similarity
0–1 vs BM25 can be 0–10+). Adding them directly would let one system dominate.
RRF normalises everything to rank positions — a fair comparison regardless of scale.

## What k controls

k prevents rank-1 from having infinite advantage over rank-2.
- Small k (e.g. 1): rank 1 = 1/2 = 0.50, rank 2 = 1/3 = 0.33 — huge gap
- Large k (e.g. 60): rank 1 = 1/61 ≈ 0.0164, rank 2 = 1/62 ≈ 0.0161 — tiny gap
- k=60 is the value from the original RRF paper (Cormack et al. 2009) and works well

## Complexity

O(Σ |list_i|) to accumulate scores + O(R log R) to sort, where R = total unique docs.
Typically a few hundred documents per list → essentially instant.
