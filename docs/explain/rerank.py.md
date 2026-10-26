# rerank.py — explained

## What it does
Takes the merged list from RRF and re-orders it so the final top-k chunks
collectively cover as many sub-queries as possible (multi-hop diversity),
while still preferring high-relevance chunks and demoting near-duplicates.

## The problem it solves
RRF ranks by aggregate relevance across retrievers. In a multi-hop question
("Who invented the telephone, and where were they born?"), the top results
might all answer hop 1 ("invented telephone") and none answer hop 2 ("born
where"). A greedy coverage reranker fixes this by rewarding chunks that cover
information not yet represented in the selection.

## The algorithm (greedy MMR-style)

At each step, pick the candidate that maximises:

```
score(c) = λ · relevance(c)
         + (1 − λ) · coverage_gain(c)
         − effective_dup_penalty · is_near_dup(c)
```

| Symbol | Meaning |
|---|---|
| λ (lambda_) | Trade-off weight: 1 = pure relevance, 0 = pure coverage |
| relevance(c) | `rrf_score` — already on a comparable scale from RRF |
| coverage_gain(c) | `new_queries_covered / total_sub_queries` |
| effective_dup_penalty | `(1 − λ) × dup_penalty` — scales to zero when λ = 1 |
| is_near_dup(c) | 1 if c.dup_group is already represented in the selected set |

**Tie-breaking:** when two chunks have the same `score`, the one with higher
`coverage_gain` wins. This ensures a chunk that covers a new sub-query is
always preferred over one that does not, even at equal relevance.

**Coverage threshold:** a chunk "covers" a sub-query when the overlap between
their token sets reaches ≥ `coverage_threshold` (default 0.4) of the query's
tokens. This tolerates paraphrase ("inventor" vs "invented").

**dup_penalty scaling:** the penalty scales by `(1 − λ)`. At λ = 1.0 the
penalty is zero (pure-relevance mode has no diversity goal). When there are no
sub-queries at all (n_queries = 0), the penalty is also zero.

## Worked example

Chunks: A(rrf=0.9, dup=X), B(rrf=0.8), C(rrf=0.3), D(rrf=0.7, dup=X)
Sub-queries: sq1 = "bell telephone inventor", sq2 = "bell born year"
λ = 0.5

Round 1 (nothing selected):
```
A: 0.5×0.9 + 0.5×0.5 = 0.70  ← covers sq1
B: 0.5×0.8 + 0.5×0   = 0.40
C: 0.5×0.3 + 0.5×0.5 = 0.40  ← covers sq2
D: 0.5×0.7 + 0.5×0.5 = 0.60  ← covers sq1
```
→ A selected (highest score). sq1 marked covered. dup_group X recorded.

Round 2 (sq1 covered, dup X seen):
```
B: 0.5×0.8 + 0.5×0   = 0.40  (sq2 not covered by B)
C: 0.5×0.3 + 0.5×0.5 = 0.40  ← covers new sq2; tie-break wins
D: 0.5×0.7 + 0.5×0 − 0.25 = 0.10  ← dup penalty
```
→ C selected (tie, but gain=0.5 > gain=0). sq2 now covered.

Round 3: B(0.40) > D(0.10) → B selected.
Round 4: D selected.

Final order: **A, C, B, D** — C (rrf=0.3) jumped from rank 4 to rank 2
because it covered the second hop.

## Complexity

| Step | Cost |
|---|---|
| Token sets (once per chunk) | O(|text|) each, O(N · |text|) total |
| Greedy selection | O(N² · Q) where N = candidates, Q = sub-queries |
| Typical values | N ≤ 40, Q ≤ 5 → negligible |

Quadratic in N but N is small (20–40 after RRF) so it is fast in practice.

## Edge cases

- **No sub-queries:** gain = 0, penalty = 0 → degrades to descending rrf_score sort.
- **lambda_ = 1.0:** coverage term and dup_penalty both vanish → pure rrf_score sort.
- **lambda_ = 0.0:** only coverage matters; near-dup penalty is maximum (1.0 × dup_penalty).
- **Empty candidates:** returns [].
- **top_k > len(candidates):** returns all candidates.
- **Chunk not covering any sub-query:** gain = 0; still included by relevance.

## 5 questions an interviewer would ask

**Q1: Why greedy instead of optimal (exhaustive) selection?**
Optimal selection is NP-hard (maximum coverage problem). Greedy gives a
(1 − 1/e) ≈ 63% approximation guarantee for submodular coverage functions,
and in practice is nearly indistinguishable from optimal for small candidate
sets (20–40 chunks).

**Q2: What does lambda_ control?**
It is the trade-off between relevance and diversity. λ = 1 is pure RRF order.
λ = 0 ignores relevance entirely and only maximises sub-query coverage. The
default λ = 0.5 balances both. This is the same parameter as in MMR
(Maximal Marginal Relevance, Carbonell & Goldstein 1998).

**Q3: Why scale dup_penalty by (1 − λ)?**
The dup_penalty exists to protect coverage diversity — it prevents picking
near-duplicate chunk B when chunk C could cover a new sub-query instead.
When λ = 1.0 the coverage goal is turned off, so the penalty is meaningless
and should not distort the pure-relevance ranking.

**Q4: How is "covers sub-query" defined? Why use token overlap?**
A chunk covers a sub-query if the intersection of their token sets is at least
`coverage_threshold` (default 0.4) of the query's tokens. Token overlap
tolerates minor paraphrase ("inventor" vs "invented"). A semantic threshold
(embeddings cosine similarity) would be more precise but requires an embedding
call per (chunk, sub-query) pair — expensive. Token overlap is O(1) after
tokenising and keeps the module library-free.

**Q5: What happens if two sub-queries are semantically the same?**
Both will be "covered" by the same first chunk, since coverage is tracked
per-index not per-content. This is acceptable — if the user decomposed two
synonymous sub-queries, any chunk covering one will cover both, and no slot is
wasted on redundant selection.
