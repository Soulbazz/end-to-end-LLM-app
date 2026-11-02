# Retrieval Evaluation Results

Corpus: 20 docs | Queries: 20 (10 semantic, 10 exact-code)

| Strategy | Recall@3 | Recall@5 | MRR |
|---|---|---|---|
| Baseline 1: Vector-only | 0.875 | 0.875 | 0.842 |
| Baseline 2: BM25-only | 0.900 | 0.950 | 0.835 |
| Upgraded: Hybrid (BM25 + Vector, RRF k=60) | 0.950 | 0.950 | 0.950 |

## Queries where Hybrid RRF beat a baseline

**Q3:** "ERR-PSU-904"  (relevant: ['doc_04', 'doc_16'])
  - Baseline 1: Vector-only: top-5 = [], RR = 0.00
  - Baseline 2: BM25-only: top-5 = ['doc_04', 'doc_16'], RR = 1.00
  - Upgraded: Hybrid (BM25 + Vector, RRF k=60): top-5 = ['doc_04', 'doc_16'], RR = 1.00

**Q15:** "By what hour must I pay to get my order shipped tonight?"  (relevant: ['doc_09'])
  - Baseline 1: Vector-only: top-5 = ['doc_09', 'doc_02', 'doc_12', 'doc_18', 'doc_10'], RR = 1.00
  - Baseline 2: BM25-only: top-5 = ['doc_06', 'doc_15', 'doc_08', 'doc_16', 'doc_09'], RR = 0.20
  - Upgraded: Hybrid (BM25 + Vector, RRF k=60): top-5 = ['doc_09', 'doc_06', 'doc_02', 'doc_10', 'doc_12'], RR = 1.00

**Q16:** "How slow is getting stuff through foreign border checks?"  (relevant: ['doc_11'])
  - Baseline 1: Vector-only: top-5 = ['doc_20', 'doc_09', 'doc_11', 'doc_18', 'doc_13'], RR = 0.33
  - Baseline 2: BM25-only: top-5 = ['doc_11', 'doc_06', 'doc_18', 'doc_20', 'doc_15'], RR = 1.00
  - Upgraded: Hybrid (BM25 + Vector, RRF k=60): top-5 = ['doc_11', 'doc_20', 'doc_18', 'doc_06', 'doc_09'], RR = 1.00

**Q2:** "BIOS-FL-202 and my board will not start anymore, how do I revive it?"  (relevant: ['doc_13'])
  - Baseline 1: Vector-only: top-5 = ['doc_03', 'doc_13', 'doc_05', 'doc_07', 'doc_19'], RR = 0.50
  - Baseline 2: BM25-only: top-5 = ['doc_13', 'doc_15', 'doc_14', 'doc_05', 'doc_03'], RR = 1.00
  - Upgraded: Hybrid (BM25 + Vector, RRF k=60): top-5 = ['doc_13', 'doc_03', 'doc_05', 'doc_06', 'doc_19'], RR = 1.00

## Failure / hard case (error analysis)

**Q12:** "My package showed up smashed up and the seal was cut, what now?"  (relevant: ['doc_12'])
  - Baseline 1: Vector-only: top-5 = ['doc_05', 'doc_13', 'doc_15', 'doc_11', 'doc_06'], RR = 0.00
  - Baseline 2: BM25-only: top-5 = ['doc_07', 'doc_08', 'doc_03', 'doc_05', 'doc_09'], RR = 0.00
  - Upgraded: Hybrid (BM25 + Vector, RRF k=60): top-5 = ['doc_05', 'doc_15', 'doc_08', 'doc_11', 'doc_13'], RR = 0.00

Analysis: the relevant document is only partly matched by lexical overlap and the n-gram vector model, so neither ranker places it high and RRF cannot recover it. Real dense embeddings (`USE_API_EMBEDDINGS=1`) or a reranker would likely help.
