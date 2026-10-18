# InsightForge Enterprise - RAG Quality Report

- **Generated at:** 2026-10-17T10:48:37.159463+00:00
- **Evaluation questions:** 8
- **Retrieval cutoff:** top 3

## Summary metrics

| Metric | Result |
|---|---:|
| Hit rate@3 | 100.00% |
| Mean Reciprocal Rank (MRR) | 1.0000 |

## Per-question results

| ID | Category | Expected source | Rank | Retrieved? |
|---|---|---|---:|---|
| eval-001 | business_metrics | `campaign_metrics.md` | 1 | Yes |
| eval-002 | responsible_analytics | `customer_analytics.md` | 1 | Yes |
| eval-003 | ml_evaluation | `sentiment_evaluation.md` | 1 | Yes |
| eval-004 | responsible_ai | `responsible_ai.md` | 1 | Yes |
| eval-005 | business_metrics | `campaign_metrics.md` | 1 | Yes |
| eval-006 | responsible_ai | `responsible_ai.md` | 1 | Yes |
| eval-007 | unanswerable | `None` | Not found | No |
| eval-008 | unanswerable | `None` | Not found | No |

## Interpretation

- Hit rate measures whether the expected source appears within the selected top-k results.
- MRR rewards expected sources that appear nearer the top.
- These metrics evaluate retrieval, not generated-answer factuality or business impact.

## Limitations

- Small, hand-authored evaluation dataset.
- Retrieval of a source does not prove answer correctness.
- Unanswerable cases require answer-level abstention tests.
- Synthetic data and local documents do not establish production performance.

## Reproduction

```powershell
python -m insightforge.rag.build_index
python -m insightforge.evaluation.rag_evaluator
python scripts/generate_quality_summary.py
```
