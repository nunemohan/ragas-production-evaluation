# Stage 1 – Insurance Golden Dataset

## Scope
This stage creates a synthetic insurance knowledge base and a 15-record golden evaluation dataset for the RAGAS Production Evaluation Lab.

**Important:** Nunez Secure Insurance and SecureCare Plus are fictional. These files are training data only and are not real insurance advice or policy wording.

## Source documents
- `01_policy_overview.md`
- `02_benefits_and_limits.md`
- `03_waiting_periods_and_exclusions.md`
- `04_claims_procedure.md`
- `05_premium_and_servicing.md`

## Golden dataset
`data/golden_dataset/ragas_golden_dataset.json`

Each record includes:
- `id`
- `category`
- `user_input`
- `reference`
- `reference_contexts`
- `source_document_ids`
- `expected_behavior`

The top-level JSON also includes dataset metadata and category counts.

## RAGAS mapping
For a later RAGAS evaluation run, map:
- `user_input` → `user_input`
- actual LangChain/FAISS retrieved chunks → `retrieved_contexts`
- actual generated answer → `response`
- `reference` → `reference`
- `reference_contexts` → `reference_contexts` (for reference-context metrics)

Do not use the golden `reference_contexts` as the actual `retrieved_contexts`. Actual retrieved chunks must come from the RAG application, otherwise retrieval evaluation would be circular.

For unanswerable questions, `reference_contexts` is empty because the supplied knowledge base intentionally does not contain the requested policy information. We will handle these records explicitly during evaluation and inspect whether the RAG application abstains rather than fabricates.

## Category distribution
- Simple factual: 3
- Numerical: 2
- Multi-condition: 2
- Conceptual: 2
- Multi-document: 2
- Ambiguous: 2
- Unanswerable: 2
- Total: 15

## Validation
Run from the repository root:
```powershell
python -m pytest -q
```
