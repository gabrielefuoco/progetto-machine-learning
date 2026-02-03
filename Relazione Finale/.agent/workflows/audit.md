---
description: Perform academic integrity and citation audits
---

# Academic Audit Workflow

1.  **Synchronize Everything**:
    `python scripts/manage.py rag sync_all --json`

2.  **Verify Bibliography Mapping**:
    `python scripts/manage.py rag verify_mapping --json`
    *Tip*: use `--fix` if unmapped files exist.

3.  **Semantic Redundancy Check**:
    `python scripts/manage.py rag check_redundancy --adaptive --json`

4.  **Citation Audit**:
    `python scripts/manage.py rag cite_check --threshold 0.6 --json`

5.  **Claim Validation**:
    `python scripts/manage.py rag validate_claims "Claim text here..." --json`

6.  **Bibliography Maintenance**:
    `python scripts/manage.py clean_bib --json`
