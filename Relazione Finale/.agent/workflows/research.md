---
description: Manage Citations, RAG, and Zotero
---

# Research Workflow

## A. Scientific Deep Research (Autonomous)
1.  **Discovery Mode**: Find papers/metadata.
    `python scripts/manage.py research search "Query" --mode discovery --json`
2.  **Deep Mode**: Download PDFs, extract citations, and auto-sync RAG.
    `python scripts/manage.py research search "Query" --mode deep --limit 3 --json`

## B. Bibliography Management
1.  **Search**: `python scripts/manage.py search_bib "Query" --json`
2.  **Add**: `python scripts/manage.py add_citation "BibTeX..." --json`

## C. Zotero Integration
1.  **Setup**: Check credentials first.
    `python scripts/manage.py zotero setup --json`
2.  **Link Collection**:
    `python scripts/manage.py zotero list_collections --json`
    `python scripts/manage.py zotero set_collection "ID" --json`
3.  **Sync**: Download PDFs and Metadata.
    `python scripts/manage.py zotero sync --json`

## D. Semantic Search (RAG)
1.  **Sync Indices**:
    *   Draft (Your writing): `python scripts/manage.py rag sync_draft --json`
    *   Research (PDFs/Docs): `python scripts/manage.py rag sync_research --path "./papers" --json`
    *   **Unified Sync**: `python scripts/manage.py rag sync_all --json`
2.  **Query**:
    *   `python scripts/manage.py rag query draft "What did I write about X?" --json`
    *   `python scripts/manage.py rag query research "What does Author Y say about Z?" --json`
3.  **Auditing**:
    *   Check for missing citations: `python scripts/manage.py rag cite_check --json`
    *   Validate a claim: `python scripts/manage.py rag validate_claims "Claim" --json`
