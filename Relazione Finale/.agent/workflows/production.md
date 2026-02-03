---
description: Validate, Lint, and Compile the Thesis
---

# Production Workflow

1.  **Quality Assurance**:
    *   **Validate**: Check for structure errors (missing chapters).
        `python scripts/manage.py validate --fix --json`
    *   **Lint**: Check for broken links, TODOs, and missing citations.
        `python scripts/manage.py lint --json`

2.  **Styling**:
    *   List Templates: `python scripts/manage.py style list --json`
    *   Set Template: `python scripts/manage.py style set "modern_report" --json`

3.  **Compilation**:
    *   **PDF**: `python scripts/manage.py compile --json`
    *   **Inspect**: `python scripts/manage.py inspect_pdf "dist/thesis.pdf" --json`

4.  **Export/Feedback**:
    *   **DOCX**: For advisor feedback.
        `python scripts/manage.py export_docx --json`
    *   **Backup**: Full project ZIP.
        `python scripts/manage.py export "backup.zip" --json`
