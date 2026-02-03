---
description: Create new content (Chapters, Paragraphs, Assets)
---

# Drafting Workflow

1.  **Analyze Context**:
    Before creating content, always check the `tree` or `context` to ensure logical flow.

2.  **Create Structure**:
    *   **New Chapter**: `python scripts/manage.py add_chapter "Title" --json`
    *   **New Paragraph (at end)**: `python scripts/manage.py add_paragraph "Title" --chapter "Target" --json`
    *   **Insert Paragraph (at position)**: `python scripts/manage.py insert "Title" --chapter 1 --after "Previous" --json`
        *   *Tip*: Use `--raw` if the paragraph shouldn't have a header (e.g., intro text).

3.  **Reorganize Content**:
    *   **Move**: `python scripts/manage.py move "Para" --from 1 --to 2 --json`
    *   **Merge**: `python scripts/manage.py merge "Para A" "Para B" --into "Combined" --chapter 1 --json`
    *   **Rename**: `python scripts/manage.py rename "Old" "New" --chapter 1 --para --json`
    *   **Delete**: `python scripts/manage.py delete "Para" --chapter 1 --para --json`

4.  **Manage Assets (Images/Charts)**:
    If the user provides an image path:
    1.  Import it: `python scripts/manage.py add_asset "/abs/path/to/img.png" --json`
    2.  Use it in Markdown: `![Description](@/img.png)` (Note the `@/` prefix).

5.  **Edit Content Directly**:
    For modifying text inside `.md` files, use native tools (`view_file`, `replace_file_content`).
    The CLI is for **structure**, native tools are for **content**.

6.  **Review**:
    After creation, confirm the new path to the user.
