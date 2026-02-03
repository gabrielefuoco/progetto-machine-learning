---
description: Initialize session, load context, and check stats
---

# Setup & Context Workflow

1.  **Load Context**:
    Get the full project state (structure, config, bibliography size).
    `python scripts/manage.py context --json`

2.  **Visual Confirmation**:
    Show the user the project structure to confirm understanding.
    `python scripts/manage.py tree --json`

3.  **Check Progress (Optional)**:
    If this is an existing project, analyze word counts and reading time.
    `python scripts/manage.py stats --json`

4.  **Validate Environment**:
    Ensure dependencies (Pandoc, Typst) are ready.
    `python scripts/manage.py check`
