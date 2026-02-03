# ThesisFlow Best Practices

## 1. File Naming
- **Chapters**: Use `01_Title`, `02_Methods`. ThesisFlow enforces this automatically.
- **Paragraphs**: Use `01_Intro.md`, `02_Body.md`.
- **Assets**: Use snake_case for images, e.g., `graph_results_v1.png`.

## 2. Citation Management
- Use `manage.py add_citation` to ensure valid BibTeX.
- Preferred format:
  ```bibtex
  @article{key,
    author = {Doe, John},
    title = {Study of X},
    year = {2024}
  }
  ```
- In Markdown: Use `@key` or `[@key]` for citations.

## 3. Mathematical Notation
ThesisFlow supports standard LaTeX math within Markdown:
- **Inline**: `$E = mc^2$`
- **Block**:
  $$
  \int_{0}^{\infty} x^2 dx
  $$

## 4. Cross-Referencing
- Use standard Markdown links for sections: `[See Methods](#methods)`.
- Use asset paths relative to project root: `![Graph](assets/graph.png)`.
