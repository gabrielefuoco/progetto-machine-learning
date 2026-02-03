---
name: thesisflow-manager
description: The Definitive Thesis Management Engine. Provides a unified, robust CLI for structuring, writing, researching, and compiling complex academic documents.
---

# Thesisflow Manager Skill

> **For Agents:** This skill is designed to be your primary interface for thesis creation. It supports structured JSON output, context dumping, and self-healing mechanisms to ensure reliability.

## ⚡ Agent Best Practices (READ FIRST)

1.  **Always use `--json`**: When acting autonomously, append `--json` to **EVERY** command. This guarantees you receive a parseable response containing `status`, `message`, and `result` data, avoiding regex guessing.
2.  **Ground yourself with `context`**: At the start of a session, run `./manage.py context --json`. This gives you the full mental model of the project (structure, config, stats) in one token-efficient block.
3.  **Trust `validate --fix`**: If you suspect the project structure is broken (or if you encounter "file not found" errors), run `./manage.py validate --fix --json` to automatically repair configuration drift.
4.  **Use `add` for References**: Prefer `add "DOI/Title"` over manual BibTeX entry. It fetches high-quality metadata from Crossref and **automatically handles LaTeX character escaping** (e.g., `í` → `{\'i}`) to ensure your thesis compiles without errors.
5.  **Audit Regularly**: Use the `rag cite_check` and `rag validate_claims` tools frequently to ensure your writing is grounded in your research and properly cited.
6.  **Do not edit structure manually**: Never use `mkdir` or `touch` for chapters/paragraphs. Use the skill commands to ensure the `thesisflow.json` registry stays in sync.
6.  **PowerShell & JSON Encoding**: When redirecting JSON output to a file (e.g., `> out.json`), PowerShell defaults to UTF-16. Use `| Out-File -Encoding utf8 out.json` to ensure tools can read it.

> ### CLI vs Native Tools: Know the Difference
> 
> | Task | Use | Why |
> | :--- | :--- | :--- |
> | **Create/delete chapters** | CLI (`add_chapter`, `delete`) | Needs `thesisflow.json` sync |
> | **Create/delete paragraphs** | CLI (`add_paragraph`, `delete`) | Needs registration + numbering |
> | **Rename/move files** | CLI (`rename`, `move`) | Needs config sync + renumbering |
> | **Edit markdown content** | `view_file`, `replace_file_content` | Safe! Content ≠ structure |
> | **Read existing content** | `view_file` or CLI (`read`) | Both work, `view_file` is faster |
> 
> **TL;DR**: The CLI protects **structure** (folders, `thesisflow.json`). Your native tools are perfect for **content** (editing `.md` text). Editing the text inside `02_Methods.md` does NOT require CLI.

---

## 🎮 The CLI Interface
**Script Location:** `.agent/skills/thesisflow-manager/scripts/manage.py`
**Base Command:** `python scripts/manage.py [COMMAND] [ARGS...] --json`

### 1. Initialization & Context
| Command | Arguments | Description |
| :--- | :--- | :--- |
| `init` | `[NAME]` | Creates a thesis structure. Uses current directory if `NAME` is `.` or omitted. |
| `context` | `--project [PATH]` | **CRITICAL.** Returns status, tree, bib-count, and config. Run this first. |
| `check` | | Verifies that Pandoc, Typst, and templates are installed/available. |
| `find_text`| `[QUERY]` | Searches for exact text matches across the **draft** paragraphs only. |

### 2. Structure & Writing
| Command | Arguments | Description |
| :--- | :--- | :--- |
| `add_chapter` | `[TITLE]` | Creates a numbered chapter folder (e.g., `01_Introduction`) and updates config. |
| `add_paragraph` | `[TITLE] --chapter [NAME] [--raw]` | Creates a detailed markdown section. use `--raw` to omit the header (e.g. for Intros). |
| `insert` | `[TITLE] --chapter [ID] --after [PARA]` | **NEW!** Inserts paragraph at specific position. Renumbers subsequent paragraphs. |
| `tree` | | Returns a text visualization of the project structure. |
| `stats` | | Analyzes word count and estimated reading time per chapter. |

### 3. Research, QA & Bibliography
| Command | Arguments | Description |
| :--- | :--- | :--- |
| `add_citation` | `[BIBTEX_STRING]` | Appends a valid BibTeX entry to `references.bib`. |
| `add` | `[DOI_OR_TITLE]` | **SMART.** Fetches perfect metadata from Crossref and **auto-escapes LaTeX characters** for flawless compilation. |
| `clean_bib` | | **MAINTENANCE.** Refreshes metadata, corrects authors, and **fixes encoding/accents** across the entire bibliography. |
| `search_bib` | `[QUERY]` | Fuzzy searches the bibliography. parsing matches. |
| `bib patch` | `[KEY] --field [F] --value [V]` | Safely updates a single field in a citation (e.g. fixing a `file` path or `title`). |
| `bib check_files`| | Verifies that every `file` path in the bibliography actually exists on disk. |
| `bib dedupe`| | Removes duplicate entries based on ID or normalized title matches. |
| `add_asset` | `[FILE_PATH]` | Imports images/PDFs into the `assets/` folder for use in the thesis. |
| `lint` | | **QA Engine.** Checks for TODOs, broken image links, and missing citations. |

### 4. Compilation, Style & Production
| Command | Arguments | Description |
| :--- | :--- | :--- |
| `compile` | | Generates a production-ready PDF using Typst templates. Includes auto-pagebreaks and smart title cleaning. |
| `style list` | | Lists available Typst templates in `assets/templates/`. |
| `style set` | `[NAME]` | Switches the active template for compilation. |
| `graph` | | Generates a **Mermaid.js** flowchart of the project structure for visualization. |
| `export_docx` | | Generates a DOCX file for advisor review/comments. |
| `export` | `[ZIP_PATH]` | Creates a full backup archive of the project. |

### 6. Semantic Intelligence (RAG)
| Command | Arguments | Description |
| :--- | :--- | :--- |
| `rag sync_draft` | `--project [PATH]` | Indices the thesis structure with **Semantic Chunking** (70-100 words, split at citations). |
| `rag sync_research` | `--path [DIR]` | Indices research files. **Recursive by default.** |
| `rag sync_all` | | **NEW.** One-command sync for both draft and research. |
| `rag query` | `[draft/research] [QUERY] [--section]` | Performs a semantic search. **NEW:** Use `--section "Results,Discussion"` to filter by PDF section. |
| `rag check_redundancy`| `[--threshold] [--adaptive]` | **NEW.** Detects repeated concepts. Enriched with snippets and similarity %. |
| `rag expand` | `[CHUNK_ID]` | **NEW.** Retrieves context (neighbors) around a semantic match. Use to read before/after logic. |
| `rag map_orphan` | `[FILE] [KEY]` | **NEW.** Manually maps an unlinked file to a citation key via sidecar file. Stores mapping without editing `.bib`. |
| `rag verify_mapping` | `[--fix]` | **NEW.** Diagnostic tool. Lists all indexed research files NOT mapped to a citation key. Use `--fix` to auto-repair. |
| `rag cross_ref` | `[--threshold]` | **NEW.** Compares each draft paragraph against research files to find overlaps. |
| `rag cite_check` | | **INTEGRITY.** Scans draft to verify if top RAG research matches are cited via `[@key]`. |
| `rag validate_claims`| `[CLAIM]` | **VALIDATION.** Side-by-side view of a draft claim vs the top 3 research matches for verification. |
| `rag audit_citations`| `[--format summary]` | Verifies each `[@key]` citation: extracts context, compares with paper, returns confidence verdict. Use `--format summary` for a condensed view. |
| `rag analyze_low` | `[--file FILENAME]` | Summarizes low-confidence citations from an audit result JSON file (e.g., `audit_result.json`). |
| `rag find_text` | `[draft/research] [QUERY]` | Keyword search across the selected scope (draft paragraphs or indexed research text). |
| `rag search_content`| `[QUERY]` | Search across research metadata for keyword matches with rich document context. |
| `read_resource` | `[PATH]` | Reads raw content of any resource file (PDF, TXT, BIB). |

### 7. Zotero Integration
| Command | Arguments | Description |
| :--- | :--- | :--- |
| `zotero setup` | | Checks for global credentials and provides instructions for creating `.env`. |
| `zotero list_collections` | | Lists available Zotero collections (useful for finding IDs). |
| `zotero set_collection` | `[COLLECTION_ID]` | Links the current project to a specific Zotero collection. |
| `zotero sync` | | Downloads metadata to `references.bib` and PDF attachments to `assets/research/zotero/`. Auto-triggers RAG sync. |
| `zotero check` | | Scans markdown files for citations (e.g. `[@key]`) missing from the bibliography. |

---

### 8. Refactoring Primitives
| Command | Arguments | Description |
| :--- | :--- | :--- |
| `read` | `--chapter [ID] [--para [ID]]` | Returns chapter or paragraph content. Supports index (`1`), partial name, or exact path. |
| `merge` | `[PARA_A] [PARA_B] --into [TITLE] --chapter [ID]` | Combines two paragraphs into one. Use `--keep-originals` to preserve source files. |
| `move` | `[PARA] --from [CHAP_A] --to [CHAP_B]` | Relocates paragraph between chapters with auto-renumbering. |
| `rename` | `[OLD] [NEW] [--chapter [ID]] [--para]` | Renames chapter or paragraph. Use `--para` flag for paragraphs. |
| `delete` | `[TARGET] [--chapter [ID]] [--para] [--force]` | Removes chapter or paragraph. `--force` required for chapter deletion. |

> **Fuzzy Matching**: All `--chapter` and `--para` arguments support:
> - **Index**: `--chapter 2` → matches second chapter
> - **Partial name**: `--chapter "Approccio"` → matches "02_LApproccio_Metodologico"
> - **Exact name**: `--chapter "01_Introduction"`

---

### 9. Maintenance & Integrity
| Command | Arguments | Description |
| :--- | :--- | :--- |
| `validate` | `--fix` | Checks config vs filesystem. With `--fix`, auto-adds orphans, removes missing entries, and migrates compilation config. |
| `test` | | Runs the internal self-test suite (including RAG validation). |

---

## 🛠️ Advanced: Bibliography & File Formats

### BibTeX `file` Field Format
To ensure RAG correctly associates your PDF with its citation, the `file` field in `references.bib` should follow one of these formats:
1.  **Zotero/JabRef**: `Description:path/to/file.pdf:type` (e.g., `:assets/research/imported/2026_Smith.pdf:pdf`)
2.  **Simple Path**: `path/to/file.pdf` (e.g., `assets/research/imported/2026_Smith.pdf`)

The skill fixes common parsing issues (like double colon prefixes), but keeping paths relative to the project root is recommended.

#### Fuzzy Matching & Sidecar
- **Improved Normalization**: The skill now uses aggressive fuzzy matching for filenames (ignoring underscores, hyphens, and multiple spaces) and prioritizes clean BibTeX titles to bridge the gap between messy PDF names and clean references.
- **Manual Mapping Sidebar**: If automatic mapping fails, use `rag map_orphan` to create a link stored in `.thesisflow/rag/research/manual_mapping.json`.
- **Parallel Extraction**: PDF/document extraction is now parallelized for significantly faster RAG indexing on multi-core systems.

---

## 💡 Common Workflows

### 🆕 Starting a New Thesis
```bash
# 1. Initialize
python scripts/manage.py init "MyThesis" --json

# 2. Add Structure (Plan)
python scripts/manage.py add_chapter "Introduction" --project "MyThesis" --json
python scripts/manage.py add_chapter "Literature Review" --project "MyThesis" --json
python scripts/manage.py add_paragraph "Problem Statement" --chapter "Introduction" --content "The problem is..." --project "MyThesis" --json
```

### 🧠 Analyzing Existing Work
```bash
# 1. Get Context (Grounding)
python scripts/manage.py context --project "MyThesis" --json

# 2. Check Stats (Progress)
python scripts/manage.py stats --project "MyThesis" --json
```

### 🛠️ Fixing "Hallucinations"
If you think a file exists but the skill says no, or if you see a folder that isn't in the config:
```bash
python scripts/manage.py validate --project "MyThesis" --fix --json
```

### 🧬 Research & Brainstorming (RAG)
```bash
# 1. Sync what you wrote
python scripts/manage.py rag sync_draft --project "MyThesis" --json

# 2. Sync research papers
python scripts/manage.py rag sync_research --path "./papers" --project "MyThesis" --json

# 3. Ask your thesis
python scripts/manage.py rag query draft "What were my findings on BERT?" --json

### 🕵️ Audit for Semantic Redundancy
Identify if you are repeating yourself or accidentally copy-pasting paragraphs.
```bash
# 1. Sync content
python scripts/manage.py rag sync_draft --json

# 2. Check with standard semantic threshold (0.8)
python scripts/manage.py rag check_redundancy --json

# 3. Aggressive outlier detection (Adaptive mode)
# Useful for finding near-exact duplicates across chapters.
python scripts/manage.py rag check_redundancy --adaptive --multiplier 1.0 --json
```

- **Threshold**: Lower means "show me only very close matches" (e.g. 0.4).
- **Adaptive**: Statistical mode that flags paragraphs significantly more similar than the project average.
- **Snippets**: Output includes paragraph previews to quickly verify the finding.

### 🔬 Cross-Referencing Draft vs Research
Compare what you wrote against every chunk in the research bibliography. High matches indicate supporting evidence or missing citations.
```bash
# High precision (looking for exact evidence/quotes)
python scripts/manage.py rag cross_ref --threshold 0.7 --json

# Discovery mode (looking for thematic connections)
python scripts/manage.py rag cross_ref --threshold 0.45 --json
```
- **Threshold**: `0.4 - 0.5` is ideal for brainstorming/connections. `0.7+` is for validation.
```

### 📚 Automated Bibliography & Research (Zotero)
```bash
# 1. Setup (One-time)
python scripts/manage.py zotero setup --json

# 2. Link Project
python scripts/manage.py zotero list_collections --json
python scripts/manage.py zotero set_collection "YOUR_COLLECTION_ID" --project "MyThesis" --json

# 3. Sync & Index
python scripts/manage.py zotero sync --project "MyThesis" --json

# 4. Verify Citations
python scripts/manage.py zotero check --project "MyThesis" --json
```

---

## 📂 Internal Structure
*   `scripts/lib/`: Core logic (Project, Compilation, Stats).
*   `assets/templates/`: Typst templates controlling the PDF look.
*   `thesisflow.json`: The "Brain" of the project. Tracks order, metadata, and **compilation settings**.


### 10. Compilation Features
The `compile` command performs smart adjustments for professional output:
*   **Page Breaks**: Automatically inserts a page break before every new chapter.
*   **Title Cleaning**: Removes numbering prefixes (e.g., `01_01_Introduction` → `Introduction`).
*   **Heading Shift**: Demotes Markdown headers (H1 → H2) to respect Chapter hierarchy.
*   **Encoding Guard**: Automatically detects and repairs mojibake (corrupted characters) in `references.bib` before build.
*   **Asset Path Rewriting**: Use `@/filename.png` in Markdown to reference assets (auto-resolves to correct path).

#### Compilation Configuration
Customize compilation behavior via `thesisflow.json`:
```json
{
  "compilation": {
    "pandoc_extensions": "+tex_math_dollars",
    "page_break_before_chapter": true,
    "typst_template": "default_thesis"
  }
}
```

| Key | Default | Description |
| :--- | :--- | :--- |
| `pandoc_extensions` | `+tex_math_dollars` | Pandoc markdown extensions for conversion |
| `page_break_before_chapter` | `true` | Insert page breaks before each chapter |
| `typst_template` | `default_thesis` | Template file from `assets/templates/` |

### 7. Working with Images (Agent Guide)
**Standard Workflow:**
1.  **Add asset**: `add_asset /path/to/image.png --project MyThesis`
2.  **Reference in Markdown**: Use `![Caption](@/image.png)` syntax
3.  **Compile**: The `@/` prefix is automatically translated to the correct relative path

**Example:**
```markdown
![Transformer Architecture](@/transformer_architecture.svg)
```

> **Why `@/`?** This prefix abstracts away the complex relative path calculation needed when `.typ` files are nested in chapter folders but assets are in a sibling `assets/` directory.


### 10. Scientific Deep Research (Autonomous)
| Command | Arguments | Description |
| :--- | :--- | :--- |
| `research search` | `[QUERY] --mode [discovery/deep]` | **Discovery**: Returns JSON/Table of papers from PubMed/ArXiv.<br>**Deep**: Downloads PDFs, adds citations to `.bib`, and triggers RAG sync. |

> **Mode Differences**:
> - `discovery`: Fast, metadata-only. Good for exploring topics.
> - `deep`: Aggressive. Downloads full-text PDFs to `assets/research/imported/` and indexes them immediately for RAG.

---

## 🤝 Dependencies
*   **Python 3.10+**
*   **Pandoc** (System PATH) - For DOCX/Markdown conversion.
*   **Typst** (System PATH) - For PDF generation.
*   **sentence-transformers** & **faiss-cpu** - For Semantic RAG intelligence.
*   **pypdf** & **langchain-text-splitters** - For research document processing.
*   **pyzotero** & **python-dotenv** - For Zotero integration.
