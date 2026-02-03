# Agent Experience Report: thesisflow-manager
**Date**: 2026-01-29
**Session Context**: Implementing RAG fuzzy matching, manual mapping, parallel processing, and Windows path stability.

## 🚨 Critical Friction (Bugs & Failures)
- **Issue**: Parallel processing with `ProcessPoolExecutor` failed initially.
- **Cause**: Methods like `_extract_text` were instance methods, which are not easily picklable for multiprocess execution in some environments (especially on Windows).
- **Proposed Fix**: Methods used in parallel execution should be made `@staticmethod` and keep no state dependance. Already implemented during the session.

- **Issue**: `TypeError: ResearchPaper.__init__() missing 1 required positional argument: 'pdf_url'`
- **Cause**: The `ResearchPaper` dataclass requires `pdf_url` but it wasn't consistently provided in manual instantiation during tests.
- **Proposed Fix**: Consider making `pdf_url` optional in the dataclass if it's not always available for all sources, or enforce its presence in all fetchers.

## ⚠️ User Experience (Usability)
- **Issue**: Windows path handling of absolute paths with spaces.
- **Cause**: Standard `Path(p).resolve()` sometimes struggles when the shell passes paths with double quotes or unusual separators.
- **Proposed Fix**: Use a dedicated `realpath` helper (implemented) that strips quotes and handles Windows path normalization explicitly.

- **Issue**: `rag verify_mapping` output in JSON can be extremely verbose, listing every citation key even when 100% are mapped.
- **Proposed Fix**: Add a summary mode or a `--missing-only` flag to reduce token usage and improve readability.

## 💡 Feature Proposals
- **RAG Extraction Cache**: Extraction of text from PDFs is expensive. Implementing a small hash-based cache (saving `.txt` versions of indexed PDFs) would speed up re-indexing significantly after minor bibliography changes.
- **Interactive Map Orphan**: A CLI interactive mode where users can choose from the top 3 fuzzy matches for an orphan file.

## 📝 Documentation Gaps
- **Parallel processing requirements**: `SKILL.md` should note that extending the RAG tools requires static methods for parallel safety.
- **Manual Mapping Sidebar**: While documented now, the location and format of `manual_mapping.json` should be explicitly mentioned for users who want to edit it directly.
