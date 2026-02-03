# Agent Experience Report: thesisflow-manager

**Date**: 29 Gennaio 2026 - 22:25  
**Session Context**: Chapter 1 expansion (citation integration + pedagogical content creation) and Production workflow execution (validation, linting, PDF compilation)

---

## 🚨 Critical Friction (Bugs & Failures)

### Issue #1: No BibTeX Duplicate Detection Before Compilation

**Severity**: 🔴 **HIGH** (Compilation blocker)

**Problem**: 
The `compile` command failed with the following error:
```
error: failed to parse BibLaTeX (duplicate key "Dillon2025Investigat")
```

Three duplicate citation keys existed in `references.bib`:
- `Dillon2025Investigat` (lines 37 & 46)
- `Eriguchi2017Learning_t` (lines 54 & 63)
- `Marshall2009The_embodi` (lines 152 & 160)

**Root Cause**: 
- No pre-compilation validation step to detect duplicate BibTeX keys
- `validate` and `lint` commands do NOT check for BibTeX duplicates
- Only Typst compilation catches this error, which is too late in the workflow

**Impact**:
- Compilation blocked until manual intervention
- Required creating a custom Python script (`remove_duplicates.py`) to fix
- Wasted ~10 minutes debugging when this should be caught automatically

**Proposed Fix**:
1. **Add `bib validate` command** that checks for:
   - Duplicate keys
   - Malformed entries
   - Missing required fields
   - Invalid file paths in `file` field
2. **Integrate into `lint` command** to make bibliography validation part of QA
3. **Auto-run before `compile`** to prevent late-stage failures

**Suggested Implementation** (in `scripts/lib/bibliography.py`):
```python
def validate_bibliography(bib_path: str) -> dict:
    """Validates references.bib for common issues."""
    with open(bib_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all entries
    entries = re.findall(r'@\w+\{([^,]+),', content)
    
    # Detect duplicates
    duplicates = [key for key in entries if entries.count(key) > 1]
    
    issues = []
    if duplicates:
        issues.append({
            "type": "duplicate_key",
            "severity": "error",
            "keys": list(set(duplicates)),
            "message": f"Found {len(set(duplicates))} duplicate keys"
        })
    
    return {
        "status": "failed" if issues else "passed",
        "issues": issues,
        "total_entries": len(entries),
        "unique_entries": len(set(entries))
    }
```

---

## ⚠️ User Experience (Usability)

### Issue #2: Non-Existent `bib clean` Command

**Severity**: 🟡 **MEDIUM** (Confusion)

**Problem**:
Attempted to run:
```bash
python scripts/manage.py bib clean --json
```

Result: `error: invalid choice: 'clean' (choose from 'patch')`

**Root Cause**:
- Documentation shows `clean_bib` (root-level) and `bib patch` (subcommand)
- Inconsistent naming led to assumption that `bib clean` existed

**Proposed Fix**:
Standardize on `bib` namespace:
- `clean_bib` → `bib clean`
- Add `bib validate`  
- Keep `bib patch`

---

### Issue #3: Minimal PDF Inspection Output

**Severity**: 🟢 **LOW** (Enhancement)

**Current Output**:
```json
{"author": "...", "title": "...", "pages": 28}
```

**Enhancement Request**:
Add file size, creation date, citation count for better validation.

---

## 💡 Feature Proposals

### Feature #1: Pre-Compilation Checklist (`preflight`)

One command that runs ALL checks before compilation:
```bash
python scripts/manage.py preflight --json
```

Runs: `validate` + `bib validate` + `lint`

---

### Feature #2: Auto Duplicate Removal

`bib validate --fix` to automatically remove duplicates, keeping the most complete entry.

---

## 📝 Documentation Gaps

### Gap #1: Missing Troubleshooting Section

Need section on common compilation errors and fixes.

### Gap #2: Bibliography Best Practices

Should document:
- Always use `add` for imports
- Run `bib validate` before compile
- Keep file paths relative

---

## 🎯 Recommendations

**Critical** (Implement Now):
1. ✅ Add `bib validate` command
2. ✅ Integrate into `lint`

**Medium Priority**:
3. ⚙️ Standardize `bib` namespace
4. ⚙️ Add `preflight` command

**Low Priority**:
5. 💡 Enhanced PDF inspection
6. 📚 Documentation updates

---

## 📊 Session Metrics

**Commands Used**: 15+  
**Failed**: `compile` (duplicates), `bib clean` (doesn't exist)  
**Workarounds**: Manual Python script for duplicates

**Overall**: ⭐⭐⭐⭐☆ (4/5)  
Excellent RAG tools, but needs bibliography pre-validation.

---

## 🚀 Action Items

Ready to implement:
1. `bib validate` with duplicate detection
2. Integration into `lint`
3. SKILL.md troubleshooting section
4. Refactor `clean_bib` → `bib clean`

**Estimated Time**: ~1-2 hoursours
