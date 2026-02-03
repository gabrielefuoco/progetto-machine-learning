import argparse
import sys
import os
import json
from pathlib import Path
from lib.project import ProjectManager
from lib.check import EnvironmentChecker
from lib.stats import StatsAnalyzer
import unittest

# Force UTF-8 encoding for stdout/stderr to handle PowerShell piping/redirection
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def realpath(path_str):
    """Normalizes and resolves a path string for Windows/Unix compatibility."""
    if not path_str: return None
    # Remove surrounding quotes if they exist (sometimes agents or shells double quote)
    path_str = path_str.strip('"').strip("'")
    return Path(path_str).resolve()

def print_output(data, json_mode=False):
    if json_mode:
        print(json.dumps(data, indent=2))
    else:
        if isinstance(data, dict) and "message" in data:
            print(data["message"])
        elif isinstance(data, list):
            for item in data:
                print(item)
        else:
            print(data)

def main():
    # Parent parser for common flags like --json
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--json", action="store_true", help="Output results as JSON")

    parser = argparse.ArgumentParser(description="Thesisflow Manager - Manage your thesis structure via CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Init
    subparsers.add_parser("init", parents=[parent_parser], help="Initialize a new project").add_argument("name", nargs="?", default=".", help="Name of the project (use '.' for current directory)")
    
    # Add Chapter
    cp = subparsers.add_parser("add_chapter", parents=[parent_parser], help="Add a new chapter")
    cp.add_argument("title", help="Title of the chapter")
    cp.add_argument("--project", default=".", help="Project directory path")
    
    # Add Paragraph
    pp = subparsers.add_parser("add_paragraph", parents=[parent_parser], help="Add a new paragraph")
    pp.add_argument("title", help="Title of the paragraph")
    pp.add_argument("--chapter", required=True, help="Name (or partial name) of the chapter to add to")
    pp.add_argument("--content", default="", help="Initial content")
    pp.add_argument("--raw", action="store_true", help="Do not add a header (H2) to the file")
    pp.add_argument("--project", default=".", help="Project directory path")
    
    # Tree
    tp = subparsers.add_parser("tree", parents=[parent_parser], help="Show project structure")
    tp.add_argument("--project", default=".", help="Project directory path")

    # Compile
    comp = subparsers.add_parser("compile", parents=[parent_parser], help="Compile project to PDF")
    comp.add_argument("--project", default=".", help="Project directory path")

    # Add Citation
    cit = subparsers.add_parser("add_citation", parents=[parent_parser], help="Add a citation to references.bib")
    cit.add_argument("bibtex", help="BibTeX entry string")
    cit.add_argument("--project", default=".", help="Project directory path")

    # Search Citation
    sc = subparsers.add_parser("search_bib", parents=[parent_parser], help="Search bibliography")
    sc.add_argument("query", help="Search term")
    sc.add_argument("--project", default=".", help="Project directory path")

    # Add Asset
    ap = subparsers.add_parser("add_asset", parents=[parent_parser], help="Import an asset file")
    ap.add_argument("path", help="Path to the asset file")
    ap.add_argument("--project", default=".", help="Project directory path")
    
    # Export & Import
    ep = subparsers.add_parser("export", parents=[parent_parser], help="Export project to ZIP")
    ep.add_argument("dest", help="Destination path for ZIP")
    ep.add_argument("--project", default=".", help="Project directory path")

    ip = subparsers.add_parser("import", parents=[parent_parser], help="Import project from ZIP")
    ip.add_argument("src", help="Source ZIP path")
    
    # Export DOCX
    ed = subparsers.add_parser("export_docx", parents=[parent_parser], help="Export project to DOCX")
    ed.add_argument("--project", default=".", help="Project directory path")
    ed.add_argument("output", nargs="?", help="Output file path (optional)")

    # Validate
    vp = subparsers.add_parser("validate", parents=[parent_parser], help="Validate project integrity")
    vp.add_argument("--project", default=".", help="Project directory path")
    vp.add_argument("--fix", action="store_true", help="Attempt to fix structural issues")

    # Utils
    ipp = subparsers.add_parser("inspect_pdf", parents=[parent_parser], help="Inspect generated PDF metadata")
    ipp.add_argument("pdf_path", help="Path to PDF")

    subparsers.add_parser("check", parents=[parent_parser], help="Check environment dependencies")

    sp = subparsers.add_parser("stats", parents=[parent_parser], help="Show project statistics")
    sp.add_argument("--project", default=".", help="Project directory path")

    subparsers.add_parser("test", parents=[parent_parser], help="Run self-tests")
    
    # Context
    cx = subparsers.add_parser("context", parents=[parent_parser], help="Dump condensed project context for Agents")
    cx.add_argument("--project", default=".", help="Project directory path")

    # Lint
    lp = subparsers.add_parser("lint", parents=[parent_parser], help="Run QA checks")
    lp.add_argument("--project", default=".", help="Project directory path")

    # Graph
    gp = subparsers.add_parser("graph", parents=[parent_parser], help="Generate project structure graph")
    gp.add_argument("--project", default=".", help="Project directory path")

    # Style
    sp = subparsers.add_parser("style", parents=[parent_parser], help="Manage visual templates")
    sp.add_argument("--project", default=".", help="Project directory path")
    sp_sub = sp.add_subparsers(dest="style_command", help="Style actions")
    
    # Inherit parent_parser here so `style list --json` works
    sp_list = sp_sub.add_parser("list", parents=[parent_parser], help="List available styles")
    sp_list.add_argument("--project", default=".", help="Project directory path")
    
    sp_set = sp_sub.add_parser("set", parents=[parent_parser], help="Set current style")
    sp_set.add_argument("style_name", help="Name of the template (without .typ)")
    sp_set.add_argument("--project", default=".", help="Project directory path")

    # Find Text (Replacing grep dependency)
    ftp = subparsers.add_parser("find_text", parents=[parent_parser], help="Search for text across the thesis draft")
    ftp.add_argument("query", help="Text to search for")
    ftp.add_argument("--case-sensitive", action="store_true", help="Perform case-sensitive search")
    ftp.add_argument("--project", default=".", help="Project directory path")

    # RAG
    rag = subparsers.add_parser("rag", parents=[parent_parser], help="Manage RAG systems")
    rag.add_argument("--project", default=".", help="Project directory path")
    rag_sub = rag.add_subparsers(dest="rag_command", help="RAG actions")
    
    rag_sync_draft = rag_sub.add_parser("sync_draft", parents=[parent_parser], help="Sync thesis content for Draft RAG")
    rag_sync_draft.add_argument("--project", default=".", help="Project directory path")
    
    rag_sync_research = rag_sub.add_parser("sync_research", parents=[parent_parser], help="Sync research sources for Research RAG")
    rag_sync_research.add_argument("--path", required=True, help="Path to research files (e.g. PDF folder)")
    rag_sync_research.add_argument("--recursive", action="store_true", default=True, help="Search recursively (default: True)")
    rag_sync_research.add_argument("--project", default=".", help="Project directory path")
    
    rag_sync_all = rag_sub.add_parser("sync_all", parents=[parent_parser], help="Sync both draft and research RAG")
    rag_sync_all.add_argument("--research-path", help="Optional path for research files")
    rag_sync_all.add_argument("--project", default=".", help="Project directory path")
    
    rag_query = rag_sub.add_parser("query", parents=[parent_parser], help="Query a RAG scope")
    rag_query.add_argument("scope", choices=["draft", "research"], help="Which RAG to query")
    rag_query.add_argument("query_text", help="Semantic query string")
    rag_query.add_argument("--section", help="Filter by section (comma-separated, e.g. 'Methods,Results')")
    rag_query.add_argument("--project", default=".", help="Project directory path")

    rag_expand = rag_sub.add_parser("expand", parents=[parent_parser], help="Expand context around a chunk")
    rag_expand.add_argument("chunk_id", help="The Chunk ID to expand")
    rag_expand.add_argument("--scope", choices=["draft", "research"], default="research", help="Which RAG to scope")
    rag_expand.add_argument("--window", type=int, default=1, help="Number of chunks before/after")
    rag_expand.add_argument("--project", default=".", help="Project directory path")

    rag_check = rag_sub.add_parser("check_redundancy", parents=[parent_parser], help="Find redundant paragraphs in draft")
    rag_check.add_argument("--threshold", type=float, default=0.8, help="Distance threshold (default 0.8)")
    rag_check.add_argument("--adaptive", action="store_true", help="Use adaptive threshold based on stats")
    rag_check.add_argument("--multiplier", type=float, default=2.0, help="Std Dev multiplier for adaptive threshold")
    rag_check.add_argument("--project", default=".", help="Project directory path")

    rag_verify = rag_sub.add_parser("verify_mapping", parents=[parent_parser], help="Verify RAG citation mapping coverage")
    rag_verify.add_argument("--fix", action="store_true", help="Attempt to automatically fix unmapped documents")
    rag_verify.add_argument("--project", default=".", help="Project directory path")

    rag_map = rag_sub.add_parser("map_orphan", parents=[parent_parser], help="Manually map an orphan file to a citation key")
    rag_map.add_argument("file", help="Filename or path to the document")
    rag_map.add_argument("key", help="Citation key (with or without @)")
    rag_map.add_argument("--project", default=".", help="Project directory path")

    rag_xref = rag_sub.add_parser("cross_ref", parents=[parent_parser], help="Cross-reference draft content against research")
    rag_xref.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold (0-1, default 0.6)")
    rag_xref.add_argument("--project", default=".", help="Project directory path")

    rag_cite = rag_sub.add_parser("cite_check", parents=[parent_parser], help="Verify if similar research matches are cited in the draft")
    rag_cite.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold")
    rag_cite.add_argument("--project", default=".", help="Project directory path")

    rag_val = rag_sub.add_parser("validate_claims", parents=[parent_parser], help="Review draft claims against research side-by-side")
    rag_val.add_argument("query", help="Claim or draft snippet to validate")
    rag_val.add_argument("--project", default=".", help="Project directory path")

    rag_audit = rag_sub.add_parser("audit_citations", parents=[parent_parser], help="Verify citation integrity: does each [@key] match its paper content?")
    rag_audit.add_argument("--high-threshold", type=float, default=0.70, help="Threshold for HIGH_CONFIDENCE (default 0.70)")
    rag_audit.add_argument("--low-threshold", type=float, default=0.50, help="Threshold for LOW_CONFIDENCE (default 0.50)")
    rag_audit.add_argument("--context", type=int, default=150, help="Characters of context around citation (default 150)")
    rag_audit.add_argument("--format", choices=["detail", "summary"], default="detail", help="Output format (default: detail)")
    rag_audit.add_argument("--project", default=".", help="Project directory path")
 
    rag_analyze = rag_sub.add_parser("analyze_low", parents=[parent_parser], help="Summarize low-confidence citations from an audit result file")
    rag_analyze.add_argument("--file", default="audit_result.json", help="Audit result JSON file to analyze (default: audit_result.json)")
    rag_analyze.add_argument("--project", default=".", help="Project directory path")

    rag_find = rag_sub.add_parser("find_text", parents=[parent_parser], help="Keyword search across the research index or draft")
    rag_find.add_argument("scope", choices=["research", "draft"], help="Search scope")
    rag_find.add_argument("query", help="Text to search for")
    rag_find.add_argument("--project", default=".", help="Project directory path")

    rag_search = rag_sub.add_parser("search_content", parents=[parent_parser], help="Rich keyword search across research metadata/PDF chunks")
    rag_search.add_argument("query", help="Text to search for")
    rag_search.add_argument("--project", default=".", help="Project directory path")

    # Read Resource
    rres = subparsers.add_parser("read_resource", parents=[parent_parser], help="Read a raw resource file (PDF, etc)")
    rres.add_argument("path", help="Path to the file (relative to project or absolute)")
    rres.add_argument("--project", default=".", help="Project directory path")

    # Zotero
    zp = subparsers.add_parser("zotero", parents=[parent_parser], help="Manage Zotero integration")
    zp.add_argument("--project", default=".", help="Project directory path")
    zp_sub = zp.add_subparsers(dest="zotero_command", help="Zotero actions")

    zp_setup = zp_sub.add_parser("setup", parents=[parent_parser], help="Check and set up Zotero credentials")
    
    zp_list = zp_sub.add_parser("list_collections", parents=[parent_parser], help="List Zotero collections")
    
    zp_set = zp_sub.add_parser("set_collection", parents=[parent_parser], help="Set the active Zotero collection")
    zp_set.add_argument("collection_id", help="Zotero Collection ID")
    zp_set.add_argument("--project", default=".", help="Project directory path")

    zp_sync = zp_sub.add_parser("sync", parents=[parent_parser], help="Sync bibliography and attachments")
    zp_sync.add_argument("--project", default=".", help="Project directory path")

    zp_check = zp_sub.add_parser("check", parents=[parent_parser], help="Validate citations against Zotero")
    zp_check.add_argument("--project", default=".", help="Project directory path")

    # Research
    research_parser = subparsers.add_parser("research", parents=[parent_parser], help="Manage research and citations")
    research_parser.add_argument("--project", default=".", help="Project directory path")
    research_subparsers = research_parser.add_subparsers(dest="research_command", help="Research actions")
    
    search_parser = research_subparsers.add_parser("search", parents=[parent_parser], help="Search for papers")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--mode", choices=["discovery", "deep"], default="discovery", help="Search mode")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results")
    search_parser.add_argument("--project", default=".", help="Project directory path")

    # Download single paper
    dl_parser = research_subparsers.add_parser("download", parents=[parent_parser], help="Download a single paper by DOI or title")
    dl_parser.add_argument("query", help="DOI (e.g. 10.1016/...) or paper title")
    dl_parser.add_argument("--project", default=".", help="Project directory path")

    # Add Reference command
    add_parser = subparsers.add_parser("add", parents=[parent_parser], help="Add a specific reference by DOI or Title to bibliography")
    add_parser.add_argument("query", help="DOI or Title of the paper to add")
    add_parser.add_argument("--project", default=".", help="Project directory path")

    # Clean Bib
    cb = subparsers.add_parser("clean_bib", parents=[parent_parser], help="Clean and repair bibliography metadata")
    cb.add_argument("--project", default=".", help="Project directory path")

    # Bib Utils (Patch)
    bib_parser = subparsers.add_parser("bib", parents=[parent_parser], help="Bibliography utilities")
    bib_parser.add_argument("--project", default=".", help="Project directory path")
    bib_sub = bib_parser.add_subparsers(dest="bib_command", help="Bib actions")
    
    bib_patch = bib_sub.add_parser("patch", parents=[parent_parser], help="Patch a bibliographic entry")
    bib_patch.add_argument("key", help="Citation key (e.g. @Author2020)")
    bib_patch.add_argument("--field", required=True, help="Field to update (e.g. title, file, year)")
    bib_patch.add_argument("--value", required=True, help="New value")
    bib_patch.add_argument("--project", default=".", help="Project directory path")

    bib_check = bib_sub.add_parser("check_files", parents=[parent_parser], help="Verify that every 'file' path in the bibliography exists")
    bib_check.add_argument("--project", default=".", help="Project directory path")

    bib_dedupe = bib_sub.add_parser("dedupe", parents=[parent_parser], help="Safe bibliography deduplication")
    bib_dedupe.add_argument("--project", default=".", help="Project directory path")

    # ========== REFACTORING PRIMITIVES ==========
    
    # Read
    rp = subparsers.add_parser("read", parents=[parent_parser], help="Read chapter or paragraph content")
    rp.add_argument("--chapter", required=True, help="Chapter ID (index, partial name, or full name)")
    rp.add_argument("--para", help="Paragraph ID (optional - if omitted, reads entire chapter)")
    rp.add_argument("--project", default=".", help="Project directory path")

    # Merge
    mp = subparsers.add_parser("merge", parents=[parent_parser], help="Merge two paragraphs into one")
    mp.add_argument("para_a", help="First paragraph to merge")
    mp.add_argument("para_b", help="Second paragraph to merge")
    mp.add_argument("--into", required=True, help="Title for the merged paragraph")
    mp.add_argument("--chapter", required=True, help="Chapter containing the paragraphs")
    mp.add_argument("--keep-originals", action="store_true", help="Keep original files after merge")
    mp.add_argument("--project", default=".", help="Project directory path")

    # Move
    mvp = subparsers.add_parser("move", parents=[parent_parser], help="Move paragraph between chapters")
    mvp.add_argument("para", help="Paragraph to move")
    mvp.add_argument("--from", dest="from_chapter", required=True, help="Source chapter")
    mvp.add_argument("--to", dest="to_chapter", required=True, help="Target chapter")
    mvp.add_argument("--project", default=".", help="Project directory path")

    # Rename
    rnp = subparsers.add_parser("rename", parents=[parent_parser], help="Rename chapter or paragraph")
    rnp.add_argument("old_name", help="Current name/ID")
    rnp.add_argument("new_name", help="New name")
    rnp.add_argument("--chapter", help="For renaming paragraphs: specify the chapter")
    rnp.add_argument("--para", action="store_true", help="Rename a paragraph instead of chapter")
    rnp.add_argument("--project", default=".", help="Project directory path")

    # Delete
    dp = subparsers.add_parser("delete", parents=[parent_parser], help="Delete chapter or paragraph")
    dp.add_argument("target", help="Chapter or paragraph to delete")
    dp.add_argument("--chapter", help="For deleting paragraphs: specify the chapter")
    dp.add_argument("--para", action="store_true", help="Delete a paragraph instead of chapter")
    dp.add_argument("--force", action="store_true", help="Required for chapter deletion")
    dp.add_argument("--project", default=".", help="Project directory path")

    # Insert (positional paragraph creation)
    inp = subparsers.add_parser("insert", parents=[parent_parser], help="Insert paragraph at specific position")
    inp.add_argument("title", help="Title of the new paragraph")
    inp.add_argument("--chapter", required=True, help="Chapter to insert into")
    inp.add_argument("--after", help="Insert after this paragraph (fuzzy match)")
    inp.add_argument("--content", default="", help="Initial content")
    inp.add_argument("--raw", action="store_true", help="Do not add a header (H2) to the file")
    inp.add_argument("--project", default=".", help="Project directory path")

    args = parser.parse_args()
    
    # Use smart root resolution
    from lib.project import resolve_project_root
    if args.command == "init":
        cwd = Path.cwd()
    else:
        cwd = resolve_project_root(Path(__file__))
    
    manager = ProjectManager(root_path=cwd)
    
    # Helper to resolve project root
    def resolve_root(args, attr='project'):
        p = getattr(args, attr, None)
        if p is None or p == '.':
            return cwd
        root = realpath(p) # Use our new helper
        if not root.exists(): root = cwd / p
        return root

    try:
        result = None
        
        if args.command == "init":
            path = manager.init_project(args.name)
            result = {"message": f"Project initialized at: {path}", "path": str(path)}
            
        elif args.command == "add_chapter":
            root = resolve_root(args)
            path = manager.add_chapter(root, args.title)
            result = {"message": f"Chapter created: {path.name}", "path": str(path)}
            
        elif args.command == "add_paragraph":
            root = resolve_root(args)
            project = manager.load_project(root)
            target_chapter = None
            for c in project.chapters:
                if args.chapter.lower() in c.title.lower():
                    target_chapter = c
                    break
            if not target_chapter:
                raise ValueError(f"Chapter matching '{args.chapter}' not found.")
                
            path = manager.add_paragraph(target_chapter.path, args.title, args.content, include_header=not args.raw)
            result = {"message": f"Paragraph created: {path.name}", "path": str(path)}
            
        elif args.command == "tree":
            root = resolve_root(args)
            # Tree usually returns string, for JSON we might want a structured dict?
            # ProjectManager.get_structure_tree returns string. 
            # Ideally we'd have a get_structure_dict. For now, returning string in JSON.
            tree_str = manager.get_structure_tree(root)
            result = {"message": tree_str, "tree": tree_str}

        elif args.command == "compile":
            root = resolve_root(args)
            pdf_path = manager.compile_project(root)
            result = {"message": f"Compilation successful: {pdf_path}", "path": str(pdf_path)}
            
        elif args.command == "export_docx":
            root = resolve_root(args)
            project = manager.load_project(root)
            out_arg = getattr(args, 'output', None)
            out = Path(out_arg) if out_arg else None
            docx_path = manager.compiler.compile_docx(project, out)
            result = {"message": f"DOCX Export successful: {docx_path}", "path": str(docx_path)}

        elif args.command == "validate":
            root = resolve_root(args)
            fix = getattr(args, 'fix', False)
            issues = manager.validate_structure(root, fix=fix)
            if issues:
                if fix:
                    result = {"message": "Validation found issues (Fixed):", "issues": issues, "status": "fixed"}
                else:
                    # In json mode, we return success=False logic? 
                    # Usually better to return a dict with "issues" list.
                    print_output({"message": "Validation Failed", "issues": issues, "status": "failed"}, args.json)
                    sys.exit(1)
            else:
                result = {"message": "Validation Passed: Project structure is intact.", "status": "passed", "issues": []}

        elif args.command == "inspect_pdf":
            try:
                from pypdf import PdfReader
                path = Path(args.pdf_path)
                if not path.exists(): raise FileNotFoundError("PDF not found")
                reader = PdfReader(path)
                meta = reader.metadata
                info = {
                    "author": meta.author,
                    "title": meta.title,
                    "pages": len(reader.pages)
                }
                result = info
                result["message"] = f"PDF Info: {info}"
            except Exception as e:
                raise RuntimeError(f"Error inspecting PDF: {e}")

        elif args.command == "add_citation":
            root = resolve_root(args)
            manager.add_citation(root, args.bibtex)
            result = {"message": "Citation added to bibliography.", "success": True}

        elif args.command == "search_bib":
            root = resolve_root(args)
            results = manager.search_bibliography(root, args.query)
            # Result is list of dicts
            formatted = []
            for r in results:
                formatted.append(f"- [{r.get('ID', '?')}] {r.get('title', 'No Title')}")
            result = results # JSON gets raw dicts
            # For text output, we cheat and swap 'result' locally just for printing if not json
            if not args.json:
                result = formatted

        elif args.command == "add_asset":
            root = resolve_root(args)
            asset_rel_path = manager.add_asset(root, args.path)
            result = {"message": f"Asset added: {asset_rel_path}", "path": asset_rel_path}
            
        elif args.command == "export":
            root = resolve_root(args)
            dest = Path(args.dest)
            out = manager.export_project(root, dest)
            result = {"message": f"Project exported to: {out}", "path": str(out)}

        elif args.command == "import":
            src = Path(args.src)
            out = manager.import_project(src)
            result = {"message": f"Project imported to: {out}", "path": str(out)}
            
        elif args.command == "read_resource":
             root = resolve_root(args)
             txt = manager.rag_service.read_full_resource(root, args.path)
             result = {"message": f"Read resource {args.path}", "content": txt}
             if not args.json:
                 print(txt)
                 result = None

        elif args.command == "check":
            all_ok = EnvironmentChecker.print_report()
            sys.exit(0 if all_ok else 1)

        elif args.command == "stats":
            root = resolve_root(args)
            project = manager.load_project(root)
            analyzer = StatsAnalyzer(project)
            analyzer.analyze()
            sys.exit(0)

        elif args.command == "lint":
            from lib.lint import Linter
            root = resolve_root(args)
            project = manager.load_project(root)
            linter = Linter(project)
            issues = linter.check()
            if issues:
                 result = {"message": "Linting completed with issues.", "issues": issues, "status": "issues_found"}
            else:
                 result = {"message": "Linting passed. No issues found.", "issues": [], "status": "passed"}

        elif args.command == "graph":
            from lib.graph import GraphGenerator
            root = resolve_root(args)
            project = manager.load_project(root)
            gen = GraphGenerator(project)
            mermaid = gen.generate_mermaid()
            result = {"message": "Graph generated.", "mermaid": mermaid}
            if not args.json:
                print(mermaid)
                result = None # Handled print

        elif args.command == "style":
            from lib.style import StyleManager
            root = resolve_root(args)
            styler = StyleManager(root)
            
            sub = getattr(args, "style_command", "list")
            if sub == "list":
                styles = styler.list_styles()
                current = styler.get_current_style()
                result = {"styles": styles, "current": current}
                if not args.json:
                    print(f"Current Style: {current}")
                    print("Available Styles:")
                    for s in styles: print(f" - {s}")
                    result = None
            elif sub == "set":
                name = getattr(args, "style_name", "default_thesis")
                if styler.set_style(name):
                    result = {"message": f"Style updated to '{name}'", "success": True}
                else:
                    raise ValueError(f"Style '{name}' not found.")

        elif args.command == "find_text":
            root = resolve_root(args)
            project = manager.load_project(root)
            query = args.query
            case_sensitive = getattr(args, "case_sensitive", False)
            
            matches = []
            for chap in project.chapters:
                for para in chap.paragraphs:
                    content = para.content
                    match_content = content if case_sensitive else content.lower()
                    match_query = query if case_sensitive else query.lower()
                    
                    if match_query in match_content:
                        # Find line numbers or just snippets
                        lines = content.split("\n")
                        for i, line in enumerate(lines):
                            if (match_query in line if case_sensitive else match_query in line.lower()):
                                matches.append({
                                    "chapter": chap.title,
                                    "paragraph": para.title,
                                    "path": str(para.path.relative_to(root)),
                                    "line": i + 1,
                                    "content": line.strip()
                                })
            
            if not matches:
                result = {"message": f"No matches found for '{query}'.", "matches": []}
            else:
                result = {"message": f"Found {len(matches)} matches for '{query}':", "matches": matches}
                if not args.json:
                    out = [result["message"]]
                    for m in matches:
                        out.append(f"[{m['chapter']} / {m['paragraph']}] L{m['line']}: {m['content']}")
                    result = out

        elif args.command == "context":
            root = resolve_root(args)
            project = manager.load_project(root)
            
            # Build Context Data
            ctx = {
                "project_name": project.name,
                "chapter_count": len(project.chapters),
                "structure": manager.get_structure_tree(root),
                "bibliography_count": len(manager.bib_service.search("")),
                "config": manager._load_config(root),
            }
            result = ctx
            result["message"] = f"Loaded Context for {project.name}"

        elif args.command == "rag":
            root = resolve_root(args)
            sub = getattr(args, "rag_command", None)
            
            if sub == "sync_draft":
                project = manager.load_project(root)
                if manager.rag_service.sync_draft(root, project.chapters):
                    result = {"message": "Draft RAG synchronized successfully.", "success": True}
                else:
                    result = {"message": "Draft RAG synchronization failed (no content?).", "success": False}
            
            elif sub == "sync_research":
                path = realpath(args.path)
                if not path.exists():
                     raise FileNotFoundError(f"Research path {path} not found.")
                
                extensions = ["*.pdf", "*.docx", "*.odt", "*.rtf", "*.txt", "*.md", "*.tex", "*.rst", "*.org"]
                research_files = []
                recursive = getattr(args, "recursive", True)
                glob_pattern = "**/" if recursive else ""
                
                for ext in extensions:
                    research_files.extend(list(path.glob(f"{glob_pattern}{ext}")))
                
                if not research_files:
                    result = {"message": f"No supported research files found in {path}", "success": False}
                elif manager.rag_service.sync_research(root, research_files):
                    result = {"message": f"Research RAG synchronized successfully with {len(research_files)} files.", "success": True}
                else:
                    result = {"message": "Research RAG synchronization failed.", "success": False}
            
            elif sub == "sync_all":
                # Sync Draft
                project = manager.load_project(root)
                draft_ok = manager.rag_service.sync_draft(root, project.chapters)
                
                # Sync Research (Auto-detect assets/research if not provided)
                res_path_str = getattr(args, "research_path", None)
                if not res_path_str:
                    res_path = root / "assets" / "research"
                else:
                    res_path = Path(res_path_str)
                
                res_ok = False
                res_count = 0
                if res_path.exists():
                    extensions = ["*.pdf", "*.docx", "*.odt", "*.rtf", "*.txt", "*.md", "*.tex", "*.rst", "*.org"]
                    research_files = []
                    for ext in extensions:
                        research_files.extend(list(res_path.glob(f"**/{ext}")))
                    res_count = len(research_files)
                    if research_files:
                        res_ok = manager.rag_service.sync_research(root, research_files)
                
                result = {
                    "message": "Sync All complete.",
                    "draft_success": draft_ok,
                    "research_success": res_ok,
                    "research_files_found": res_count,
                    "success": draft_ok and (res_ok or res_count == 0)
                }
            
            elif sub == "query":
                sec_filter = args.section.split(",") if args.section else None
                results = manager.rag_service.query(root, args.scope, args.query_text, section_filter=sec_filter)
                if not results:
                    result = {"message": f"No results found in {args.scope} RAG.", "results": []}
                else:
                    result = {"message": f"Found {len(results)} matches in {args.scope} RAG:", "results": results}
                    if not args.json:
                        # Print pretty for user
                        out = [result["message"]]
                        for r in results:
                            dist = f"(dist: {r['distance']:.3f})"
                            # Display key if available
                            key = r.get('citation_key', '')
                            key_str = f"[Key: @{key}] " if key else ""
                            
                            if args.scope == "draft":
                                out.append(f" - {r['chapter']} / {r['title']} {dist}")
                            else:
                                out.append(f" - {key_str}{Path(r['path']).name} [Page {r.get('page','?')}] [ChunkID: {r.get('chunk_id','')}] {dist}")
                        result = out
            
            elif sub == "expand":
                 data = manager.rag_service.expand_context(root, args.scope, args.chunk_id, window=args.window)
                 if not data:
                      result = {"message": f"Chunk {args.chunk_id} not found or no context available.", "context": None, "success": False}
                 else:
                      result = {"message": "Context expanded.", "context": data, "success": True}
                      if not args.json:
                           # Pretty print
                           c = data["center"]
                           out = ["=== Expanded Context ===", ""]
                           
                           # Before
                           for b in data["before"]:
                               out.append(f"[PREV] {b.get('text', '(No Text)')[:100]}...")
                               out.append("")
                           
                           # Center
                           out.append(f"[TARGET] {c.get('text', '(No Text)')}")
                           out.append("")
                           
                           # After
                           for a in data["after"]:
                               out.append(f"[NEXT] {a.get('text', '(No Text)')[:100]}...")
                           
                           result = out

            elif sub == "check_redundancy":
                res_list = manager.rag_service.detect_redundancies(
                    root, 
                    threshold=args.threshold, 
                    adaptive=args.adaptive, 
                    multiplier=args.multiplier
                )
                if not res_list:
                    result = {"message": "No redundant paragraphs detected.", "redundancies": []}
                else:
                    msg = f"Detected {len(res_list)} potentially redundant pairs:"
                    result = {"message": msg, "redundancies": res_list}
                    if not args.json:
                        # Richer Human Output
                        out = [msg, ""]
                        for i, r in enumerate(res_list):
                            score = int(r.get('similarity', 0.0) * 100)
                            a = r['para_a']
                            b = r['para_b']
                            
                            out.append(f"Pair #{i+1} [Similarity: {score}%]")
                            out.append(f"  A: {a['chapter']} / {a['title']}")
                            out.append(f"     \"{a.get('snippet', '')}\"")
                            out.append(f"  B: {b['chapter']} / {b['title']}")
                            out.append(f"     \"{b.get('snippet', '')}\"")
                            out.append(f"  L2 Distance: {r['distance']:.3f}")
                            out.append("-" * 60)
                        
                        if args.adaptive:
                            t_used = res_list[0]['threshold_used']
                            out.append(f"Using ADAPTIVE threshold: {t_used:.3f} (Multiplier: {args.multiplier}x)")
                        else:
                            out.append(f"Using FIXED threshold: {args.threshold}")
                        result = out
                        result = out
            
            elif sub == "verify_mapping":
                if getattr(args, "fix", False):
                    result = manager.rag_service.fix_mapping(root)
                else:
                    stats = manager.rag_service.verify_mapping(root)
                    result = {"message": "Verification complete.", "stats": stats}
                    if not args.json:
                         print("=== RAG Mapping Verification ===")
                         print(f"Total Chunks:      {stats['total_chunks']}")
                         print(f"Total Documents:   {stats['total_documents']}")
                         print(f"Mapped Documents:  {stats['mapped_documents']}")
                         print(f"Unmapped Documents:{len(stats['unmapped_documents'])}")
                         print(f"Bib Mapping Size:  {stats['bib_mapping_size']}")
                         if stats['unmapped_documents']:
                             print("\n[!] The following files are NOT mapped to any citation key:")
                             for p in stats['unmapped_documents'][:10]:
                                 print(f" - {Path(p).name}")
                             if len(stats['unmapped_documents']) > 10:
                                 print(f" ... and {len(stats['unmapped_documents']) - 10} more.")
                         else:
                             print("\n[OK] All indexed documents are mapped.")
                         result = None
            
            elif sub == "map_orphan":
                if manager.rag_service.map_orphan(root, args.file, args.key):
                    result = {"message": f"Successfully mapped '{args.file}' to @{args.key.lstrip('@')}.", "success": True}
                else:
                    result = {"message": f"Failed to map '{args.file}'.", "success": False}
            
            elif sub == "cross_ref":
                xref_results = manager.rag_service.cross_reference(root, threshold=args.threshold)
                if not xref_results:
                    result = {"message": "No significant overlaps detected between draft and research.", "results": []}
                else:
                    msg = f"Found {len(xref_results)} draft paragraphs with research matches:"
                    result = {"message": msg, "results": xref_results}
                    if not args.json:
                        out = ["=== RAG Cross-Reference (Draft vs Research) ===", ""]
                        for item in xref_results:
                            d = item["draft"]
                            matches = item["matches"]
                            
                            out.append(f"📄 [DRAFT] {d['chapter']} / {d['title']}")
                            out.append(f"   \"{d.get('snippet', '')}\"")
                            out.append("")
                            for m in matches:
                                sim = int(m['similarity'] * 100)
                                key = m.get('citation_key', 'Unknown')
                                out.append(f"   🔍 [MATCH] @{key} ({sim}%) - Page {m.get('page', '?')}")
                                # Chunk snippet
                                out.append(f"      \"{m.get('text', '')[:120]}...\"")
                            out.append("-" * 60)
                        result = out

            elif sub == "cite_check":
                xref_results = manager.rag_service.cross_reference(root, threshold=args.threshold)
                missing = []
                for item in xref_results:
                    d = item["draft"]
                    matches = item["matches"]
                    d_content = manager.rag_service.read_full_resource(root, d["path"])
                    
                    for m in matches:
                        key = m.get("citation_key")
                        if key and f"@{key}" not in d_content:
                            missing.append({
                                "chapter": d["chapter"],
                                "paragraph": d["title"],
                                "match_key": key,
                                "similarity": m["similarity"],
                                "path": d["path"]
                            })
                
                if not missing:
                    result = {"message": "All similar research matches seem to be cited in the draft.", "missing": []}
                else:
                    msg = f"Found {len(missing)} potentially missing citations:"
                    result = {"message": msg, "missing": missing}
                    if not args.json:
                        out = [msg, ""]
                        for item in missing:
                            out.append(f"❌ [MISSING] @{item['match_key']} ({int(item['similarity']*100)}%) in {item['chapter']} / {item['paragraph']}")
                        result = out

            elif sub == "validate_claims":
                # Side-by-side view of a claim vs research
                results = manager.rag_service.query(root, "research", args.query, top_k=3)
                
                # Fetch full context for the top match to show side-by-side
                if not results:
                    result = {"message": "No research matches found for this claim.", "success": False}
                else:
                    msg = "Side-by-Side Validation View:"
                    result = {"message": msg, "claim": args.query, "matches": results}
                    if not args.json:
                        out = [msg, "", f"📢 [DRAFT CLAIM] {args.query}", ""]
                        out.append("=" * 60)
                        for i, m in enumerate(results):
                            sim = int(m['similarity'] * 100) if 'similarity' in m else int((1/(1+m.get('distance', 1)))*100)
                            key = m.get('citation_key', 'Unknown')
                            out.append(f"🔍 [MATCH #{i+1}] @{key} (Sim: {sim}%)")
                            out.append(f"   \"{m.get('text', '')}\"")
                            out.append("-" * 40)
                        result = out

            elif sub == "audit_citations":
                audit_result = manager.rag_service.audit_citations(
                    root,
                    context_chars=args.context,
                    high_threshold=args.high_threshold,
                    low_threshold=args.low_threshold
                )
                
                if not audit_result.get("success", False):
                    result = audit_result
                else:
                    result = audit_result
                    if not args.json:
                        out = ["=== Citation Integrity Audit ===", ""]
                        out.append(f"Total Citations: {audit_result['total_citations']}")
                        out.append(f"  ✅ HIGH_CONFIDENCE:   {audit_result['high_confidence']}")
                        out.append(f"  ⚠️  REVIEW_SUGGESTED: {audit_result['review_suggested']}")
                        out.append(f"  ❌ LOW_CONFIDENCE:    {audit_result['low_confidence']}")
                        out.append(f"  🔗 UNMAPPED:          {audit_result['unmapped_keys']}")
                        out.append("")
                        
                        if audit_result['issues'] and getattr(args, "format", "detail") == "detail":
                            out.append("=== Issues Found ===")
                            for issue in audit_result['issues']:
                                sim_str = f"({int(issue['similarity']*100)}%)" if issue['similarity'] else ""
                                out.append(f"[{issue['type']}] @{issue['key']} {sim_str}")
                                out.append(f"   Location: {issue['location']}")
                                out.append(f"   ⚡ {issue['suggestion']}")
                                out.append("")
                        elif audit_result['issues'] and getattr(args, "format", "detail") == "summary":
                            out.append("=== Summary of Issues ===")
                            # Groups by type
                            by_type = {}
                            for issue in audit_result['issues']:
                                t = issue['type']
                                by_type[t] = by_type.get(t, 0) + 1
                            for t, count in by_type.items():
                                out.append(f" - {t}: {count} instance(s)")
                            out.append("")
                            out.append("Run without '--format summary' for full details.")
                        else:
                            out.append("✅ No issues found. All citations appear well-supported.")
                        
                        result = out

            elif sub == "find_text":
                # rag find_text <scope> <query>
                if args.scope == "draft":
                    # Reuse root find_text logic
                    # This is slightly redundant but preserves the critique's requested structure
                    project = manager.load_project(root)
                    query = args.query
                    matches = []
                    for chap in project.chapters:
                        for para in chap.paragraphs:
                            content = para.content
                            if query.lower() in content.lower():
                                lines = content.split("\n")
                                for i, line in enumerate(lines):
                                    if query.lower() in line.lower():
                                        matches.append({
                                            "chapter": chap.title, "paragraph": para.title,
                                            "path": str(para.path.relative_to(root)),
                                            "line": i + 1, "content": line.strip()
                                        })
                    if not matches:
                        result = {"message": f"No matches found in draft for '{query}'.", "matches": []}
                    else:
                        result = {"message": f"Found {len(matches)} matches in draft for '{query}':", "matches": matches}
                else:
                    # Research search
                    query = args.query
                    matches = manager.rag_service.research_text_search(root, query)
                    if not matches:
                        result = {"message": f"No matches found in research for '{query}'.", "matches": []}
                    else:
                        result = {"message": f"Found {len(matches)} matches in research for '{query}':", "matches": matches}
                        if not args.json:
                            out = [result["message"]]
                            for m in matches:
                                key = m.get("citation_key", "Unknown")
                                out.append(f"[@{key}] {Path(m['path']).name} (Page {m.get('page','?')})")
                                out.append(f"   {m['snippet']}")
                            result = out

            elif sub == "search_content":
                query = args.query
                matches = manager.rag_service.search_content(root, query)
                if not matches:
                    result = {"message": f"No matches found for '{query}'.", "matches": []}
                else:
                    result = {"message": f"Found {len(matches)} matches for '{query}':", "matches": matches}
                    if not args.json:
                        out = [result["message"]]
                        for m in matches:
                            key = m.get("citation_key", "Unknown")
                            out.append(f"📄 {Path(m['path']).name} [@{key}] P{m.get('page','?')}")
                            out.append(f"   {m['snippet']}")
                        result = out
 
            elif sub == "analyze_low":
                root = resolve_root(args)
                audit_file = root / args.file
                if not audit_file.exists():
                    raise FileNotFoundError(f"Audit file not found: {audit_file}. Run 'rag audit_citations --json | Out-File -Encoding utf8 {args.file}' first.")
                
                try:
                    # Handle potential UTF-8 BOM
                    content = audit_file.read_text(encoding='utf-8-sig')
                    data = json.loads(content)
                    
                    citations = data.get("citations", [])
                    low = [c for c in citations if c.get("verdict") == "LOW_CONFIDENCE"]
                    
                    msg = f"Analysis of {args.file}: Found {len(low)} low-confidence citations."
                    result = {"message": msg, "count": len(low), "citations": low, "success": True}
                    
                    if not args.json:
                        out = [msg, ""]
                        for i, c in enumerate(low):
                            sim = int(c['best_match']['similarity']*100) if c.get('best_match') else 0
                            loc = f"{c['location']['chapter']} / {c['location']['paragraph']}"
                            out.append(f"{i+1}. @{c['key']} ({sim}%)")
                            out.append(f"   Location: {loc}")
                            if c.get("warning"):
                                out.append(f"   ⚠️  {c['warning']}")
                            out.append("")
                        result = out
                except Exception as e:
                    raise RuntimeError(f"Failed to analyze audit file: {e}")

            else:
                parser.print_help()
                sys.exit(1)

        elif args.command == "zotero":
            sub = getattr(args, "zotero_command", None)
            
            if sub == "setup":
                env_path = Path(__file__).parent / ".env"
                if env_path.exists():
                     result = {"message": f"Global credentials found at {env_path}", "env_path": str(env_path), "success": True}
                else:
                     msg = (
                         f"Missing .env file at {env_path}.\n"
                         "Please create it with:\n"
                         "ZOTERO_USER_ID=your_id\n"
                         "ZOTERO_API_KEY=your_key"
                     )
                     if args.json:
                         print_output({"message": msg, "success": False}, json_mode=True)
                         sys.exit(1)
                     else:
                         print(msg)
                         sys.exit(1)

            elif sub == "list_collections":
                colls = manager.zotero_service.list_collections()
                if not colls:
                     result = {"message": "No collections found or authentication failed.", "collections": []}
                else:
                     result = {"message": f"Found {len(colls)} collections.", "collections": colls}
                     if not args.json:
                         print("Available Collections:")
                         for c in colls:
                             print(f" - [{c['id']}] {c['name']}")
                         result = None

            elif sub == "set_collection":
                 root = resolve_root(args)
                 config = manager._load_config(root)
                 config["ZOTERO_COLLECTION_ID"] = args.collection_id
                 manager._save_config(root, config)
                 result = {"message": f"Project collection set to {args.collection_id}", "success": True}

            elif sub == "sync":
                root = resolve_root(args)
                res = manager.sync_zotero(root)
                result = res
                
            elif sub == "check":
                root = resolve_root(args)
                issues = manager.validate_citations(root)
                if issues:
                    result = {"message": "Citation check found issues.", "issues": issues, "status": "issues_found"}
                else:
                    result = {"message": "Citation check passed. All citations found in bibliography.", "issues": [], "status": "passed"}
            
            else:
                parser.print_help()
                sys.exit(1)

        elif args.command == "research":
            sub = getattr(args, "research_command", None)
            root = resolve_root(args)
            
            if sub == "search":
                results = manager.research_service.search(args.query, mode=args.mode, limit=args.limit)
                
                # In deep mode, results are ResearchPaper dicts with local_path set
                if args.mode == "deep":
                    downloaded_files = []
                    citations_added = 0
                    
                    for p in results:
                        if p.get("local_path"):
                            downloaded_files.append(Path(p["local_path"]))
                        
                        if p.get("citation_key"):
                            citations_added += 1
                            
                    msg = f"Deep Research completed. Acquired {len(downloaded_files)} PDFs and added {citations_added} citations."
                    
                    # TRIGGER RAG SYNC
                    if downloaded_files:
                        manager.rag_service.sync_research(root, downloaded_files)
                        msg += " Auto-synced with Research RAG."
                        
                    result = {"message": msg, "papers": results, "success": True}
                else:
                    # Discovery mode
                    msg = f"Found {len(results)} papers via ArXiv/PubMed."
                    if not args.json:
                        # Print table
                        from rich.console import Console
                        from rich.table import Table
                        console = Console()
                        table = Table(title=f"Research Results: {args.query}")
                        table.add_column("Year", style="cyan", no_wrap=True)
                        table.add_column("Title", style="magenta")
                        table.add_column("Authors", style="green")
                        table.add_column("Source", style="blue")
                        
                        for p in results:
                            authors = ", ".join(p["authors"])[:30] + "..." if len(str(p["authors"])) > 30 else ", ".join(p["authors"])
                            table.add_row(p["year"], p["title"][:60]+"...", authors, p["source"])
                        
                        console.print(table)
                        result = None
                    else:
                        result = {"message": msg, "papers": results}
            elif sub == "download":
                query = args.query.strip()
                
                # Determine if DOI or title
                doi = None
                title = None
                if query.startswith("10.") and "/" in query:
                    doi = query
                else:
                    title = query
                
                # Get metadata from Crossref
                paper = manager.research_service.get_paper_metadata_crossref(doi=doi, title=title)
                
                if not paper:
                    result = {"message": f"Could not find paper matching '{query}' via Crossref.", "success": False}
                else:
                    # Attempt to download
                    manager.research_service._acquire_content(paper)
                    
                    if paper.local_path:
                        # Add to bib if not present
                        bib_path = root / "references.bib"
                        bib_entry = paper.to_bibtex()
                        
                        # Check if key already exists
                        try:
                            existing = bib_path.read_text(encoding='utf-8') if bib_path.exists() else ""
                            if paper.citation_key not in existing:
                                with open(bib_path, "a", encoding="utf-8") as f:
                                    f.write("\n\n" + bib_entry)
                        except Exception:
                            pass
                        
                        # Sync with RAG
                        try:
                            manager.rag_service.sync_research(root, [Path(paper.local_path)])
                        except Exception:
                            pass
                        
                        result = {
                            "message": f"Downloaded '{paper.title}' to {paper.local_path}",
                            "path": paper.local_path,
                            "citation_key": paper.citation_key,
                            "success": True
                        }
                    else:
                        result = {
                            "message": f"Found paper '{paper.title}' but could not download PDF (abstract saved as .txt)",
                            "path": paper.local_path,
                            "success": False
                        }
            else:
                parser.print_help()
                sys.exit(1)

        elif args.command == "clean_bib":
            from lib.bib_cleaner import BibCleaner
            root = resolve_root(args)
            cleaner = BibCleaner(root)
            cleaner.clean()
            result = {"message": "Bibliography cleaning complete.", "success": True}

        elif args.command == "add":
            from lib.utils import bibtex_escape
            root = resolve_root(args)
            query = args.query.strip()
            
            # Smart determination of DOI vs Title
            doi = None
            title = None
            if query.startswith("10.") and "/" in query:
                doi = query
            else:
                title = query
                
            paper = manager.research_service.get_paper_metadata_crossref(doi=doi, title=title)
            
            if paper:
                 # The paper.to_bibtex() now uses bibtex_escape internally.
                 bib_entry = paper.to_bibtex()
                 bib_path = root / "references.bib"
                 
                 # Append to file
                 with open(bib_path, "a", encoding="utf-8") as f:
                     f.write("\n\n" + bib_entry)
                     
                 msg = f"Added reference '{paper.title}' (@{paper.citation_key}) to bibliography."
                 # Enhancement: explicitly mention if it was a title match to alert user to possible mismatch
                 if not doi:
                     msg += f" (Matched via title search: '{query}')"
                     
                 result = {"message": msg, "success": True, "citation_key": paper.citation_key, "title": paper.title}
            else:
                 msg = f"Could not find any paper matching '{query}' via Crossref."
                 result = {"message": msg, "success": False}

        elif args.command == "bib":
            root = resolve_root(args)
            sub = getattr(args, "bib_command", None)
            
            if sub == "patch":
                manager.load_project(root)
                if manager.bib_service.patch_reference(args.key, args.field, args.value):
                     result = {"message": f"Successfully patched {args.key}: {args.field} = {args.value}", "success": True}
                else:
                     raise RuntimeError(f"Failed to patch {args.key}. Key not found or file error.")
            elif sub == "check_files":
                manager.load_project(root)
                issues = manager.bib_service.check_files(root)
                if issues:
                    result = {"message": f"Found {len(issues)} broken file links in bibliography.", "issues": issues, "success": False}
                    if not args.json:
                        out = [result["message"]]
                        for iss in issues:
                            out.append(f"❌ [@{iss['key']}] Missing: {iss['file']}")
                        result = out
                else:
                    result = {"message": "All bibliography file links are valid.", "issues": [], "success": True}
            elif sub == "dedupe":
                manager.load_project(root)
                stats = manager.bib_service.deduplicate()
                result = stats
                if not args.json and stats.get("success"):
                    out = [f"Deduplication complete. {stats['initial_count']} -> {stats['final_count']} entries."]
                    for r in stats.get("removed", []):
                        out.append(f" - Removed: {r}")
                    result = out
            else:
                parser.print_help()
                sys.exit(1)

        elif args.command == "test":
            print("Running ThesisFlow Manager Tests...")
            loader = unittest.TestLoader()
            start_dir = Path(__file__).parent / "tests"
            suite = loader.discover(start_dir)
            runner = unittest.TextTestRunner(verbosity=2)
            res = runner.run(suite)
            sys.exit(0 if res.wasSuccessful() else 1)

        # ========== REFACTORING PRIMITIVES HANDLERS ==========
        
        elif args.command == "read":
            root = resolve_root(args)
            if hasattr(args, 'para') and args.para:
                # Read specific paragraph
                data = manager.read_paragraph(root, args.chapter, args.para)
                result = {"message": f"Read paragraph '{data['paragraph_title']}'", **data}
            else:
                # Read entire chapter
                data = manager.read_chapter(root, args.chapter)
                result = {"message": f"Read chapter '{data['chapter_title']}' ({data['paragraph_count']} paragraphs)", **data}

        elif args.command == "merge":
            root = resolve_root(args)
            keep = getattr(args, 'keep_originals', False)
            new_path = manager.merge_paragraphs(
                root, args.chapter, 
                args.para_a, args.para_b, 
                args.into, keep_originals=keep
            )
            result = {"message": f"Merged into '{new_path.name}'", "path": str(new_path), "success": True}

        elif args.command == "move":
            root = resolve_root(args)
            new_path = manager.move_paragraph(
                root, args.para,
                args.from_chapter, args.to_chapter
            )
            result = {"message": f"Moved to '{new_path.name}' in chapter '{args.to_chapter}'", "path": str(new_path), "success": True}

        elif args.command == "rename":
            root = resolve_root(args)
            if args.para:
                # Rename paragraph
                if not args.chapter:
                    raise ValueError("--chapter is required when renaming a paragraph")
                new_path = manager.rename_paragraph(root, args.chapter, args.old_name, args.new_name)
                result = {"message": f"Renamed paragraph to '{new_path.name}'", "path": str(new_path), "success": True}
            else:
                # Rename chapter
                new_path = manager.rename_chapter(root, args.old_name, args.new_name)
                result = {"message": f"Renamed chapter to '{new_path.name}'", "path": str(new_path), "success": True}

        elif args.command == "delete":
            root = resolve_root(args)
            if args.para:
                # Delete paragraph
                if not args.chapter:
                    raise ValueError("--chapter is required when deleting a paragraph")
                deleted = manager.delete_paragraph(root, args.chapter, args.target)
                result = {"message": f"Deleted paragraph '{deleted}'", "deleted": deleted, "success": True}
            else:
                # Delete chapter
                force = getattr(args, 'force', False)
                deleted = manager.delete_chapter(root, args.target, force=force)
                result = {"message": f"Deleted chapter '{deleted}'", "deleted": deleted, "success": True}

        elif args.command == "insert":
            root = resolve_root(args)
            after = getattr(args, 'after', None)
            new_path = manager.insert_paragraph(
                root, args.chapter, args.title,
                content=args.content,
                after_para_id=after,
                include_header=not args.raw
            )
            if after:
                result = {"message": f"Inserted '{new_path.name}' after '{after}'", "path": str(new_path), "success": True}
            else:
                result = {"message": f"Inserted '{new_path.name}' at beginning", "path": str(new_path), "success": True}
            
        else:
            parser.print_help()
            sys.exit(1)

        if result:
            print_output(result, args.json)

    except Exception as e:
        err = {"status": "error", "message": str(e), "type": type(e).__name__}
        if args.json:
            print(json.dumps(err))
        else:
            print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
