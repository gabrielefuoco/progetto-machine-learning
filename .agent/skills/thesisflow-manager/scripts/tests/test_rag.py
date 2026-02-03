import unittest
import shutil
import tempfile
import os
from pathlib import Path
from lib.project import ProjectManager
from lib.rag_service import RAGService

class TestRAG(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.manager = ProjectManager(self.test_dir)
        self.project_name = "RAGTestProject"
        self.proj_path = self.manager.init_project(self.project_name)
        
        # Create dummy content
        self.chap_path = self.manager.add_chapter(self.proj_path, "Intro")
        self.para_path = self.manager.add_paragraph(self.chap_path, "Welcome", "This is a test paragraph about transformers.")
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_sync_draft(self):
        # Sync
        success = self.manager.rag_service.sync_draft(self.proj_path, self.manager.load_project(self.proj_path).chapters)
        self.assertTrue(success)
        
        # Check files
        rag_dir = self.proj_path / ".thesisflow" / "rag" / "draft"
        self.assertTrue((rag_dir / "index.faiss").exists())
        self.assertTrue((rag_dir / "metadata.json").exists())

    def test_query_draft(self):
        self.manager.rag_service.sync_draft(self.proj_path, self.manager.load_project(self.proj_path).chapters)
        
        # Query
        results = self.manager.rag_service.query(self.proj_path, "draft", "transformers")
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["title"], "01_Welcome")
        self.assertIn("Intro", results[0]["chapter"])

    def test_sync_research_missing_path(self):
        # Should return false or raise error if list is empty
        success = self.manager.rag_service.sync_research(self.proj_path, [])
        self.assertFalse(success)

    def test_sync_research_multi_format(self):
        # Create a dummy txt and md file
        txt_file = self.test_dir / "research.txt"
        with open(txt_file, "w", encoding="utf-8") as f:
            f.write("This is a research document about attention mechanisms.")
        
        md_file = self.test_dir / "notes.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# Notes\n\nBERT is a bidirectional model.")
        
        success = self.manager.rag_service.sync_research(self.proj_path, [txt_file, md_file])
        self.assertTrue(success)
        
        # Query
        results = self.manager.rag_service.query(self.proj_path, "research", "attention")
        self.assertGreater(len(results), 0)
        self.assertIn("research.txt", results[0]["path"])

if __name__ == '__main__':
    unittest.main()
