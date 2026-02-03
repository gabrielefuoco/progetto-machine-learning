import unittest
import shutil
import tempfile
from pathlib import Path
from lib.project import ProjectManager
from lib.stats import StatsAnalyzer

class TestThesisFlow(unittest.TestCase):
    def setUp(self):
        # Create a temp directory
        self.test_dir = Path(tempfile.mkdtemp())
        self.manager = ProjectManager(self.test_dir)
        self.project_name = "TestProject"

    def tearDown(self):
        # Cleanup
        shutil.rmtree(self.test_dir)

    def test_init_project(self):
        path = self.manager.init_project(self.project_name)
        self.assertTrue(path.exists())
        self.assertTrue((path / "thesisflow.json").exists())
        self.assertTrue((path / "references.bib").exists())
        self.assertTrue((path / "assets").exists())

    def test_structure_creation(self):
        proj_path = self.manager.init_project(self.project_name)
        
        # Add Chapter
        chap_path = self.manager.add_chapter(proj_path, "Introduction")
        self.assertTrue(chap_path.exists())
        self.assertIn("01_Introduction", chap_path.name)
        
        # Add Paragraph
        para_path = self.manager.add_paragraph(chap_path, "Background", "Some content here.")
        self.assertTrue(para_path.exists())
        self.assertIn("01_Background.md", para_path.name)
        
        content = para_path.read_text(encoding="utf-8")
        self.assertIn("## Background", content)
        self.assertIn("Some content here", content)

    def test_stats(self):
        proj_path = self.manager.init_project(self.project_name)
        chap_path = self.manager.add_chapter(proj_path, "Intro")
        self.manager.add_paragraph(chap_path, "P1", "word " * 100) # 100 words
        
        project = self.manager.load_project(proj_path)
        analyzer = StatsAnalyzer(project)
        
        # 10 words in title (## P1) counts as 2 + 100 = 102 roughly? 
        # Actually logic is split(): "## P1\n\nword word..." 
        # ##, P1, word... (100) -> 102 words
        
        count = analyzer.count_words(project.chapters[0].paragraphs[0].content)
        self.assertTrue(count >= 100)

if __name__ == '__main__':
    unittest.main()
