import unittest
import shutil
import tempfile
import time
from pathlib import Path
from lib.project import ProjectManager

class TestAgentConfig(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.manager = ProjectManager(self.test_dir)
        self.project_name = "AgentTestProject"

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_agent_files_creation(self):
        """Verify that init_project creates the expected agent config files."""
        proj_path = self.manager.init_project(self.project_name)
        
        # Check Rules
        rule_path = proj_path / ".agent" / "rules" / "thesisflow.md"
        self.assertTrue(rule_path.exists(), "Rule file thesisflow.md not created")
        content = rule_path.read_text(encoding="utf-8")
        self.assertIn("trigger: always_on", content)
        self.assertIn("SKILL.md", content)
        self.assertIn("Infer Language", content)

        # Check Workflows
        workflows_dir = proj_path / ".agent" / "workflows"
        self.assertTrue((workflows_dir / "draft.md").exists())
        self.assertTrue((workflows_dir / "research.md").exists())
        self.assertTrue((workflows_dir / "production.md").exists())

    def test_no_overwrite(self):
        """Verify that existing agent config files are NOT overwritten."""
        proj_path = self.manager.init_project(self.project_name)
        
        # Modify a workflow file
        draft_path = proj_path / ".agent" / "workflows" / "draft.md"
        original_content = draft_path.read_text(encoding="utf-8")
        
        # Wait a bit to ensure potential timestamp diffs (though not checking that)
        # Just write new content
        new_content = "MODIFIED CONTENT"
        draft_path.write_text(new_content, encoding="utf-8")
        
        # Re-run logic (cannot run init_project effectively as it raises FileExistsError for the project itself)
        # We need to simulate the state where project exists but we want to re-apply agent config?
        # Actually init_project raises FileExistsError if project dir exists (line 106).
        # So we can effectively test the *internal method* _setup_agent_config directly,
        # or we accept that init prevents overwrite of the whole project anyway.
        
        # But wait, success criteria said "when the agent initializes the project".
        # If I can't re-init, I can't overwrite.
        # But if the user manually deleted .agent folder but kept the project?
        # A more robust test:
        
        # Call _setup_agent_config directly on existing project
        self.manager._setup_agent_config(proj_path)
        
        # Check if content matches NEW content
        self.assertEqual(draft_path.read_text(encoding="utf-8"), new_content, "Existing file was overwritten!")

if __name__ == '__main__':
    unittest.main()
