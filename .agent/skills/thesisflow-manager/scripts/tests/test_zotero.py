import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from lib.zotero_service import ZoteroService

class TestZoteroIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_root = Path("/tmp/mock_skill")
        self.service = ZoteroService(self.mock_root)
        
    @patch("lib.zotero_service.zotero.Zotero")
    @patch("lib.zotero_service.os.getenv")
    def test_auth_success(self, mock_getenv, mock_zotero):
        mock_getenv.side_effect = lambda k: "123" if k == "Zotero_USER_ID" else "key"
        self.service.user_id = "123"
        self.service.api_key = "key"
        
        self.assertTrue(self.service.authenticate())
        mock_zotero.assert_called_with("123", "user", "key")

    @patch("lib.zotero_service.zotero.Zotero")
    def test_list_collections(self, mock_zotero):
        # Setup mock
        mock_instance = MagicMock()
        mock_zotero.return_value = mock_instance
        mock_instance.collections.return_value = [
            {"data": {"key": "A1", "name": "Thesis"}},
            {"data": {"key": "B2", "name": "Research"}}
        ]
        
        self.service.user_id = "123"
        self.service.api_key = "key"
        
        colls = self.service.list_collections()
        self.assertEqual(len(colls), 2)
        self.assertEqual(colls[0]["id"], "A1")
        self.assertEqual(colls[0]["name"], "Thesis")

if __name__ == "__main__":
    unittest.main()
