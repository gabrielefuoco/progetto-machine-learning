import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np
import faiss
from lib.rag_service import RAGService

class TestRedundancy(unittest.TestCase):
    def setUp(self):
        self.rag = RAGService()
        self.project_root = Path("/tmp/test_project")

    @patch("lib.rag_service.RAGService.load_index")
    def test_detect_redundancies_fixed(self, mock_load):
        # Setup: 3 vectors, 2 are identical
        dim = 384
        index = faiss.IndexFlatL2(dim)
        
        v1 = np.random.rand(dim).astype('float32')
        v2 = v1.copy() # Redundant with v1
        v3 = np.random.rand(dim).astype('float32') # Unique
        
        index.add(np.stack([v1, v2, v3]))
        
        metadata = [
            {"chapter": "Chap 1", "title": "Para 1", "path": "p1.md"},
            {"chapter": "Chap 1", "title": "Para 2", "path": "p2.md"},
            {"chapter": "Chap 2", "title": "Para 3", "path": "p3.md"}
        ]
        
        mock_load.return_value = (index, metadata)
        
        # Run
        res = self.rag.detect_redundancies(self.project_root, threshold=0.1)
        
        # Verify
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["para_a"]["title"], "Para 1")
        self.assertEqual(res[0]["para_b"]["title"], "Para 2")
        self.assertLess(res[0]["distance"], 0.001)

    @patch("lib.rag_service.RAGService.load_index")
    def test_detect_redundancies_adaptive(self, mock_load):
        # Setup: 4 vectors, one pair is closer than others
        dim = 384
        index = faiss.IndexFlatL2(dim)
        
        # Vectors far apart
        v1 = np.zeros(dim).astype('float32')
        v1[0] = 1.0
        v2 = np.zeros(dim).astype('float32')
        v2[1] = 1.0
        v3 = np.zeros(dim).astype('float32')
        v3[2] = 1.0
        
        # v4 is close to v1 (dist approx 0.1)
        v4 = v1.copy()
        v4[10] = 0.3 # Small perturbation
        
        index.add(np.stack([v1, v2, v3, v4]))
        
        metadata = [
            {"chapter": "C1", "title": "P1", "path": "p1.md"},
            {"chapter": "C1", "title": "P2", "path": "p2.md"},
            {"chapter": "C1", "title": "P3", "path": "p3.md"},
            {"chapter": "C1", "title": "P4", "path": "p4.md"}
        ]
        
        mock_load.return_value = (index, metadata)
        
        # Run adaptive with high threshold (so only adaptive kicks in)
        res = self.rag.detect_redundancies(self.project_root, threshold=10.0, adaptive=True, multiplier=1.0)
        
        # Distances:
        # dist(1,4) is small
        # others are large (approx sqrt(2))
        
        # The adaptive threshold should flag the P1-P4 pair
        self.assertTrue(any(r["para_a"]["title"] == "P1" and r["para_b"]["title"] == "P4" for r in res))

if __name__ == "__main__":
    unittest.main()
