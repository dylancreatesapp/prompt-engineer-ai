import unittest
from prompt_engineer_ai import build_refiner

class TestBasics(unittest.TestCase):
    def test_build(self):
        ref = build_refiner()
        self.assertTrue(ref is not None)

if __name__ == '__main__':
    unittest.main()
