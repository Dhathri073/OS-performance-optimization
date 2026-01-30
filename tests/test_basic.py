"""
Unit tests for the OS Performance Optimization project.
"""

import unittest
from src.collectors.data_collector import collect_data
from src.database.db_manager import init_db, load_data

class TestDataCollector(unittest.TestCase):
    def test_collect_data(self):
        df = collect_data(duration=1, interval=0.5)
        self.assertIsNotNone(df)
        self.assertGreater(len(df), 0)
        self.assertIn('cpu_percent', df.columns)

class TestDatabase(unittest.TestCase):
    def test_init_db(self):
        init_db()  # Should not raise exception
        df = load_data(limit=1)
        self.assertIsNotNone(df)

if __name__ == '__main__':
    unittest.main()