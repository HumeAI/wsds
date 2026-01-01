import unittest
import doctest
from wsds import ws_dataset, ws_shard, ws_sink

def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(ws_dataset))
    tests.addTests(doctest.DocTestSuite(ws_shard))
    tests.addTests(doctest.DocTestSuite(ws_sink))
    return tests

if __name__ == '__main__':
    unittest.main()
