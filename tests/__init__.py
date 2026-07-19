import doctest
import os
import unittest

import wsds
from wsds import ws_dataset, ws_shard, ws_sink, ws_audio, audio_codec

from . import test_audio_seek

_README = os.path.join(os.path.dirname(__file__), os.pardir, "README.md")


def load_tests(loader, tests, ignore):
    tests.addTests(doctest.DocTestSuite(wsds))
    tests.addTests(doctest.DocTestSuite(ws_dataset))
    tests.addTests(doctest.DocTestSuite(ws_shard))
    tests.addTests(doctest.DocTestSuite(ws_sink))
    tests.addTests(doctest.DocTestSuite(ws_audio))
    tests.addTests(doctest.DocTestSuite(audio_codec))
    tests.addTests(doctest.DocFileSuite(_README, module_relative=False))
    tests.addTests(loader.loadTestsFromModule(test_audio_seek))
    return tests
