import sys
from unittest.mock import MagicMock
import unittest

# Mock out heavy libraries before importing test files
heavy_modules = [
    'torch',
    'transformers',
    'datasets',
    'peft',
    'huggingface_hub',
    'evaluate',
    'sentence_transformers',
    'wandb',
    'trl'
]

for mod in heavy_modules:
    sys.modules[mod] = MagicMock()

# Now discover and run tests
if __name__ == '__main__':
    tests = unittest.defaultTestLoader.discover('tests')
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    sys.exit(not result.wasSuccessful())
