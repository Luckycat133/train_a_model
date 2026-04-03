import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest


# Fixtures used by the test suite.  The original project shipped helper
# functions inside the test modules to create temporary environments.  The
# fixtures below simply call those helpers in lightweight "dry run" mode so the
# unit tests can execute without requiring large external resources.


@pytest.fixture
def test_files():
    from test.test_tokenizer import setup_test_environment, cleanup_test_environment

    test_dir, paths = setup_test_environment(dry_run=True)
    try:
        yield paths
    finally:
        cleanup_test_environment(test_dir)


@pytest.fixture
def test_dir():
    from test.test_train_model import setup_test_environment, cleanup_test_environment

    directory = setup_test_environment()
    try:
        yield directory
    finally:
        cleanup_test_environment(directory)
