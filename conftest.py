# Empty conftest so pytest puts the project root on sys.path.
# Also registers the --run-exercises flag for tests/test_exercises.py.

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-exercises",
        action="store_true",
        default=False,
        help="run the exercise tests (they skip by default)",
    )
