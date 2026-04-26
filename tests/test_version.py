"""Version metadata consistency tests."""

import re
from pathlib import Path

import mnemostack


def test_package_version_matches_project_metadata():
    pyproject = Path("pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)

    assert match is not None
    assert mnemostack.__version__ == match.group(1)
