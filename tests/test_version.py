"""Version metadata consistency tests."""

import re
from pathlib import Path

import mnemostack


def test_package_version_matches_project_metadata():
    pyproject = Path("pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)

    assert match is not None
    assert mnemostack.__version__ == match.group(1)


def test_current_version_has_dated_changelog_entry():
    changelog = Path("CHANGELOG.md").read_text()
    version_heading = re.search(
        rf"^## \[{re.escape(mnemostack.__version__)}\] - \d{{4}}-\d{{2}}-\d{{2}}$",
        changelog,
        re.MULTILINE,
    )

    assert version_heading is not None
