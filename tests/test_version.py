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


def test_release_notes_state_openbao_revocation_latency():
    """Do not advertise file-keystore revocation semantics for every backend."""
    changelog = Path("CHANGELOG.md").read_text()
    release_match = re.search(
        rf"^## \[{re.escape(mnemostack.__version__)}\].*?(?=^## \[)",
        changelog,
        re.MULTILINE | re.DOTALL,
    )

    assert release_match is not None
    current_release = release_match.group(0)
    assert "file-keystore revocation takes effect on the caller's next call" in current_release
    assert "MNEMOSTACK_OPENBAO_CACHE_TTL" in current_release
