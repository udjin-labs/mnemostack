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


def test_docs_do_not_advertise_file_keystore_revocation_for_every_backend():
    """The claim this guards is an operational one — how fast a revoked key
    stops working — and it belongs where an operator reads it.

    It was pinned to the CURRENT release's notes when the correction
    shipped in 2.2.0, which made every later release repeat a 2.2.0
    correction verbatim or fail. Release notes are a historical record:
    2.3.0 restating a fix it did not make would be false. So the guard now
    holds the two places the claim actually lives — the deployment guide an
    operator reads, and the changelog entry that recorded the correction —
    neither of which a future release can quietly drop.
    """
    deployment = Path("docs/deployment.md").read_text()
    assert "takes effect on the caller's next call" in deployment
    assert "MNEMOSTACK_OPENBAO_CACHE_TTL" in deployment
    # ...and specifically that the two backends are distinguished, rather
    # than the file keystore's semantics being claimed for both.
    assert "the OpenBao adapter's positive cache instead bounds" in deployment

    # Scoped to the entry that RECORDED the correction, not the whole file:
    # the constant appears in older entries too, so a whole-file search
    # would keep passing while 2.2.0's own note quietly lost the
    # distinction — the very regression this guards.
    changelog = Path("CHANGELOG.md").read_text()
    recorded_in = re.search(
        r"^## \[2\.2\.0\].*?(?=^## \[)", changelog, re.MULTILINE | re.DOTALL
    )
    assert recorded_in is not None, "the 2.2.0 entry is where this correction lives"
    entry = recorded_in.group(0)
    assert "file-keystore revocation takes effect on the caller's next call" in entry
    assert "MNEMOSTACK_OPENBAO_CACHE_TTL" in entry
