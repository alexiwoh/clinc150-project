"""The pip compatibility route must be portable and agree with the uv lock."""

from __future__ import annotations

import re
import tomllib

from src.constants import PROJECT_ROOT

_PINNED_REQUIREMENT = re.compile(r"([A-Za-z0-9_-]+)==([^; ]+)\s*(?:;.*)?")


def test_secondary_requirements_are_portable_and_locked() -> None:
    exported = (PROJECT_ROOT / "requirements.txt").read_text()
    entries = [line.strip() for line in exported.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    assert entries.count("-e .") == 1, "The pip route must install this repository"
    lock = tomllib.loads((PROJECT_ROOT / "uv.lock").read_text())
    locked_versions = {(package["name"], package["version"]) for package in lock["package"]}
    for entry in entries:
        if entry == "-e .":
            continue
        match = _PINNED_REQUIREMENT.fullmatch(entry)
        assert match is not None, f"Unpinned or nonportable dependency: {entry}"
        name, version = match.groups()
        assert (name.lower().replace("_", "-"), version) in locked_versions
    assert "/Users/" not in exported and "file://" not in exported
