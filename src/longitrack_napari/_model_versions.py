from __future__ import annotations

import re

from longitrack_backend.model import CHECKPOINT_NAME, DEFAULT_REVISION, REQUIRED_FILES, available_versions

# the same trailing number the backend matches an '@<version>' selector against
_VERSION = re.compile(r"(\d+(?:\.\d+)*)\s*$")
_SELECTOR = re.compile(r"@\d+(?:\.\d+)*$")


def version_number(name: str) -> str | None:
    match = _VERSION.search(name)
    return match.group(1) if match else None


def split_version(location: str) -> tuple[str, str | None]:
    match = _SELECTOR.search(location)
    return (location[: match.start()], match.group(0)[1:]) if match else (location, None)


def with_version(location: str, version: str | None) -> str:
    return location if version is None else f"{split_version(location)[0]}@{version}"


def _newest_first(names) -> list[str]:
    numbered = [name for name in names if version_number(name) is not None]
    return sorted(numbered, key=lambda name: tuple(int(part) for part in version_number(name).split(".")), reverse=True)


def _hub_versions(repo_id: str) -> list[str]:
    from huggingface_hub import HfApi

    files = set(HfApi().list_repo_files(repo_id, revision=DEFAULT_REVISION))
    fold = re.compile(rf"^fold_(\d+|all)/{re.escape(CHECKPOINT_NAME)}$")
    folders: dict[str, set[str]] = {}
    for file in files:
        if "/" in file:
            top, rest = file.split("/", 1)
            folders.setdefault(top, set()).add(rest)
    return [
        name
        for name, contents in folders.items()
        if all(required in contents for required in REQUIRED_FILES) and any(fold.match(item) for item in contents)
    ]


def list_model_versions(location: str, local: bool) -> list[str]:
    """Versioned model folders under a local folder or Hub repo, newest first."""
    location = split_version(location.strip())[0]
    return _newest_first(available_versions(location) if local else _hub_versions(location))
