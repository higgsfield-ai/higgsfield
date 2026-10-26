import re
from pathlib import Path

import os

from typing import Optional, Tuple
from urllib.parse import urlsplit

regex = re.compile("^[a-zA-Z_][a-zA-Z0-9_]*$")


def check_name(name: str):
    if len(name) < 1 or len(name) > 20:
        raise ValueError("Name must be between 1 and 20 characters long")

    if not regex.match(name):
        raise ValueError("Name must match regex ^[a-zA-Z_][a-zA-Z0-9_]*$")

    return name


def wd_path() -> Path:
    return Path.cwd()


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def templates_path() -> Path:
    return Path(ROOT_DIR) / "static" / "templates"

_REPOSITORY_PATH_PATTERN = re.compile(
    r"([a-zA-Z0-9_-]+)/([a-zA-Z0-9_-]+)\.git"
)
_HOST_LABEL_PATTERN = re.compile(
    r"[a-zA-Z0-9](?:[a-zA-Z0-9-]*[a-zA-Z0-9])?"
)


def _has_url_whitespace_or_controls(link: str) -> bool:
    return re.search(r"[\x00-\x20\x7f]", link) is not None


def _is_github_host(host: str) -> bool:
    labels = host.lower().split(".")
    if len(labels) < 2 or labels[0] != "github":
        return False

    return all(_HOST_LABEL_PATTERN.fullmatch(label) for label in labels)


def _parse_https_repo_link(link: str) -> Optional[Tuple[str, str, str]]:
    if _has_url_whitespace_or_controls(link):
        return None

    try:
        parsed = urlsplit(link)
        host = parsed.hostname
        port = parsed.port
    except ValueError:
        return None

    if (
        parsed.scheme != "https"
        or host is None
        or not _is_github_host(host)
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
    ):
        return None

    if not parsed.path.startswith("/"):
        return None

    match = _REPOSITORY_PATH_PATTERN.fullmatch(parsed.path[1:])
    if match is None:
        return None

    return host.lower(), match.group(1), match.group(2)


def _parse_ssh_repo_link(link: str) -> Optional[Tuple[str, str, str]]:
    if _has_url_whitespace_or_controls(link):
        return None

    match = re.fullmatch(r"git@([^/:?#]+):(.+)", link)
    if match is None:
        return None

    host, path = match.groups()
    if not _is_github_host(host):
        return None

    repository_match = _REPOSITORY_PATH_PATTERN.fullmatch(path)
    if repository_match is None:
        return None

    return host.lower(), repository_match.group(1), repository_match.group(2)


def match_https_link(link: str) -> bool:
    return _parse_https_repo_link(link) is not None


def convert_https_to_ssh(link: str) -> str:
    parsed = _parse_https_repo_link(link)
    if parsed is None:
        raise ValueError("link must be a valid GitHub HTTPS repository URL")

    host, owner, repository = parsed
    return f"git@{host}:{owner}/{repository}.git"


def parse_origin_link_or_else(link: str) -> Optional[str]:
    https_repository = _parse_https_repo_link(link)
    if https_repository is not None:
        return convert_https_to_ssh(link)

    ssh_repository = _parse_ssh_repo_link(link)
    if ssh_repository is not None:
        host, owner, repository = ssh_repository
        return f"git@{host}:{owner}/{repository}.git"

    return None
