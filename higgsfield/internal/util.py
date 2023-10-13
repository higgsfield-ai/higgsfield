import re
from pathlib import Path

import os

from typing import Optional

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

https_repo_url_pattern = re.compile(
    r"^https\:\/\/github\.com\/[a-zA-Z0-9\-\_]+\/[a-zA-Z0-9\-\_]+\.git$"
)


def match_https_link(link: str) -> bool:
    return https_repo_url_pattern.match(link) is not None


def convert_https_to_ssh(link: str) -> str:
    gh, user, repo = link[8:-4].split("/")
    return f"git@{gh}:{user}/{repo}.git"


def parse_origin_link_or_else(link: str) -> Optional[str]:
    if match_https_link(link):
        return convert_https_to_ssh(link)
    if link.startswith("git@github.com:"):
        return link

    return None
