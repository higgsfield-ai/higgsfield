from pathlib import Path
import os
import io

from cryptography.hazmat.primitives import serialization, asymmetric
import paramiko
from jinja2 import Environment, FileSystemLoader
from .util import templates_path, ROOT_DIR

from typing import Tuple

templates = Environment(loader=FileSystemLoader(templates_path()))


def init(wd: Path, project_name: str):
    if project_name is None or project_name == "":
        raise ValueError("Project name cannot be empty")
    project_path = wd / project_name
    if project_path.exists():
        raise ValueError(f"Project {project_name} already exists")

    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / "src").mkdir(parents=True, exist_ok=True)

    config_path = project_path / "src" / "config.py"
    if config_path.exists():
        raise ValueError(f"Config file {config_path} already exists")

    readme_path = project_path / "README.md"
    if readme_path.exists():
        raise ValueError(f"README.md file {readme_path} already exists")

    source_path = Path(ROOT_DIR) / "static" / "project"
    fileset = [
        ".gitignore",
        "env",
        "Dockerfile",
        "src/alpaca_bf16.py",
        "src/alpaca_fp16.py",
        "src/dataset.py",
        "requirements.txt",
    ]

    for file in fileset:
        (project_path / file).write_bytes((source_path / file).read_bytes())

    config_path.write_text(
        templates.get_template("config_py.j2").render(project_name=project_name)
    )

    readme_path.write_text(
        templates.get_template("README_md.j2").render(project_name=project_name)
    )

    hf_deploy_ssh_folder = Path.home() / ".ssh/higgsfield/"
    hf_deploy_ssh_folder.mkdir(parents=True, exist_ok=True)
    priv, pub, _ = generate_deploy_keys()
    (hf_deploy_ssh_folder / f"{project_name}-github-deploy.key").write_bytes(priv)

    # set permissions to 400
    os.system(f"chmod 600 {hf_deploy_ssh_folder / f'{project_name}-github-deploy.key'}")
    public_key_path = hf_deploy_ssh_folder / f"{project_name}-github-deploy.key.pub"
    public_key_path.write_text(pub.decode() + "\n")

    # set permissions to 400
    os.system(
        f"chmod 600 {hf_deploy_ssh_folder / f'{project_name}-github-deploy.key.pub'}"
    )


def generate_deploy_keys() -> Tuple[bytes, bytes, paramiko.Ed25519Key]:
    c_ed25519key = asymmetric.ed25519.Ed25519PrivateKey.generate()  # type: ignore
    privpem = c_ed25519key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.OpenSSH,
        encryption_algorithm=serialization.NoEncryption(),
    )
    priv_obj = io.StringIO(privpem.decode())
    p_ed25519key = paramiko.Ed25519Key.from_private_key(priv_obj)

    pub = c_ed25519key.public_key()
    openssh_pub = pub.public_bytes(
        encoding=serialization.Encoding.OpenSSH,
        format=serialization.PublicFormat.OpenSSH,
    )

    return privpem, openssh_pub, p_ed25519key
