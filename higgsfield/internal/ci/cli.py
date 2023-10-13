import asyncio
import json

import click

from higgsfield.internal.util import wd_path, parse_origin_link_or_else
from higgsfield.internal.cfg import AppConfig
from .setup import Setup

from base64 import b64decode


@click.command("get-hosts")
def hosts():
    wd = wd_path()
    app_config = AppConfig.from_path(wd)
    click.echo(",".join(app_config.hosts))


@click.command("get-nproc-per-node")
def proc_per_node():
    wd = wd_path()
    app_config = AppConfig.from_path(wd)
    click.echo(str(app_config.number_of_processes_per_node))


@click.command("get-ssh-details")
def ssh_details():
    wd = wd_path()
    app_config = AppConfig.from_path(wd)
    print(
        json.dumps(
            {
                "key": app_config.key,
                "user": app_config.user,
                "port": app_config.port,
                "hosts": ",".join(app_config.hosts),
            },
            indent=2,
        )
    )


@click.command("decode-secrets")
@click.argument("env", type=str, required=True)
def decode_secrets(env: str):
    env_path = wd_path() / "env"
    if env_path.exists():
        raise ValueError("env file already exists")

    env_path.write_text(b64decode(env.encode()).decode())




@click.command("setup-nodes")
def setup_nodes():
    wd = wd_path()
    app_config = AppConfig.from_path(wd)

    project_path = wd

    origin_url = app_config.get_git_origin_url(project_path)

    if origin_url is None:
        raise ValueError("Have you pushed your project to github?")

    origin_url = parse_origin_link_or_else(origin_url)

    if origin_url is None:
        raise ValueError("Please use ssh or https url for github repo.")

    app_config.github_repo_url = origin_url

    app_config.set_git_origin_url(project_path)

    setup = Setup(app_config, project_path)

    try:
        setup.create_ssh_key_file()
        asyncio.run(setup.setup_nodes())
    finally:
        setup.finish()
