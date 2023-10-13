import asyncio
from pathlib import Path
from typing import List

from jinja2 import Environment, FileSystemLoader
from higgsfield.internal.cfg import AppConfig
import asyncssh

from higgsfield.internal.util import templates_path
from higgsfield.internal.experiment.builder import header


docker_install_script = '''/bin/bash -c "$(curl -fsSL https://gist.githubusercontent.com/arpanetus/1c1210b9e432a04dcfb494725a407a70/raw/5d47baa19b7100261a2368a43ace610528e0dfa2/install.sh)"'''
invoker_install_script = """wget https://github.com/ml-doom/invoker/releases/download/latest/invoker-latest-linux-amd64.tar.gz && \
        tar -xvf invoker-latest-linux-amd64.tar.gz && \
        sudo mv invoker /usr/bin/invoker && \
        rm invoker-latest-linux-amd64.tar.gz"""


def deploy_key_script(key: str, project_name: str, deploy_key_string: str):
    return f"""sudo mkdir -p ~/.ssh && \
    echo "{key}" > ~/.ssh/{project_name}-github-deploy.key && \
    chmod 600 ~/.ssh/{project_name}-github-deploy.key && \
    sudo touch ~/.ssh/config && \
    sudo chmod 644 ~/.ssh/config && \
    echo "{deploy_key_string}" | sudo tee -a ~/.ssh/config
    """


class Setup:
    app_config: AppConfig
    path: str
    deploy_key: str
    project_path: Path

    def __init__(
        self,
        app_config: AppConfig,
        project_path: Path,
    ):
        self.app_config = app_config

        if reason := self.app_config.is_valid() is not None:
            raise ValueError(reason)

        self.project_path = project_path

    def create_ssh_key_file(self):
        if self.app_config.key is None:
            raise ValueError("SSH_KEY in env is None")

        with Path.home() / ".ssh" / f"temp-{self.app_config.name}.key" as f:
            f.write_text(self.app_config.key)
            f.chmod(0o600)
            self.path = str(Path.resolve(f.absolute()))

    def finish(self):
        Path(self.path).unlink()

    async def establish_connections(self):
        if self.app_config.key is None:
            raise ValueError("SSH_KEY in env is None")

        self.connections: List[asyncssh.SSHClientConnection] = []
        for host in self.app_config.hosts:
            self.connections.append(
                await asyncssh.connect(
                    host,
                    port=self.app_config.port,
                    username=self.app_config.user,
                    client_keys=[self.path],
                )
            )

    def set_deploy_key(self):
        with Path.home() / ".ssh" / "higgsfield" / f"{self.app_config.name}-github-deploy.key" as f:
            self.deploy_key = f.read_text()

    def _build_deploy_key_string(self):
        return f"Host github.com-{self.app_config.name}\n\tHostName github.com\n\tIdentityFile ~/.ssh/{self.app_config.name}-github-deploy.key\n\tIdentitiesOnly yes\n\tStrictHostKeyChecking no\n\tUserKnownHostsFile=/dev/null\n\tLogLevel=ERROR\n"

    async def setup_nodes(self):
        await self.establish_connections()

        if len(self.connections) == 0:
            print("\n\n\nNO CONNECTIONS!!!\n\n\n")
        
        print("\n\n\nINSTALLING DOCKER\n\n\n")
        async def printer(thing):
            print(thing)
        to_run = []
        for conn in self.connections:
            to_run.append(printer(await conn.run(docker_install_script)))

        await asyncio.gather(*to_run)

        print("\n\n\nINSTALLING INVOKER\n\n\n")
        to_run = []
        for conn in self.connections:
            to_run.append(printer(await conn.run(invoker_install_script)))

        await asyncio.gather(*to_run)

        print("\n\n\nSETTING UP DEPLOY KEY\n\n\n")
        self.set_deploy_key()
        dk_script = deploy_key_script(
            self.deploy_key, self.app_config.name, self._build_deploy_key_string()
        )
        to_run = []
        for conn in self.connections:
            to_run.append(printer(await conn.run(dk_script)))
    
        await asyncio.gather(*to_run)
        
        print("\n\n\nPULLING BASE DOCKER IMAGE\n\n\n")
        to_run = []
        for conn in self.connections:
            to_run.append(printer(await conn.run(f"docker pull higgsfield/pytorch:latest")))

        await asyncio.gather(*to_run)


        print("\n\n\nSeems like everything is done by now. Go run your experiments.\n\n\n")

