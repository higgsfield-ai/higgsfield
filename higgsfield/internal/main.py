import click

from higgsfield.internal.cli import (
    init_cmd,
    ci,
    run_experiment,
    show_deploy_key,
    build_experiments,
    ci_cli,
)


@click.group()
def cli():
    """Higgsfield CLI"""
    pass


cli.add_command(init_cmd)
cli.add_command(ci)
cli.add_command(run_experiment)
cli.add_command(build_experiments)
cli.add_command(show_deploy_key)
cli.add_command(ci_cli.setup_nodes)
