from typing import Tuple
import click
import requests
from .about import __version__, __herokuapp_url__
from .config import load_config, show_config


def to_tuple(version_string: str) -> Tuple:
    return tuple(version_string.split('.'))


def to_str(version_tuple: Tuple) -> str:
    return ".".join(version_tuple)


@click.group(help="SadedeGel commandline")
def cli():
    pass


@cli.command()
@click.option("--section", "-s", default=None, help="Filter by section")
def config(section):
    cfg = load_config()

    show_config(cfg, section)


@cli.command()
def info():
    """SadedeGel version information in details"""
    most_recent_version = requests.get("https://pypi.python.org/pypi/sadedegel/json").json()['info']['version']
    most_recent_version = to_tuple(most_recent_version)

    click.echo(f"sadedeGel (@PyPI): {click.style(to_str(most_recent_version), fg='green')}")

    installed_version = to_tuple(__version__)

    if installed_version < most_recent_version:
        color = "yellow"
    else:
        color = "green"

    click.echo(f"sadedeGel (installed): {click.style(to_str(installed_version), fg=color)}")

    heroku_version = requests.get(f"{__herokuapp_url__}/api/info").json()['version']
    heroku_version = to_tuple(heroku_version)

    if heroku_version < most_recent_version:
        color = "yellow"
    else:
        color = "green"

    click.echo(f"sadedeGel Server ({__herokuapp_url__}): {click.style(to_str(heroku_version), fg=color)}")


if __name__ == '__main__':
    cli()  # pylint: disable=E1120
