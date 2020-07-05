import click
from .core import load


@click.command
def cli():
    pass


@cli.command()
@click.argument('doc')
def summarize(doc):
    summarizer = load()

    summary = summarizer(doc)

    click.echo(summary)


if __name__ == '__main__':
    summarize()  # pylint: disable=E1120
