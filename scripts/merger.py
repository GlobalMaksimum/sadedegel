from pathlib import Path
from typing import Iterable
from os.path import basename, splitext
import sys
from difflib import context_diff
import click
import numpy as np
from sadedegel.dataset._core import safe_json_load
from sadedegel.dataset import load_sentence_corpus, file_paths


def file_diff(i1: Iterable, i2: Iterable):
    l1, l2 = list(i1), list(i2)

    if len(l1) != len(l2):
        click.secho(f"Iterable sizes are not equal {len(l1)} != {len(l2)}")

    s1, s2 = set(l1), set(l2)

    if len(s1) != len(s2):
        click.secho(f"Set sizes are not equal {len(s1)} != {len(s2)}")

    for e1 in list(s1):
        if e1 not in s2:
            click.secho(f"{e1} in I1 but not in I2")

    for e2 in list(s2):
        if e2 not in s1:
            click.secho(f"{e2} in I2 but not in I1")


@click.command()
def cli():
    sents = load_sentence_corpus(False)

    fns = [splitext(basename(fp))[0] for fp in file_paths()]

    reference = dict((fn, sent['sentences']) for fn, sent in zip(fns, sents))

    for fn in fns:
        anno_path = Path('sadedegel/work/Labeled') / f"{fn}_labeled.json"

        if anno_path.exists():
            anno = safe_json_load(anno_path)

            anno_sents = [s['content'] for s in anno['sentences']]
            _ = np.array([s['deletedInRound'] for s in anno['sentences']])

            refe_sents = reference[fn]

            if refe_sents != anno_sents:
                click.secho(f"Mismatch in number of sentences for document {fn}", fg="red")

                diff = context_diff(refe_sents, anno_sents)

                click.secho('\n'.join(diff), fg="red")

                sys.exit(1)
            else:
                click.secho(f"MATCH: {fn}", fg="green")

        else:
            click.secho(f"Annotated corpus member {anno_path} not found.", fg="red")


if __name__ == '__main__':
    cli()
