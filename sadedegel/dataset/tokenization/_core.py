from os.path import expanduser, getsize
from pathlib import Path
import glob
import click
import numpy as np
import time

from sadedegel.config import tokenizer_context
from sadedegel import Doc
from tabulate import tabulate


__general_download_message__ = """Ensure that you have properly downloaded extended or tokenization corpus using

            python -m sadedegel.dataset.extended download --access-key xxx --secret-key xxxx
            python -m sadedegel.dataset.tokenization download --access-key xxx --secret-key xxxx

        Unfortunately due to data licensing issues we could not share data publicly. 
        Get in touch with sadedegel team to obtain a download key.

        """

__tokenization_download_message__ = """Ensure that you have properly downloaded tokenization corpus using

            python -m sadedegel.dataset.tokenization download --access-key xxx --secret-key xxxx

        Unfortunately due to data licensing issues we could not share data publicly. 
        Get in touch with sadedegel team to obtain a download key.

        """

__tokenization_version_message__ = "Ensure your dataset is in versioned format."


def dot_progress(i, length, t0):
    n = np.ceil(length * 0.05)

    if i == length - 1:
        click.echo(f". {(time.time() - t0):.2f} sec", nl=True, color="yellow")
    elif i % n == 0 and i != 0:
        click.echo(".", nl=False, color="yellow")


def check_directory_structure(path: str, version) -> bool:
    if not Path(expanduser(path)).exists():
        click.secho(f"{path} not found.\n", fg="red")
        click.secho(__general_download_message__, fg="red")

        return False

    elif not (Path(expanduser(path)) / "tokenization").exists():
        click.secho(f"Tokenization Dataset not found.\n", fg="red")
        click.secho(__tokenization_download_message__, fg="red")

        return False

    elif not(Path(expanduser(path)) / "tokenization" / version).exists():
        click.secho(f"Stated version is not found.\n", fg="red")
        click.secho(__tokenization_version_message__, fg="red")

        return False

    elif not (Path(expanduser(path)) / "tokenization" / version / "raw").exists():
        click.secho(f"Tokenization Raw Dataset not found.\n", fg="red")

        return False

    elif not (Path(expanduser(path)) / "tokenization" / version / "tokenized").exists():
        click.secho(f"Tokenization Tokenized Dataset not found.\n", fg="red")

        return False

    else:
        return True


def raw_stats(data_home: str, version) -> int:
    sz = 0
    for f in glob.glob(str((Path(expanduser(data_home)) / "tokenization" / version / "raw" / "*.txt").absolute())):
        sz += getsize(f)
    return sz


def tokenized_stats(data_home: str, version) -> int:
    sz = 0
    for f in glob.glob(str((Path(expanduser(data_home)) / "tokenization" / version / "tokenized" / "*.txt").absolute())):
        sz += getsize(f)
    return sz


def check_and_display(data_home: str, version='v2'):
    if check_directory_structure(data_home, version):
        return dict(byte=dict(raw=raw_stats(data_home, version) / 1e6,
                              tokens=tokenized_stats(data_home, version) / 1e6))


def tok_eval(raw_docs, tokenized_docs, tokenizers=['simple', 'bert']):
    """Evaluate tokenizers on a downsized version of tokenization dataset.

    :param raw_docs: Raw documents of tokenization corpus.
    :type raw_docs: dict
    :param tokenized_docs: Tokenized documents of tokenization corpus.
    :type tokenized_docs: dict
    :param tokenizers: List of tokenizer names to evaluate. Defaults to ["simple", "bert"].
    :type tokenizers: List[str]

    :return: IoU score between list of true tokens and list of tokenized tokens.
    """
    scores = {}
    for tokenizer in tokenizers:
        t0 = time.time()
        with tokenizer_context(tokenizer):
            click.echo("Word Tokenizer: " + click.style(f"{tokenizer}", fg="blue"), nl=False)

            ious = []
            for i, (raw_doc, tokenized_doc) in enumerate(zip(raw_docs, tokenized_docs)):

                true_tokens = set(tokenized_doc['tokens'])

                d = Doc(raw_doc['text'])
                tokens = [token for sentence in d for token in sentence.tokens]

                if tokenizer == 'bert':
                    fixed_hashtags = []
                    for tok in tokens:
                        if "##" not in tok:
                            fixed_hashtags.append(tok)
                        else:
                            merged = fixed_hashtags[-1] + tok.split("##")[1]
                            fixed_hashtags.pop(-1)
                            fixed_hashtags.append(merged)
                    tokens = fixed_hashtags

                pred_tokens = set(tokens)

                union = len(true_tokens.union(pred_tokens))
                intersection = len(true_tokens.intersection(pred_tokens))

                ious.append(intersection/union)

                dot_progress(i, len(raw_docs), t0)
            scores[tokenizer] = np.mean(ious)

    table = [[item[0], item[1]] for item in scores.items()]

    return table
