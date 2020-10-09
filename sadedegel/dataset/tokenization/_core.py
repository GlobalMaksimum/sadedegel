from os.path import expanduser, getsize
from pathlib import Path
import glob
import click

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


def check_directory_structure(path: str) -> bool:
    if not Path(expanduser(path)).exists():
        click.secho(f"{path} not found.\n", fg="red")
        click.secho(__general_download_message__, fg="red")

        return False

    elif not (Path(expanduser(path)) / "tokenization").exists():
        click.secho(f"Tokenization Dataset not found.\n", fg="red")
        click.secho(__tokenization_download_message__, fg="red")

        return False

    elif not (Path(expanduser(path)) / "tokenization" / "raw").exists():
        click.secho(f"Tokenization Raw Dataset not found.\n", fg="red")

        return False

    elif not (Path(expanduser(path)) / "tokenization" / "tokenized").exists():
        click.secho(f"Tokenization Tokenized Dataset not found.\n", fg="red")

        return False

    else:
        return True


def raw_stats(data_home: str) -> int:
    sz = 0
    for f in glob.glob(str((Path(expanduser(data_home)) / "tokenization" / "raw" / "*.txt").absolute())):
        sz += getsize(f)
    return sz


def tokenized_stats(data_home: str) -> int:
    sz = 0
    for f in glob.glob(str((Path(expanduser(data_home)) / "tokenization" / "tokenized" / "*.txt").absolute())):
        sz += getsize(f)
    return sz


def check_and_display(data_home: str):
    if check_directory_structure(data_home):
        return dict(byte=dict(raw=f"{raw_stats(data_home) / 1e6} MB",
                              tokens=f"{tokenized_stats(data_home) / 1e6} MB"))
