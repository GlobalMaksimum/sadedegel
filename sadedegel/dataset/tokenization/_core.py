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

__tokenization_version_message__ = "Ensure your dataset is in versioned format."


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
