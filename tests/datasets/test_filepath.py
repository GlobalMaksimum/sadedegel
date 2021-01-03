import pytest
from pathlib import Path
from .context import file_paths, CorpusTypeEnum


def test_corpustype_enum():
    with pytest.raises(ValueError):
        file_paths("my_flumsy_corpus_type")


def test_corpus_equality():
    fp_raw = file_paths(CorpusTypeEnum.RAW, use_basename=True, noext=True)
    fp_sent = file_paths(CorpusTypeEnum.SENTENCE, use_basename=True, noext=True)

    assert fp_raw == fp_sent


@pytest.mark.parametrize("corpus_type", [CorpusTypeEnum.RAW, CorpusTypeEnum.SENTENCE, CorpusTypeEnum.ANNOTATED])
def test_corpus_equality(corpus_type):
    files = file_paths(corpus_type, use_basename=True, noext=False)

    assert all((Path(file).name == file for file in files))
