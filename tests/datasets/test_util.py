import pytest
from json.decoder import JSONDecodeError
from .context import util


def test_safe_file_open_failed():
    with pytest.warns(UserWarning, match=r'Error in reading \w+$'), pytest.raises(FileNotFoundError):
        _ = util.safe_read("NoSuchFile")


@pytest.fixture(scope="session")
def corrupted_json(tmpdir_factory):
    fn = tmpdir_factory.mktemp("data").join("CorruptedJson.json")

    with open(fn, 'w') as wp:
        print('["name:"Jack"]', file=wp)

    return fn


def test_safe_json_load_failed(corrupted_json):
    with pytest.warns(UserWarning, match=fr'JSON Decoding error for {corrupted_json}$'), pytest.raises(
            JSONDecodeError):
        _ = util.safe_json_load(corrupted_json)
