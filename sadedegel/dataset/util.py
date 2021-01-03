import warnings
import json


def safe_read(file: str):
    try:
        with open(file) as fp:
            return fp.read()
    except:
        warnings.warn(f"Error in reading {file}", UserWarning)
        raise


def safe_json_load(file: str):
    try:
        return json.loads(safe_read(file))
    except:
        warnings.warn(f"JSON Decoding error for {file}", UserWarning)
        raise
