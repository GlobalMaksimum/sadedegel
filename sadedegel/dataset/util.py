from loguru import logger
import json


def safe_read(file: str):
    try:
        with open(file) as fp:
            return fp.read()
    except:
        logger.exception(f"Error in reading {file}")
        raise


def safe_json_load(file: str):
    try:
        return json.loads(safe_read(file))
    except:
        logger.exception(f"JSON Decoding error in {file}")
        raise
