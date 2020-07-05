from loguru import logger
import pytest


@pytest.fixture(scope='function')
def example_fixture():
    logger.info("Setting Up Example Fixture...")
    yield
    logger.info("Tearing Down Example Fixture...")
