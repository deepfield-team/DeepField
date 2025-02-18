"""Commandline option and fixture definitions"""
import pytest

def pytest_addoption(parser):
    """Commandline option for assigning tNavigator tutorials path."""
    parser.addoption("--path_to_tnav_tutorials",
                        action="store",
                        default="./tNav_tutorials")

@pytest.fixture(scope="module")
def path_to_tnav_tutorials(request):
    """Returns tNavigator tutorials path"""
    return request.config.getoption("--path_to_tnav_tutorials")
