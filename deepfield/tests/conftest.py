"""Commandline option and fixture definitions"""
import pytest

def pytest_addoption(parser):
    """Commandline option for assigning benchmarks path."""
    parser.addoption("--path_to_benchmarks",
                        action="store",
                        default="./benchmarks")

@pytest.fixture(scope="module")
def path_to_benchmarks(request):
    """Returns benchmarks path"""
    return request.config.getoption("--path_to_benchmarks")
