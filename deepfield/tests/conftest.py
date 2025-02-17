import pytest

def pytest_addoption(parser):
    parser.addoption("--path_to_tnav_tutorials", 
                        action="store", 
                        default="./tNav_tutorials")

@pytest.fixture(scope="module")
def path_to_tnav_tutorials(request):
    return request.config.getoption("--path_to_tnav_tutorials")