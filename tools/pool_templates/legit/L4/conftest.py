def pytest_addoption(parser):
    parser.addoption("--rows", type=int, default=16)
