import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if call.excinfo and call.excinfo.errisinstance(FileNotFoundError) \
            and "golden-set" in str(call.excinfo.value):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), 0, "Skipped: private golden set not in public repo")
