"""This is a test module for the package."""
from respiratory_sound import __version__


def test_version() -> None:
    """This test is a reminder for a version change."""
    assert __version__ == "0.1.0"
