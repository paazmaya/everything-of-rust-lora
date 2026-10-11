"""Tests package root for lib."""

import lib


def test_package_exports() -> None:
    """Verify lib exports all symbols declared in __all__."""
    assert hasattr(lib, "__all__")
    assert isinstance(lib.__all__, list)
    assert len(lib.__all__) > 0

    for symbol_name in lib.__all__:
        assert hasattr(lib, symbol_name), f"lib is missing symbol: {symbol_name}"
        assert getattr(lib, symbol_name) is not None
