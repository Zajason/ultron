"""OS-specific actions behind one interface, selected once at startup."""

from __future__ import annotations

import sys

from .base import PlatformActions


def get_platform() -> PlatformActions:
    if sys.platform == "darwin":
        from .macos import MacOSActions

        return MacOSActions()
    raise NotImplementedError(f"no platform adapter for {sys.platform} yet")


__all__ = ["PlatformActions", "get_platform"]
