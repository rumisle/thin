"""The runtime's error classes; the HIP module loads only these."""

from __future__ import annotations

__all__ = ["Collision", "DeviceFault", "LaunchError", "NotEmulated", "Refused"]


class LaunchError(ValueError):
    """A refusal of a launch, a binding or a device call."""


class Refused(LaunchError):
    """A refusal a kernel's own host raises."""


class DeviceFault(RuntimeError):
    """A fault while a kernel runs; ``cause`` says which."""

    def __init__(self, message: str, cause: str = "fault") -> None:
        super().__init__(message)
        self.cause = cause
        """``fault``, ``bounds:<buffer>``, ``kept:<argument>`` or the emulator's own."""


class NotEmulated(RuntimeError):
    """What the emulator does not model; never the kernel's fault."""


class Collision(Exception):
    """A store address holding other bytes; deliberately not a ValueError."""
