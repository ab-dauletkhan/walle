"""
Lifecycle Manager for WALL-E.

Pattern: Command — each shutdown hook is a callable registered at startup.
Hooks execute in reverse order (LIFO) so that dependencies shut down before
the resources they depend on.
"""

import logging
from typing import Callable

_log = logging.getLogger("walle.lifecycle")


class LifecycleManager:
    """Manages orderly startup registration and reverse-order shutdown."""

    def __init__(self):
        self._hooks: list[tuple[str, Callable[[], None]]] = []
        self._done = False

    def register(self, label: str, hook: Callable[[], None]) -> None:
        """Register a shutdown hook. Last registered = first to execute."""
        self._hooks.append((label, hook))

    def shutdown(self) -> None:
        """Execute all hooks in reverse registration order. Idempotent."""
        if self._done:
            return
        self._done = True
        _log.info("Shutting down WALL-E...")
        for label, hook in reversed(self._hooks):
            try:
                _log.debug("Shutdown: %s", label)
                hook()
            except Exception as e:
                _log.warning("Shutdown '%s' failed: %s", label, e)
        _log.info("WALL-E offline. Goodbye!")
