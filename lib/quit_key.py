"""Non-blocking single-key 'quit' watcher for recording scripts.

Usage:
    qk = QuitKey()            # puts stdin in cbreak; restores on exit
    while running:
        if qk.pressed():
            break
        ...

If stdin is not a TTY (e.g. when launched by record_session.py with
stdin=DEVNULL), the watcher silently disables itself and pressed()
always returns False.
"""
from __future__ import annotations

import atexit
import select
import sys
import termios
import tty


class QuitKey:
    def __init__(self, key: str = "q"):
        self.key = key.lower()
        self._fd = None
        self._old = None
        try:
            if sys.stdin.isatty():
                self._fd = sys.stdin.fileno()
                self._old = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
                atexit.register(self.restore)
        except (termios.error, ValueError, OSError):
            self._fd = None
            self._old = None

    @property
    def enabled(self) -> bool:
        return self._fd is not None

    def pressed(self) -> bool:
        if self._fd is None:
            return False
        r, _, _ = select.select([self._fd], [], [], 0)
        if not r:
            return False
        try:
            ch = sys.stdin.read(1)
        except (OSError, ValueError):
            return False
        return ch and ch.lower() == self.key

    def restore(self) -> None:
        if self._fd is not None and self._old is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)
            except (termios.error, OSError):
                pass
        self._fd = None
        self._old = None
