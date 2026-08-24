"""Force UTF-8 as the process-wide default text encoding (Japanese Windows cp932 fix).

``torch.compile`` → ``torch._inductor`` reads kernel templates (``.py.jinja``)
and source files through the builtin ``open()`` with the *locale* default
encoding. On Japanese Windows the locale is cp932, so reading any UTF-8 file
raises::

    UnicodeDecodeError: 'cp932' codec can't decode byte 0x94 in position ...:
    illegal multibyte sequence

which is fatal inside ``torch.compile`` (e.g. ``torch/_inductor/utils.py``
``load_template``, reached via ``load_kernel_template`` / ``load_flex_template``).

This module replicates Python's ``PYTHONUTF8=1`` (UTF-8 mode) behaviour at
runtime, which is otherwise impossible to enable after interpreter startup:

1. set ``PYTHONUTF8`` / ``PYTHONIOENCODING`` env vars (helps subprocesses);
2. reconfigure stdin/stdout/stderr to UTF-8 (avoids ``UnicodeEncodeError``
   when torch/logging print non-ASCII to a cp932 console);
3. patch ``io.open`` / ``builtins.open`` so a text-mode ``open(path)`` with no
   explicit ``encoding`` defaults to UTF-8 instead of the locale.

Idempotent — safe to import from multiple entry points (``prestartup_script.py``
and the node ``__init__.py``).
"""
from __future__ import annotations

import builtins
import io
import os
import sys

_MARKER = "_hswq_utf8_patched"


def _apply() -> None:
    # Idempotency: if we (or another copy of this module) already patched,
    # builtins.open carries our marker.
    if getattr(builtins.open, _MARKER, False):
        return

    # 1) Environment — picked up by subprocesses and late readers.
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    # 2) Stdio streams → UTF-8.
    for _name in ("stdin", "stdout", "stderr"):
        _stream = getattr(sys, _name, None)
        if _stream is None:
            continue
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    # 3) Default text encoding for open()/io.open()/pathlib → UTF-8.
    _orig_open = io.open

    def _utf8_open(
        file,
        mode="r",
        buffering=-1,
        encoding=None,
        errors=None,
        newline=None,
        closefd=True,
        opener=None,
    ):
        # Only change the *default* (encoding=None) for text mode on a path.
        # File objects / descriptors / binary mode / explicit encoding pass
        # through untouched — strictly additive, no behaviour regression.
        if (
            encoding is None
            and isinstance(mode, str)
            and "b" not in mode
            and isinstance(file, (str, bytes, os.PathLike))
        ):
            encoding = "utf-8"
        return _orig_open(file, mode, buffering, encoding, errors, newline, closefd, opener)

    _utf8_open.__name__ = "open"
    _utf8_open.__qualname__ = "open"
    _utf8_open.__doc__ = _orig_open.__doc__
    _utf8_open.__module__ = "io"
    _utf8_open.__wrapped__ = _orig_open
    setattr(_utf8_open, _MARKER, True)

    io.open = _utf8_open
    builtins.open = _utf8_open


_apply()
