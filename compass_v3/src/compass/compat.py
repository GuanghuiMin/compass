"""Windows compatibility shims for AppWorld.

AppWorld's ``timeout_call`` uses ``signal.SIGALRM``, which does not exist on
Windows. ``appworld.environment`` imports the name directly, so the patch must
target that module attribute. The replacement runs the cell in a daemon thread
and raises the same message shape on timeout. A timed-out cell keeps running in
the background (threads cannot be killed) -- AppWorld's own 1000-request guard
bounds runaway API loops, and the environment is discarded after each rollout.
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Any, Callable


def _thread_timeout_call(function: Callable[..., Any], timeout_seconds: int | None = None,
                         *args: Any, **kwargs: Any) -> Any:
    if timeout_seconds is None:
        return function(*args, **kwargs)
    timeout_seconds = int(timeout_seconds)
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["result"] = function(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - re-raised in caller thread
            box["error"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout_seconds)
    if t.is_alive():
        raise Exception(
            f"Function {function.__name__} execution timed out after {timeout_seconds} seconds.")
    if "error" in box:
        raise box["error"]
    return box.get("result")


def install() -> None:
    # child processes (ProcessPoolExecutor) must open AppWorld's log files as UTF-8:
    # agents occasionally emit non-GBK characters (emoji) in code, which crashes
    # a locale-encoded write on Windows
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if not sys.platform.startswith("win"):
        return
    import appworld.environment as env_mod
    import appworld.common.utils as utils_mod
    env_mod.timeout_call = _thread_timeout_call
    utils_mod.timeout_call = _thread_timeout_call
