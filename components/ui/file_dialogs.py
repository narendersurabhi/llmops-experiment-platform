from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_initial_dialog_dir(current_value: str | None) -> str:
    if not current_value:
        return str(REPO_ROOT)

    candidate = Path(current_value).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    candidate = candidate.resolve()

    if candidate.exists():
        return str(candidate if candidate.is_dir() else candidate.parent)

    parent = candidate.parent
    if parent.exists():
        return str(parent)
    return str(REPO_ROOT)


def apple_script_quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def build_macos_choose_file_script(*, title: str, initial_dir: str) -> list[str]:
    escaped_title = apple_script_quote(title)
    escaped_dir = apple_script_quote(initial_dir)
    return [
        f'set startDir to POSIX file "{escaped_dir}"',
        f'set chosenItem to choose file with prompt "{escaped_title}" default location startDir',
        "POSIX path of chosenItem",
    ]


def build_macos_choose_directory_script(*, title: str, initial_dir: str) -> list[str]:
    escaped_title = apple_script_quote(title)
    escaped_dir = apple_script_quote(initial_dir)
    return [
        f'set startDir to POSIX file "{escaped_dir}"',
        f'set chosenItem to choose folder with prompt "{escaped_title}" default location startDir',
        "POSIX path of chosenItem",
    ]


def run_osascript(lines: list[str]) -> str | None:
    args = ["osascript"]
    for line in lines:
        args.extend(["-e", line])

    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode == 0:
        output = result.stdout.strip()
        return output or None

    combined_error = f"{result.stdout}\n{result.stderr}".strip()
    if "User canceled" in combined_error or "(-128)" in combined_error:
        return None
    raise RuntimeError(f"macOS file dialog failed: {combined_error}")


def _create_tk_dialog_root():
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    return root


def choose_file_path(*, title: str, current_value: str | None = None) -> str | None:
    initial_dir = resolve_initial_dialog_dir(current_value)
    if sys.platform == "darwin":
        return run_osascript(build_macos_choose_file_script(title=title, initial_dir=initial_dir))

    from tkinter import filedialog

    root = _create_tk_dialog_root()
    try:
        selected = filedialog.askopenfilename(title=title, initialdir=initial_dir)
        return selected or None
    finally:
        root.destroy()


def choose_directory_path(*, title: str, current_value: str | None = None) -> str | None:
    initial_dir = resolve_initial_dialog_dir(current_value)
    if sys.platform == "darwin":
        return run_osascript(build_macos_choose_directory_script(title=title, initial_dir=initial_dir))

    from tkinter import filedialog

    root = _create_tk_dialog_root()
    try:
        selected = filedialog.askdirectory(title=title, initialdir=initial_dir)
        return selected or None
    finally:
        root.destroy()
