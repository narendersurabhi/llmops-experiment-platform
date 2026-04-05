from __future__ import annotations

import subprocess
from pathlib import Path

import components.ui.file_dialogs as file_dialogs
from components.ui.file_dialogs import (
    REPO_ROOT,
    build_macos_choose_directory_script,
    build_macos_choose_file_script,
    resolve_initial_dialog_dir,
    run_osascript,
)


def test_resolve_initial_dialog_dir_defaults_to_repo_root():
    assert resolve_initial_dialog_dir(None) == str(REPO_ROOT)


def test_resolve_initial_dialog_dir_uses_existing_file_parent(tmp_path: Path):
    file_path = tmp_path / "config.json"
    file_path.write_text("{}", encoding="utf-8")

    assert resolve_initial_dialog_dir(str(file_path)) == str(tmp_path)


def test_resolve_initial_dialog_dir_uses_existing_directory(tmp_path: Path):
    assert resolve_initial_dialog_dir(str(tmp_path)) == str(tmp_path)


def test_build_macos_choose_file_script_contains_prompt_and_initial_dir():
    lines = build_macos_choose_file_script(title='Pick "config"', initial_dir="/tmp/demo")

    assert 'choose file with prompt "Pick \\"config\\""' in lines[1]
    assert 'POSIX file "/tmp/demo"' in lines[0]


def test_build_macos_choose_directory_script_contains_prompt_and_initial_dir():
    lines = build_macos_choose_directory_script(title="Pick output", initial_dir="/tmp/out")

    assert 'choose folder with prompt "Pick output"' in lines[1]
    assert 'POSIX file "/tmp/out"' in lines[0]


def test_run_osascript_returns_trimmed_stdout(monkeypatch):
    def fake_run(args, capture_output, text):
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="/tmp/demo\n", stderr="")

    monkeypatch.setattr(file_dialogs.subprocess, "run", fake_run)

    assert run_osascript(["return \"/tmp/demo\""]) == "/tmp/demo"


def test_run_osascript_returns_none_on_user_cancel(monkeypatch):
    def fake_run(args, capture_output, text):
        return subprocess.CompletedProcess(
            args=args,
            returncode=1,
            stdout="",
            stderr="execution error: User canceled. (-128)",
        )

    monkeypatch.setattr(file_dialogs.subprocess, "run", fake_run)

    assert run_osascript(["return \"ignored\""]) is None
