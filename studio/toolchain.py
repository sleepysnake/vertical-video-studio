"""Discover and verify the two local media executables."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
from .edits import no_window


def _registry_paths() -> list[str]:
    if sys.platform != "win32":
        return []
    import winreg

    paths = []
    locations = ((winreg.HKEY_CURRENT_USER, r"Environment"),
                 (winreg.HKEY_LOCAL_MACHINE,
                  r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"))
    for hive, key in locations:
        try:
            with winreg.OpenKey(hive, key) as registry:
                value, _ = winreg.QueryValueEx(registry, "Path")
                paths.extend(os.path.expandvars(part.strip(' "')) for part in value.split(os.pathsep))
        except OSError:
            continue
    return [path for path in paths if path]


def _candidates(name: str, configured: str, sibling: str = "") -> list[str]:
    candidates = [configured.strip(), sibling]
    found = shutil.which(name)
    if found:
        candidates.append(found)
    if sys.platform == "win32":
        for directory in _registry_paths():
            candidates.append(str(Path(directory) / (name + ".exe")))
    return list(dict.fromkeys(value for value in candidates if value))


def resolve_tool(name: str, configured: str = "", sibling: str = "") -> tuple[str | None, str]:
    """Return a working executable and a concise diagnostic."""
    if name not in ("ffmpeg", "ffprobe"):
        raise ValueError("未知的媒体工具")
    errors = []
    for candidate in _candidates(name, configured, sibling):
        path = str(Path(candidate).resolve()) if Path(candidate).is_file() else shutil.which(candidate)
        if not path:
            errors.append(f"{candidate} 不存在")
            continue
        # Windows may display a compatibility prompt while CreateProcess opens
        # a text file named .exe, before subprocess's timeout can take effect.
        if sys.platform == "win32" and Path(path).suffix.lower() == ".exe":
            try:
                with Path(path).open("rb") as executable:
                    if executable.read(2) != b"MZ":
                        errors.append(f"{path} 不是可用的 {name}")
                        continue
            except OSError:
                errors.append(f"{path} 无法读取")
                continue
        try:
            result = subprocess.run([path, "-version"], capture_output=True, text=True,
                                    timeout=5, check=False, creationflags=no_window())
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{path} 无法运行：{exc.__class__.__name__}")
            continue
        first = result.stdout.splitlines()[0] if result.stdout else ""
        if result.returncode == 0 and first.lower().startswith(name + " version "):
            return path, first
        errors.append(f"{path} 不是可用的 {name}")
    return None, errors[0] if errors else f"未找到 {name}"


def detect_tools(ffmpeg: str = "", ffprobe: str = "") -> tuple[tuple[str | None, str], tuple[str | None, str]]:
    encoder = resolve_tool("ffmpeg", ffmpeg)
    sibling = str(Path(encoder[0]).with_name("ffprobe.exe" if sys.platform == "win32" else "ffprobe")) if encoder[0] else ""
    probe = resolve_tool("ffprobe", ffprobe, sibling)
    return encoder, probe
