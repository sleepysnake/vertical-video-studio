"""Register bundled Qt DLL directories before importing PySide6 extensions."""
from __future__ import annotations

import os
from pathlib import Path
import sys


if sys.platform == "win32" and getattr(sys, "frozen", False):
    bundled = Path(sys._MEIPASS)  # type: ignore[attr-defined]
    handles = []
    for directory in (bundled, bundled / "PySide6", bundled / "shiboken6"):
        if directory.is_dir():
            handles.append(os.add_dll_directory(str(directory)))
    # add_dll_directory removes the path when its handle is collected.
    sys._vvs_dll_directory_handles = handles
