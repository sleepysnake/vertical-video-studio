"""Put the release in a fresh top-level folder when extracted beside older builds."""
from __future__ import annotations

from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile


source = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from studio.version import __version__
top_level = sys.argv[3] if len(sys.argv) > 3 else f"VerticalVideoStudio-{__version__}"
if not top_level.startswith("VerticalVideoStudio"):
    top_level = "VerticalVideoStudio"
if Path(top_level).name != top_level or "/" in top_level or "\\" in top_level:
    raise ValueError("ZIP 顶层目录需要单个名称")
with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            archive.write(path, Path(top_level) / path.relative_to(source))
print(destination)
