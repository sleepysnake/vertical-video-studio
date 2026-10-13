"""Create a clean source archive without venv, bytecode or build artifacts."""
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile
from studio.version import __version__


project = Path(__file__).resolve().parent
destination = (Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else
               project / "release" / f"VerticalVideoStudio-{__version__}-Source.zip")
destination.parent.mkdir(parents=True, exist_ok=True)
files = [project / name for name in (
    "app.py", "qt_app.py", "VerticalVideoStudio.spec", "requirements.txt", "requirements-build.txt", "build_windows.ps1",
    "package_source.py", "README.md", "LICENSE", ".gitignore", ".gitattributes",
    "AGENTS.md", "CHANGELOG.md", "CONTRIBUTING.md",
)]
files += sorted((project / "studio").rglob("*.py"))
files += sorted((project / "tests").rglob("*.py"))
files += [path for path in sorted((project / "docs").rglob("*.md"))
          if not path.name.endswith(".local.md") and path.name != "EDIT_REGRESSIONS.md"]
files += [project / "docs" / "nameplate-styles.png"]
files += sorted((project / "docs" / "screenshots").glob("*.png"))
files += sorted((project / ".github").rglob("*.yml"))
files += sorted((project / "config").rglob("*.json"))
files += sorted((project / "config").rglob("*.txt"))
files += sorted((project / "runtime_hooks").rglob("*.py"))
files += sorted((project / "scripts").rglob("*.py"))
with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, Path(f"VerticalVideoStudio-{__version__}-Source") / path.relative_to(project))
print(destination)
