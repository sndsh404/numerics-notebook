"""Assemble the mkdocs source tree into site_src/ and build the site.

The repo keeps notes/, docs/, and README.md at the top level, and mkdocs
refuses a docs_dir that is the repo root. This script copies the markdown
into site_src/ (gitignored) so mkdocs.yml can point at a real subdirectory.
Run: python scripts/build_docs.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "site_src"


def main() -> None:
    if SRC.exists():
        shutil.rmtree(SRC)
    (SRC / "notes").mkdir(parents=True)
    (SRC / "docs").mkdir()

    shutil.copy(ROOT / "README.md", SRC / "index.md")
    for note in sorted((ROOT / "notes").glob("*.md")):
        shutil.copy(note, SRC / "notes" / note.name)
    for doc in (ROOT / "docs").glob("*.md"):
        shutil.copy(doc, SRC / "docs" / doc.name)

    # Links like ../tests/test_x.py point outside the copied tree. On the
    # site they would 404, so point them at GitHub instead.
    github = "https://github.com/sndsh404/numerics-notebook/blob/main/"
    for md in list((SRC / "notes").glob("*.md")) + list((SRC / "docs").glob("*.md")):
        text = md.read_text(encoding="utf-8")
        md.write_text(text.replace("](../", f"]({github}"), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build"], cwd=ROOT
    )
    if result.returncode != 0:
        sys.exit(result.returncode)
    print(f"site built into {ROOT / 'site'}")


if __name__ == "__main__":
    main()
