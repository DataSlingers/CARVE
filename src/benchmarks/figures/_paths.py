"""Where figures are written.

code/vis/ is gitignored. Copying a figure into overleaf/vis/ stays a manual
step, per the existing convention, so a regenerated figure never silently
changes the manuscript.

VIS_ROOT is anchored to the repository root, derived from this file's own
location, rather than the interpreter's working directory. nbconvert
executes each notebook with that notebook's own directory as the working
directory, so a CWD-relative path would scatter figures under
notebooks/vis/ and notebooks/case_studies/vis/ instead of the one code/vis/
tree every entry point shares.
"""

from pathlib import Path

# src/benchmarks/figures/_paths.py -> repo root is three directories up.
REPO_ROOT = Path(__file__).resolve().parents[3]
VIS_ROOT = REPO_ROOT / "vis"
BENCHMARKING_DIR = VIS_ROOT / "benchmarking"
CASE_STUDY_DIR = VIS_ROOT / "case_studies"


def figure_path(name: str, *, subdir: Path, out_dir: Path | None = None) -> Path:
    """Resolve a figure's output path, honoring an explicit override."""
    base = Path(out_dir) if out_dir is not None else subdir
    return base / name
