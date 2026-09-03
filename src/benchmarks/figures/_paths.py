"""Where figures are written.

code/vis/ is gitignored. Copying a figure into overleaf/vis/ stays a manual
step, per the existing convention, so a regenerated figure never silently
changes the manuscript.
"""

from pathlib import Path

VIS_ROOT = Path("vis")
BENCHMARKING_DIR = VIS_ROOT / "benchmarking"
CASE_STUDY_DIR = VIS_ROOT / "case_studies"


def figure_path(name: str, *, subdir: Path, out_dir: Path | None = None) -> Path:
    """Resolve a figure's output path, honoring an explicit override."""
    base = Path(out_dir) if out_dir is not None else subdir
    return base / name
