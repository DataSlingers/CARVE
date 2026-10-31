"""Where figures are written.

code/vis/ is gitignored. Copying a figure into overleaf/vis/ is a manual
step.

VIS_ROOT is anchored to the repository root, derived from this file's own
location, rather than the working directory, which nbconvert sets to each
notebook's own directory.
"""

from pathlib import Path

# src/benchmarks/figures/_paths.py -> repo root is three directories up.
REPO_ROOT = Path(__file__).resolve().parents[3]
VIS_ROOT = REPO_ROOT / "vis"
BENCHMARKING_DIR = VIS_ROOT / "benchmarking"
CASE_STUDY_DIR = VIS_ROOT / "case_studies"
GUIDANCE_DIR = VIS_ROOT / "guidance"
# The main figures as uploaded to PLOS, one TIFF per figure named by its
# number in the manuscript (Fig1.tif, ...), as PLOS asks.
PLOS_DIR = VIS_ROOT / "plos"


def figure_path(name: str, *, subdir: Path, out_dir: Path | None = None) -> Path:
    """Resolve a figure's output path, honoring an explicit override."""
    base = Path(out_dir) if out_dir is not None else subdir
    return base / name


def plos_figure_path(number: int, *, out_dir: Path | None = None) -> Path:
    """Resolve the PLOS figure file for manuscript figure ``number``."""
    base = Path(out_dir) if out_dir is not None else PLOS_DIR
    return base / f"Fig{number}.tif"
