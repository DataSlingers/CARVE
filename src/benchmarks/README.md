# CARVE benchmarks

Reproduces the simulated benchmarks, tables, and figures for the CARVE
manuscript.

## Install

Benchmarks require a repo checkout with an editable install. A wheel install
is not enough: `carve.sim`, which generates the simulated data, is excluded
from the wheel, and this package is excluded too.

```bash
pip install -e ".[dev,benchmarks]"
```

The editable install writes a `.pth` file pointing at `code/src`, which is
what makes `import benchmarks` work from any working directory, including
`notebooks/`. If `import benchmarks` fails immediately after an editable
install, setuptools has switched from its static-path strategy to the strict
import finder; the fix is to add `benchmarks` back to
`[tool.setuptools.packages.find]` and exclude it from the wheel another way.

## Run

```bash
python -m benchmarks.run --scenario gaussians
python -m benchmarks.run --all --n-jobs 8
```

Working runs land in `results/runs/<scenario>/<config-hash>/` and are
gitignored. Publishing a run is a separate, deliberate step:

```bash
python -m benchmarks.run --promote results/runs/gaussians/<config-hash>
```
