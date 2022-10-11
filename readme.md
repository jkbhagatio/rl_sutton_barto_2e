[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen)](https://github.com/pre-commit/pre-commit)
[![build_and_tests](https://github.com/jkbhagatio/rl2e/actions/workflows/build_env_run_tests.yml/badge.svg)](https://github.com/jkbhagatio/rl2e/actions/workflows/build_env_run_tests.yml)
[![code_coverage](https://codecov.io/gh/jkbhagatio/rl2e/branch/main/graph/badge.svg?token=L2HD3QLAQG)](https://codecov.io/gh/jkbhagatio/rl2e)
[![sunburst_of_coverage](https://codecov.io/gh/jkbhagatio/rl2e/branch/main/graphs/sunburst.svg?token=L2HD3QLAQG)](https://codecov.io/gh/jkbhagatio/rl2e)

# rl2e

Notes and Exercises for Reinforcement Learning 2e by Sutton and
Barto.

Organized in ipython notebooks, by chapter.

## Quickstart
```
git clone https://github.com/jkbhagatio/rl2e
cd rl2e
conda env create -f env.yml
conda activate rl2e
pip install -e .
```

### Coding Best Practices

- Linting and formatting:
	- flake8 (flake8-docstrings, pydocstyle, flake8-pyproject)
	- black
	- isort
	- pyupgrade
- Code analysis and testing:
	- mypy
	- bandit
	- pytest (pytest-cov)
	- dependabot
	- CodeQL scanning
- CI/CD:
	- pre-commit
	- GitHub actions
	- codecov
