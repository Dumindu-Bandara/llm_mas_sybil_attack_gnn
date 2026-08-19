# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Environment Setup (UF HiPerGator)

This project runs on the UF HiPerGator HPC cluster. Before running **any** Python
command — scripts, tests, `pip`/`conda` installs, notebooks — activate the project
environment first:

```bash
ml conda
conda activate mas_framework_test
```

- `ml conda` loads the conda module via Lmod (`ml` is HiPerGator's shorthand for
  `module load`) — the `conda` command is not available until this runs.
- Always use the `mas_framework_test` environment. Never fall back to system/base
  Python or a bare `python3` outside this environment for this project.
- Run the activation once per session; it persists across subsequent commands in
  the same shell. If a new command errors with `conda: command not found` or
  `ModuleNotFoundError`, re-run the two lines above before retrying.
- To verify you're in the right environment: `conda info --envs` should show
  `mas_framework_test` with an asterisk, and `which python` should point inside
  `.../envs/mas_framework_test/bin/python`.

## Testing

Always activate the environment before running tests:

```bash
ml conda
conda activate mas_framework_test
pytest
```

- Install any missing test dependencies with
  `conda install -n mas_framework_test <package>` or
  `conda run -n mas_framework_test pip install <package>` — do not `pip install`
  outside the environment.
- If tests need to run inside a SLURM job (`sbatch`/`srun`), include the same
  `ml conda && conda activate mas_framework_test` line before invoking `pytest`
  or any Python entry point in the job script.

## Troubleshooting

- `conda activate` fails or the environment isn't found: run `conda env list` to
  confirm `mas_framework_test` exists, and `ml spider conda` to check which
  conda module versions are available on the cluster.
- Commands seem to silently use the wrong Python: re-run `ml conda && conda
  activate mas_framework_test` — a new shell/session on HiPerGator starts
  without any module loaded by default.