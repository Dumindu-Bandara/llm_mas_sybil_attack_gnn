# Container Creation and Maintenance

## Build Commands for Apptainer

```bash
ml apptainer

export APPTAINER_CACHEDIR=/blue/prabhat/duminduaelamurem/.apptainer/cache
export APPTAINER_TMPDIR=/blue/prabhat/duminduaelamurem/.apptainer/tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

apptainer build --fakeroot containers/env_$(date +%F).sif containers/env.def
```

## Run the Apptainer as a Shell

```bash
ml apptainer

HF_CACHE=/blue/prabhat/duminduaelamurem/.cache/huggingface

apptainer shell --nv --contain \
    --bind "$PWD" \
    --bind "$HF_CACHE" \
    --bind "$HOME/.claude" \
    --bind "$HOME/.claude.json" \
    --env PYTHONPATH="$PWD/src" \
    --env HF_HOME="$HF_CACHE" \
    containers/env_2026-11-20.sif
```

Run this from the repo root. `--contain` turns off Apptainer's default mounts of `$HOME`, `/tmp` and the host working directory, so only the repo and the Hugging Face cache are visible from the host. Inside the container, `$HOME` and `/tmp` are empty and temporary. Anything written there is lost on exit, including `~/.cache/vllm` and the Triton cache. To keep one of those, bind a host directory for it and point the matching env var (for example `TRITON_CACHE_DIR`) at it.

## Run the Apptainer as a Instance

```bash
ml apptainer
unset DBUS_SESSION_BUS_ADDRESS
HF_CACHE=/blue/prabhat/duminduaelamurem/.cache/huggingface

apptainer instance start --nv --contain \
    --bind "$PWD" \
    --bind "$HF_CACHE" \
    --bind "$HOME/.claude" \
    --bind "$HOME/.claude.json" \
    --env PYTHONPATH="$PWD/src" \
    --env HF_HOME="$HF_CACHE" \
    containers/env_2026-11-20.sif sybil
```

```bash
apptainer shell --pwd "$PWD" --env PYTHONPATH="$PWD/src" --env HF_HOME="$HF_CACHE" instance://sybil


# or run a single command
apptainer exec --pwd "$PWD" instance://sybil python -c "import torch; print(torch.cuda.is_available())"
```

```bash
apptainer instance list        # see running instances
apptainer instance stop sybil  # stop it
```

## Verify the Apptainer Container

```bash
apptainer exec --nv containers/env_2026-11-20.sif python -c "import torch; print(torch.cuda.get_device_name())"
apptainer exec --nv containers/env_2026-11-20.sif vllm --version
```

## Installing a Dependency

The image is read-only, so you can't update packages in place. Change the versions in `pyproject.toml` and `uv.lock` on the host, then rebuild the `.sif`. The build installs exactly what's in `uv.lock`, because `env.def` runs `uv sync --frozen`.

### 1. Change the dependencies on the host

Run these from the repo root, outside the container (uv is at `~/.local/bin/uv`):

```bash
cd /blue/prabhat/duminduaelamurem/wd/2026_fall/llm_mas_sybil_attack_gnn/llm_mas_sybil_attack_gnn
export UV_CACHE_DIR=/blue/prabhat/duminduaelamurem/.cache/uv
```

| Task | Command |
| --- | --- |
| Add a package | `uv add --no-sync networkx` |
| Add a dev-only package | `uv add --no-sync --dev ruff` |
| Remove a package | `uv remove --no-sync networkx` |
| Change a pinned version | edit it in `pyproject.toml`, then `uv lock` |
| Upgrade one package within its allowed range | `uv lock --upgrade-package openai` |
| Upgrade everything | `uv lock --upgrade` |

Pass `--no-sync` to `uv add` and `uv remove`. Without it, uv also installs everything into a `.venv` in the repo, which means several GB of torch and vLLM you don't need on the host. `uv lock` never installs anything.

Most packages in `pyproject.toml` are pinned with `==`. For those, `--upgrade-package` won't move them; change the pin first.

### 2. Check what changed

```bash
git diff pyproject.toml
git diff --stat uv.lock
```

Watch for changes to torch, vLLM or transformers in particular. A small bump can change model outputs between experiment runs.

### 3. Rebuild the image

```bash
ml apptainer
export APPTAINER_CACHEDIR=/blue/prabhat/duminduaelamurem/.apptainer/cache
export APPTAINER_TMPDIR=/blue/prabhat/duminduaelamurem/.apptainer/tmp
apptainer build --fakeroot containers/env_$(date +%F).sif containers/env.def
```

The date in the filename keeps the old image, so you can go back to it or rerun old experiments with the exact environment they used.

### 4. Commit both files

Commit `pyproject.toml` and `uv.lock` together. The lock file plus `env.def` let anyone rebuild the same environment.

### Trying a package before committing to it

To test a package before changing the lock, start the container with a temporary writable layer. It's lost on exit:

```bash
apptainer shell --nv --contain --writable-tmpfs --bind "$PWD" --pwd "$PWD" containers/env_2026-11-20.sif
Apptainer> uv pip install --python /opt/venv/bin/python some-package
```

Use this only for quick experiments. Any results you keep should come from a rebuilt image, so they can be reproduced.
