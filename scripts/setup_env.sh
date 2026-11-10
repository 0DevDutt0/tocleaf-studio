#!/usr/bin/env bash
# Create the Tocleaf development environment. Run from WSL, where the GPU is visible.
# The virtualenv lives on the Linux filesystem (ext4), not on /mnt/e, for speed and symlink support.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/tocleaf}"
uv sync --group dev
echo "Environment ready: $UV_PROJECT_ENVIRONMENT"
