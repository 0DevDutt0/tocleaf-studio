#!/usr/bin/env bash
# Clone and pin the upstream STAIR implementation OUTSIDE this repository.
#
# The upstream repository has no licence file, so its code must not be copied or redistributed
# from here. This script only creates a local checkout at the commit this project was validated
# against, and installs that checkout's locked environment with its own uv.lock.
set -euo pipefail

UPSTREAM_REPO="https://github.com/meghanadhpulivarthi/stair-clean.git"
UPSTREAM_COMMIT="3729c54f54e6e28aa3a4d714fc7ff8a3af72c0f3"
UPSTREAM_DIR="${TOCLEAF_UPSTREAM_DIR:-$HOME/stair-work/upstream/stair-clean}"

mkdir -p "$(dirname "$UPSTREAM_DIR")"
if [ ! -d "$UPSTREAM_DIR/.git" ]; then
  git clone "$UPSTREAM_REPO" "$UPSTREAM_DIR"
fi
cd "$UPSTREAM_DIR"
git fetch --quiet origin
git checkout --quiet "$UPSTREAM_COMMIT"
if [ "$(git rev-parse HEAD)" != "$UPSTREAM_COMMIT" ]; then
  echo "upstream checkout is not at $UPSTREAM_COMMIT" >&2
  exit 1
fi
uv sync
echo "Upstream ready: $UPSTREAM_DIR @ $UPSTREAM_COMMIT"
