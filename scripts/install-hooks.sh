#!/usr/bin/env bash
# Installs the repo's versioned git hooks (.githooks/) by pointing
# core.hooksPath at this directory. Run once per clone.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

git config core.hooksPath .githooks

echo "✓ core.hooksPath = .githooks"
echo "  Installed hooks:"
ls -1 .githooks | sed 's/^/    - /'
