#!/bin/bash
set -euo pipefail

# Only run this setup in Claude Code on the web (remote) sessions.
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

if command -v Rscript >/dev/null 2>&1; then
  echo "R is already installed: $(Rscript -e 'cat(R.version.string)')"
  exit 0
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq r-base r-base-dev

echo "Installed R: $(Rscript -e 'cat(R.version.string)')"
