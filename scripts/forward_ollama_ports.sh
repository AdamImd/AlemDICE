#!/usr/bin/env bash
set -euo pipefail

# Forward four Ollama servers from the remote host to matching local ports.
# An optional first argument overrides the SSH host (default: kingpin).
ssh_host="${1:-kingpin}"

exec ssh -N \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L 127.0.0.1:11434:127.0.0.1:11434 \
  -L 127.0.0.1:11435:127.0.0.1:11435 \
  -L 127.0.0.1:11436:127.0.0.1:11436 \
  -L 127.0.0.1:11437:127.0.0.1:11437 \
  "${ssh_host}"
