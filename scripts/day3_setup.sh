#!/usr/bin/env bash
# Day 3: install openpi (its own Python 3.11 env via uv) next to the LIBERO env,
# and the small openpi-client package inside the LIBERO env.
# Run from anywhere on the instance:  bash ~/project/vla/scripts/day3_setup.sh
set -euo pipefail

echo "== disk before =="; df -h ~ | tail -1

# 1. uv (openpi's package manager)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

# 2. openpi, pinned to the commit this harness was written against
cd ~/project
if [ ! -d openpi ]; then
  git clone https://github.com/Physical-Intelligence/openpi.git
fi
cd openpi
git fetch -q origin && git checkout -q 215abfb || echo "(pin failed; using current HEAD)"
GIT_LFS_SKIP_SMUDGE=1 uv sync

# 3. the client, inside the LIBERO env (whatever python start.sh activates)
python -m pip install -e ~/project/openpi/packages/openpi-client
python -c "import openpi_client, numpy; print('openpi_client OK, numpy', numpy.__version__)"

echo "== disk after =="; df -h ~ | tail -1
cat <<'EOF'

Next: start the policy server in its own tmux window (it downloads
pi05_libero, ~12 GB, into ~/.cache/openpi on first run):

  tmux new -s server
  cd ~/project/openpi && uv run scripts/serve_policy.py --env LIBERO
  # wait for a line about serving on port 8000, then Ctrl+B D to detach

Then, back in your normal shell:
  source ~/project/start.sh && cd ~/project/vla
  python tests/test_policy.py 2>&1 | tee docs/test_policy.txt
EOF
