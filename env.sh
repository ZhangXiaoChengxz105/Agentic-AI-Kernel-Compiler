# 用法：source env.sh（在 521 目录下），或 source /path/to/521/env.sh
export CS521="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$CS521/tools/node/bin:$CS521/npm-global/bin:$PATH"
export NPM_CONFIG_PREFIX="$CS521/npm-global"
export NPM_CONFIG_CACHE="$CS521/.npm-cache"
export CLAUDE_CONFIG_DIR="$CS521/.claude"
export DISABLE_AUTOUPDATER=1
export PIP_CACHE_DIR="$CS521/.cache/pip"
export CONDA_PKGS_DIRS="$CS521/.cache/conda-pkgs"
export XDG_CACHE_HOME="$CS521/.cache"
export TRITON_CACHE_DIR="$CS521/.cache/triton"
export TORCH_HOME="$CS521/.cache/torch"
export TORCH_EXTENSIONS_DIR="$CS521/.cache/torch_extensions"
export CUDA_CACHE_PATH="$CS521/.cache/nv"

# --- Git: keep all git config/credentials inside the workspace (never touch ~/.gitconfig or ~/.ssh) ---
export GIT_CONFIG_GLOBAL="$CS521/.gitconfig"          # replaces ~/.gitconfig as the "global" config
if [ -f "$CS521/.secrets/ssh/id_ed25519" ]; then
  export GIT_SSH_COMMAND="ssh -i $CS521/.secrets/ssh/id_ed25519 -o IdentitiesOnly=yes -o UserKnownHostsFile=$CS521/.secrets/ssh/known_hosts"
fi

# --- Rust toolchain (installed into workspace; built VOLTA with it) ---
export RUSTUP_HOME="$CS521/tools/rustup"
export CARGO_HOME="$CS521/tools/cargo"
export PATH="$CARGO_HOME/bin:$PATH"

# --- Z3 shared library (from pip z3-solver in tools/z3), needed to link/run VOLTA ---
export LIBRARY_PATH="$CS521/tools/z3/z3/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export LD_LIBRARY_PATH="$CS521/tools/z3/z3/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
