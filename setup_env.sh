#!/usr/bin/env bash
# =====================================================================================================
# setup_env.sh: build the whole akc environment from scratch, inside this folder only (NCSA Delta).
#
#   bash setup_env.sh              # full setup (idempotent: finished steps are skipped)
#   bash setup_env.sh --check      # only verify an existing setup
#   bash setup_env.sh --tests      # full setup + run the CPU test suite at the end
#   AKC_ENV=envs/other bash setup_env.sh   # build the Python env somewhere else (default envs/akc)
#
# What it installs (all under the repository root; nothing in $HOME, no sudo, no `pip --user`,
# rustup runs with --no-modify-path so ~/.profile is untouched):
#   1. folders: tools/ envs/ third_party/ build/ results/ logs/ .secrets/ (mode 700)
#   2. Rust (rustup) in tools/rustup + tools/cargo: stable (builds VOLTA) and 1.95.0 (TIRx NumSim; cargo 1.100's
#      new build-dir layout breaks tirx-harness's artifact lookup)
#   3. libz3 from pip `z3-solver` into tools/z3 (VOLTA links against it)
#   4. third_party sources at pinned commits: VOLTA, TIRx-harness, tirx-kernels, (Lumen docs)
#   5. VOLTA release build (cargo build --release, 4 jobs)
#   6. Python 3.13 venv ($AKC_ENV) with the exact package lock setup/requirements-akc.lock
#      (torch 2.14.1+cu132; apache-tvm-tirx 0.1.0rc2 from its original PyPI file URL, sha256-pinned:
#      the version was removed from the PyPI index but tirx-harness 0.1.3rc0 needs it; installed with
#      --no-deps because torch and tirx-harness pin different nvrtc patch versions), then `pip install -e .`
#   7. verification: tool versions, imports, VOLTA, nvcc, (tests)
# Requirements: git, curl, a C compiler, nvcc (module cudatoolkit), Python 3.13 (module python/3.13.5-gcc13.3.1),
# ~10 GB free in the quota, internet access. Secrets (Lumen key) are NOT handled here: see the end of the run.
# =====================================================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
MODE="setup"; RUN_TESTS=0
for arg in "$@"; do
  case "$arg" in
    --check) MODE="check" ;;
    --tests) RUN_TESTS=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown option $arg"; exit 2 ;;
  esac
done

AKC_ENV="${AKC_ENV:-envs/akc}"
PY313_MODULE="python/3.13.5-gcc13.3.1"
RUST_EXTRA_TOOLCHAIN="1.95.0"
Z3_VERSION="5.4.1.0"
TORCH_INDEX="https://download.pytorch.org/whl/cu132"
# name|url|commit
PINNED_REPOS=(
  "volta|https://github.com/willtunnels/volta.git|c09155917ae45137c5535f4649982b8e51947e28"
  "tirx-harness|https://github.com/mlc-ai/TIRx-harness.git|99522bf2118d88d7c2b7974108ffb48269705fba"
  "tirx-kernels|https://github.com/mlc-ai/tirx-kernels.git|6d2f3e2b4b469e39eaef0b37d4653e57f7d56895"
  "lumen|https://github.com/ncsa/lumen.git|42c88227a30ac24fd1a2625cafe9ee8c8a4d0ecf"
)

mkdir -p logs
LOG="logs/setup_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
step() { printf '\n\033[1m==> [%s] %s\033[0m\n' "$1" "$2"; }
ok()   { printf '    ok: %s\n' "$*"; }
skip() { printf '    skip: %s\n' "$*"; }
die()  { printf '    ERROR: %s\n' "$*"; exit 1; }

# shellcheck disable=SC1091
source "$ROOT/env.sh"          # workspace-local caches, Rust/Z3 paths, git config location

# ---------------------------------------------------------------------------------------------------
step 0 "preflight ($MODE mode, log: $LOG)"
for c in git curl cc; do command -v "$c" >/dev/null || die "$c not found"; done
if ! command -v nvcc >/dev/null; then
  type module >/dev/null 2>&1 && module load cudatoolkit 2>/dev/null || true
fi
command -v nvcc >/dev/null && ok "nvcc: $(nvcc --version | tail -1)" || echo "    warning: nvcc not found (module load cudatoolkit); VOLTA builds and GPU work need it"
PY313="${PYTHON313:-}"
if [ -z "$PY313" ]; then
  if type module >/dev/null 2>&1; then module load "$PY313_MODULE" 2>/dev/null || true; fi
  PY313="$(command -v python3.13 || true)"
fi
[ -n "$PY313" ] || PY313="/sw/rh9.4/spack/v1.0.0/sw/linux-x86_64_v2/python-3.13.5-7bp2p4z/bin/python3"
"$PY313" -c 'import sys; assert sys.version_info[:2] == (3, 13)' 2>/dev/null || die "Python 3.13 not found (set PYTHON313=/path/to/python3.13)"
ok "python 3.13: $PY313"
if command -v quota >/dev/null; then
  used_soft=$(quota -s 2>/dev/null | awk -F'|' '/\/u\//{gsub(/ /,"",$3); gsub(/ /,"",$4); print $3" of "$4; exit}') || true
  if [ -n "${used_soft:-}" ]; then ok "home quota used: $used_soft (the full setup needs ~10 GB)"; fi
fi

if [ "$MODE" = "setup" ]; then
# ---------------------------------------------------------------------------------------------------
step 1 "folders"
mkdir -p tools envs third_party build results logs .secrets
chmod 700 .secrets
ok "tools/ envs/ third_party/ build/ results/ logs/ .secrets/"

# ---------------------------------------------------------------------------------------------------
step 2 "Rust toolchains (tools/rustup, tools/cargo)"
if [ ! -x "$CARGO_HOME/bin/rustup" ]; then
  curl -sSf https://sh.rustup.rs -o build/rustup-init.sh
  sh build/rustup-init.sh -y --no-modify-path --profile minimal --default-toolchain stable
  rm -f build/rustup-init.sh
  ok "rustup installed"
else
  skip "rustup present"
fi
"$CARGO_HOME/bin/rustup" toolchain list | grep -q "^$RUST_EXTRA_TOOLCHAIN" || \
  "$CARGO_HOME/bin/rustup" toolchain install "$RUST_EXTRA_TOOLCHAIN" --profile minimal
ok "$("$CARGO_HOME/bin/rustc" --version); extra toolchain $RUST_EXTRA_TOOLCHAIN"

# ---------------------------------------------------------------------------------------------------
step 3 "libz3 (tools/z3, from pip z3-solver==$Z3_VERSION)"
if [ ! -f tools/z3/z3/lib/libz3.so ]; then
  "$PY313" -m pip install --no-cache-dir -q --target tools/z3 "z3-solver==$Z3_VERSION"
  ok "installed"
else
  skip "tools/z3/z3/lib/libz3.so present"
fi

# ---------------------------------------------------------------------------------------------------
step 4 "third_party sources at pinned commits"
for entry in "${PINNED_REPOS[@]}"; do
  IFS='|' read -r name url commit <<< "$entry"
  dir="third_party/$name"
  if [ -d "$dir/.git" ] && [ "$(git -C "$dir" rev-parse HEAD)" = "$commit" ]; then
    skip "$name @ ${commit:0:7}"
    continue
  fi
  if [ ! -d "$dir/.git" ]; then
    git init -q "$dir"
    git -C "$dir" remote add origin "$url"
  fi
  git -C "$dir" fetch -q --depth 1 origin "$commit"
  git -C "$dir" checkout -q --detach FETCH_HEAD
  ok "$name @ ${commit:0:7}"
done

# ---------------------------------------------------------------------------------------------------
step 5 "VOLTA release build"
if [ ! -x "$ROOT/third_party/volta/target/release/volta" ]; then
  (cd third_party/volta && cargo build --release -j 4)
  ok "built"
else
  skip "third_party/volta/target/release/volta present"
fi

# ---------------------------------------------------------------------------------------------------
step 6 "Python environment $AKC_ENV (exact lock: setup/requirements-akc.lock)"
if [ ! -x "$AKC_ENV/bin/python" ]; then
  "$PY313" -m venv "$AKC_ENV"
  ok "venv created"
fi
"$AKC_ENV/bin/python" -m pip install -q --no-cache-dir -U pip
if ! "$AKC_ENV/bin/python" -c "import torch, tvm_tirx, tirx_harness" 2>/dev/null; then
  "$AKC_ENV/bin/python" -m pip install --no-cache-dir --no-deps -r setup/requirements-akc.lock --extra-index-url "$TORCH_INDEX"
  ok "packages installed (--no-deps, exact versions)"
else
  skip "torch / tvm_tirx / tirx_harness already importable"
fi
"$AKC_ENV/bin/python" -m pip install -q --no-deps --no-build-isolation -e . && ok "akc installed (editable)"
fi   # MODE=setup

# ---------------------------------------------------------------------------------------------------
step 7 "verification"
fail=0
check() { if eval "$2" >/dev/null 2>&1; then ok "$1"; else echo "    FAIL: $1   ($2)"; fail=1; fi; }
check "rustc stable"            "$CARGO_HOME/bin/rustc --version"
check "rust $RUST_EXTRA_TOOLCHAIN" "$CARGO_HOME/bin/rustup run $RUST_EXTRA_TOOLCHAIN cargo --version"
check "libz3"                   "test -f tools/z3/z3/lib/libz3.so"
check "VOLTA binary runs"       "third_party/volta/target/release/volta --help"
check "nvcc"                    "nvcc --version"
check "python env"              "$AKC_ENV/bin/python --version"
check "imports (torch, tvm_tirx, tirx_harness, akc)" \
      "$AKC_ENV/bin/python -c 'import torch, tvm_tirx, tvm_ffi, tirx_kernels.tirx_lite, tirx_harness, akc.validation'"
check "akc CLI"                 "$AKC_ENV/bin/akc --help"
"$AKC_ENV/bin/python" - <<'EOF' || fail=1
import torch, tvm_tirx, sys
print(f"    torch {torch.__version__} (CUDA {torch.version.cuda}), GPU visible here: {torch.cuda.is_available()}")
EOF
if [ "$RUN_TESTS" = 1 ]; then
  step 8 "CPU test suite"
  "$AKC_ENV/bin/python" -m unittest discover -s tests 2>&1 | tail -4 || fail=1
fi
if [ ! -f .secrets/lumen.env ]; then
  echo
  echo "    Lumen key not set (needed only for LLM calls). Create it yourself, e.g.:"
  echo "      echo 'export LUMEN_API_KEY=sk_...' > .secrets/lumen.env && chmod 600 .secrets/lumen.env"
fi
echo
[ "$fail" = 0 ] && echo "setup OK. Use it with:  source env.sh   (activates $AKC_ENV)" || { echo "setup finished with failures (see $LOG)"; exit 1; }
