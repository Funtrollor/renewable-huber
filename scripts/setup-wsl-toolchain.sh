#!/usr/bin/env bash
# One-time WSL2 toolchain bootstrap for renewable-huber.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/setup-wsl-toolchain.sh [--cuda]

Without --cuda this installs the C/C++/Python/Rust CPU build toolchain.
--cuda additionally installs NVIDIA's WSL CUDA 12 toolkit, never a Linux
display driver.
EOF
}

WITH_CUDA=0
while (($#)); do
    case "$1" in
        --cuda) WITH_CUDA=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

echo "==> base toolchain"
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    ninja-build \
    pkg-config \
    python3-venv \
    python3-pip \
    python3-dev

if ! command -v cargo >/dev/null; then
    echo "==> Rust stable toolchain (rustup minimal profile)"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
        sh -s -- -y --profile minimal --default-toolchain stable
    # shellcheck disable=SC1091
    . "$HOME/.cargo/env"
fi
rustup component add rustfmt clippy

if [ "$WITH_CUDA" = "1" ]; then
    echo "==> CUDA 12 toolkit for WSL"
    # WSL uses the Windows display driver through passthrough. The wsl-ubuntu
    # repository below contains the toolkit without installing a Linux driver.
    RH_CUDA_KEYRING_TMP=$(mktemp -d)
    cleanup() { rm -rf "$RH_CUDA_KEYRING_TMP"; }
    trap cleanup EXIT
    curl -fsSL -o "$RH_CUDA_KEYRING_TMP/cuda-keyring.deb" \
        https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i "$RH_CUDA_KEYRING_TMP/cuda-keyring.deb"
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit-12-9 || sudo apt-get install -y cuda-toolkit-12-6
    if ! grep -q '/usr/local/cuda/bin' "$HOME/.bashrc" 2>/dev/null; then
        echo 'export PATH=/usr/local/cuda/bin:$PATH' >> "$HOME/.bashrc"
        echo 'added /usr/local/cuda/bin to PATH in ~/.bashrc'
    fi
fi

echo
echo "==> result"
for tool in gcc g++ cmake ninja python3 cargo rustc; do
    printf '  %-10s %s\n' "$tool" "$(command -v "$tool")"
done
if [ "$WITH_CUDA" = "1" ]; then
    printf '  %-10s %s\n' nvcc "$(command -v nvcc 2>/dev/null || echo '/usr/local/cuda/bin/nvcc (open a new shell)')"
fi
echo
echo "next: bash scripts/setup-wsl-venv.sh --profile minimal"
