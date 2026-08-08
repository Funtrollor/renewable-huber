#!/usr/bin/env bash
# Create a WSL development environment and build the native extensions.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/setup-wsl-venv.sh [--profile PROFILE]

Profiles:
  minimal    base package, development tools and native CPU extension
  cpu-full   minimal + pandas/SciPy/scikit-learn/PyTorch/TensorFlow CPU tests
  cuda-full  cpu-full + CuPy and the native CUDA extension

Legacy --cuda is accepted as an alias for --profile cuda-full.
EOF
}

PROFILE=minimal
while (($#)); do
    case "$1" in
        --profile)
            [ "$#" -ge 2 ] || { echo "--profile requires a value" >&2; exit 2; }
            PROFILE=$2
            shift
            ;;
        --cuda) PROFILE=cuda-full ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done
case "$PROFILE" in
    minimal|cpu-full|cuda-full) ;;
    *) echo "unknown profile: $PROFILE" >&2; usage >&2; exit 2 ;;
esac

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV="$REPO/.venv"

# shellcheck disable=SC1091
[ -f "$HOME/.cargo/env" ] && . "$HOME/.cargo/env"
command -v cargo >/dev/null || {
    echo "cargo missing; run scripts/setup-wsl-toolchain.sh first" >&2
    exit 1
}
command -v cc >/dev/null || {
    echo "no C compiler; run scripts/setup-wsl-toolchain.sh first" >&2
    exit 1
}

[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/python" -m pip install -q --upgrade pip

EXTRAS=dev,native-build
if [ "$PROFILE" = "cpu-full" ] || [ "$PROFILE" = "cuda-full" ]; then
    EXTRAS=$EXTRAS,pandas,sklearn,gpu-torch,gpu-tensorflow
fi
if [ "$PROFILE" = "cuda-full" ]; then
    EXTRAS=$EXTRAS,gpu-cupy
fi
echo "==> editable package profile: $PROFILE ($EXTRAS)"
"$VENV/bin/python" -m pip install -q -e "$REPO[$EXTRAS]"

echo "==> native CPU extension"
( cd "$REPO/native/python-cpu" && \
  VIRTUAL_ENV="$VENV" "$VENV/bin/python" -m maturin build --release \
      --out "$REPO/build/native-cpu-wheel" )
"$VENV/bin/python" -m pip install -q --force-reinstall --no-deps \
    "$(ls -t "$REPO"/build/native-cpu-wheel/renewable_huber_native_cpu-*.whl | head -1)"

if [ "$PROFILE" = "cuda-full" ]; then
    echo "==> native CUDA extension"
    export PATH=/usr/local/cuda/bin:$PATH
    command -v nvcc >/dev/null || {
        echo "nvcc missing; rerun setup-wsl-toolchain.sh --cuda" >&2
        exit 1
    }
    export CMAKE_GENERATOR=Ninja
    export RH_CUDA_ARCHITECTURES=native
    # Local development links against the installed CUDA toolkit. Release
    # wheel repair and dependency policy belong to the packaging workflow.
    ( cd "$REPO/native/python-cuda" && \
      VIRTUAL_ENV="$VENV" "$VENV/bin/python" -m maturin build --release \
          --auditwheel skip --out "$REPO/build/native-cuda-wheel" )
    "$VENV/bin/python" -m pip install -q --force-reinstall --no-deps \
        "$(ls -t "$REPO"/build/native-cuda-wheel/renewable_huber_native_cuda-*.whl | head -1)"
fi

echo
"$VENV/bin/python" "$REPO/scripts/verify_wsl_environment.py" --profile "$PROFILE"
echo
echo "next: $VENV/bin/python -m unittest discover -s tests -v"
