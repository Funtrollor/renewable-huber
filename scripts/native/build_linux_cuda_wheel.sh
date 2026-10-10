#!/usr/bin/env bash
# Build one manylinux_2_28 x86-64 CUDA 12 wheel of renewable-huber-native-cuda.
#
# Runs *inside* quay.io/pypa/manylinux_2_28_x86_64 (AlmaLinux 8, glibc 2.28),
# so the platform tag the wheel claims is true of the binary. The release
# workflow and the pull-request CI job both call this script, which is what
# lets ordinary CI prove the release build before any tag exists.
#
# The wheel links the CUDA runtime, cuBLAS and cuSOLVER dynamically and does
# not bundle them: they arrive as the package's nvidia-*-cu12 dependencies and
# renewable_huber._cuda_runtime preloads them at import. auditwheel is skipped
# for exactly that reason -- repairing would vendor several hundred megabytes of
# NVIDIA libraries into every wheel.
#
# Usage (from the repository root, mounted at /io):
#   docker run --rm -e RH_CUDA_ARCHITECTURES -v "$PWD:/io" -w /io \
#     quay.io/pypa/manylinux_2_28_x86_64@sha256:<digest pinned in the workflows> \
#     bash scripts/native/build_linux_cuda_wheel.sh --python 3.12 --out dist-native-cuda
set -euo pipefail

PYTHON_VERSION=""
OUT="dist-native-cuda"
while (($#)); do
    case "$1" in
        --python) PYTHON_VERSION=$2; shift ;;
        --out) OUT=$2; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
[ -n "$PYTHON_VERSION" ] || { echo "--python is required" >&2; exit 2; }
: "${RH_CUDA_ARCHITECTURES:?set RH_CUDA_ARCHITECTURES, e.g. 75-real;80-real;120}"

# The release compiles against CUDA 12.9; the wheel's nvidia-*-cu12 floors
# (scripts/native/validate_release_artifacts.py) name the same components.
CUDA_SERIES="12-9"
export CUDA_PATH="/usr/local/cuda-12.9"

echo "==> CUDA ${CUDA_SERIES} toolkit from NVIDIA's RHEL 8 repository"
dnf install -y dnf-plugins-core
dnf config-manager --add-repo \
    https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
dnf install -y \
    "cuda-nvcc-${CUDA_SERIES}" \
    "cuda-cudart-devel-${CUDA_SERIES}" \
    "cuda-cuobjdump-${CUDA_SERIES}" \
    "libcublas-devel-${CUDA_SERIES}" \
    "libcusolver-devel-${CUDA_SERIES}" \
    "libcusparse-devel-${CUDA_SERIES}" \
    "libnvjitlink-devel-${CUDA_SERIES}"
export PATH="${CUDA_PATH}/bin:${PATH}"
nvcc --version
gcc --version | head -n 1

echo "==> Rust toolchain"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable
# shellcheck disable=SC1091
. "${HOME}/.cargo/env"
rustc --version

tag="cp${PYTHON_VERSION/./}"
PY="/opt/python/${tag}-${tag}/bin/python"
[ -x "$PY" ] || { echo "no CPython ${PYTHON_VERSION} in this image: $PY" >&2; exit 2; }
"$PY" -m pip install --disable-pip-version-check "maturin>=1.8,<2" cmake ninja
export PATH="$(dirname "$PY"):${PATH}"
export CMAKE_GENERATOR=Ninja

echo "==> Wheel for CPython ${PYTHON_VERSION}, architectures ${RH_CUDA_ARCHITECTURES}"
mkdir -p "$OUT"
out_dir=$(cd "$OUT" && pwd)
(
    cd native/python-cuda
    "$PY" -m maturin build --release --locked \
        --interpreter "$PY" \
        --compatibility manylinux_2_28 \
        --auditwheel skip \
        --out "$out_dir"
)

echo "==> Fat-binary payload"
shopt -s nullglob
wheels=("$out_dir"/renewable_huber_native_cuda-*-"${tag}"-*manylinux_2_28_x86_64.whl)
if ((${#wheels[@]} != 1)); then
    echo "expected one manylinux_2_28 CUDA wheel for ${tag}, found ${#wheels[@]}" >&2
    exit 1
fi
inspection=$(mktemp -d)
"$PY" -m zipfile -e "${wheels[0]}" "$inspection"
extensions=("$inspection"/_renewable_huber_native_cuda/*.so)
if ((${#extensions[@]} != 1)); then
    echo "expected one CUDA extension module, found ${#extensions[@]}" >&2
    exit 1
fi
sass=$(cuobjdump --list-elf "${extensions[0]}")
ptx=$(cuobjdump --list-ptx "${extensions[0]}" || true)
# "NN-real" requests SASS only; a bare "NN" requests SASS plus PTX, which is
# what keeps the wheel loadable on GPUs newer than the build.
expected_ptx=()
IFS=';' read -r -a requested <<<"$RH_CUDA_ARCHITECTURES"
for entry in "${requested[@]}"; do
    architecture=${entry%-real}
    grep -q "sm_${architecture}\.cubin" <<<"$sass" \
        || { echo "missing SM ${architecture} SASS" >&2; exit 1; }
    if [ "$entry" = "$architecture" ]; then
        expected_ptx+=("$architecture")
    fi
done
found_ptx=$( (grep -oE '(compute|sm)_[0-9]+\.ptx' <<<"$ptx" || true) | grep -oE '[0-9]+' | sort -u | xargs || true)
if [ "$found_ptx" != "${expected_ptx[*]}" ]; then
    echo "expected PTX for [${expected_ptx[*]}], found [${found_ptx}]" >&2
    exit 1
fi
if readelf -d "${extensions[0]}" | grep -E 'NEEDED' | grep -vE \
    'lib(cudart\.so\.12|cublas\.so\.12|cusolver\.so\.11|stdc\+\+\.so\.6|gcc_s\.so\.1|m\.so\.6|c\.so\.6|pthread\.so\.0|dl\.so\.2|rt\.so\.1)|ld-linux-x86-64\.so\.2'; then
    echo "the extension links a library outside the declared runtime closure" >&2
    exit 1
fi
echo "Built ${wheels[0]##*/}: SASS [${RH_CUDA_ARCHITECTURES}], PTX [${found_ptx}]"
