#!/usr/bin/env bash
# Build OpenCV + opencv_contrib with CUDA (cudaoptflow / Farneback) for this host.
#
# Target machine notes (DGX Spark / GB10):
#   - aarch64, CUDA 13, compute capability 12.1
#   - installs to $INSTALL_PREFIX (default: ~/.local/opencv-cuda)
#
# Usage:
#   bash scripts/setup/build_opencv_cuda.sh
#   OPENCV_TAG=4.14.0 bash scripts/setup/build_opencv_cuda.sh

set -euo pipefail

OPENCV_TAG="${OPENCV_TAG:-4.14.0}"
SRC_ROOT="${SRC_ROOT:-$HOME/src/opencv-cuda-build}"
INSTALL_PREFIX="${INSTALL_PREFIX:-$HOME/.local/opencv-cuda}"
CUDA_ARCH_BIN="${CUDA_ARCH_BIN:-12.1}"
CUDA_ARCH_PTX="${CUDA_ARCH_PTX:-12.1}"
JOBS="${JOBS:-$(nproc)}"
PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-$(command -v python3)}"

export CC="${CC:-/usr/bin/gcc-13}"
export CXX="${CXX:-/usr/bin/g++-13}"
export CUDAHOSTCXX="${CUDAHOSTCXX:-$CXX}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
# Help nvcc tolerate newer host compilers if needed.
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:--allow-unsupported-compiler}"

# Prefer conda ffmpeg/pkg-config when present (common on this host).
CONDA_PREFIX_DETECT="${CONDA_PREFIX:-$("$PYTHON_EXECUTABLE" -c 'import sys; print(sys.prefix)')}"
if [[ -d "$CONDA_PREFIX_DETECT/lib/pkgconfig" ]]; then
  export PKG_CONFIG_PATH="$CONDA_PREFIX_DETECT/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
  export LD_LIBRARY_PATH="$CONDA_PREFIX_DETECT/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ ! -x "$PYTHON_EXECUTABLE" ]]; then
  echo "ERROR: python not found: $PYTHON_EXECUTABLE" >&2
  exit 1
fi

PYTHON_INCLUDE_DIR="$("$PYTHON_EXECUTABLE" -c 'import sysconfig; print(sysconfig.get_path("include"))')"
PYTHON_LIBRARY="$("$PYTHON_EXECUTABLE" -c 'import sysconfig, pathlib; lib=pathlib.Path(sysconfig.get_config_var("LIBDIR") or ""); name=sysconfig.get_config_var("LDLIBRARY") or ""; print(lib / name if lib and name else "")')"
NUMPY_INCLUDE="$("$PYTHON_EXECUTABLE" -c 'import numpy; print(numpy.get_include())')"
PYTHON_PACKAGES_PATH="$INSTALL_PREFIX/lib/python/site-packages"
PYTHON_VERSION_MAJOR_MINOR="$("$PYTHON_EXECUTABLE" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"

echo "==> OpenCV CUDA build"
echo "    tag            : $OPENCV_TAG"
echo "    src            : $SRC_ROOT"
echo "    prefix         : $INSTALL_PREFIX"
echo "    arch           : BIN=$CUDA_ARCH_BIN PTX=$CUDA_ARCH_PTX"
echo "    python         : $PYTHON_EXECUTABLE ($PYTHON_VERSION_MAJOR_MINOR)"
echo "    jobs           : $JOBS"

mkdir -p "$SRC_ROOT"
cd "$SRC_ROOT"

if [[ ! -d opencv/.git ]]; then
  git clone --depth 1 --branch "$OPENCV_TAG" https://github.com/opencv/opencv.git
else
  git -C opencv fetch --depth 1 origin "refs/tags/$OPENCV_TAG:refs/tags/$OPENCV_TAG" || true
  git -C opencv checkout -f "$OPENCV_TAG"
fi

if [[ ! -d opencv_contrib/.git ]]; then
  git clone --depth 1 --branch "$OPENCV_TAG" https://github.com/opencv/opencv_contrib.git
else
  git -C opencv_contrib fetch --depth 1 origin "refs/tags/$OPENCV_TAG:refs/tags/$OPENCV_TAG" || true
  git -C opencv_contrib checkout -f "$OPENCV_TAG"
fi

# Keep the build lean: Farneback needs cudaoptflow (+ cudaarithm / cudaimgproc / cudawarping).
rm -rf "$SRC_ROOT/build"
mkdir -p "$SRC_ROOT/build"
cd "$SRC_ROOT/build"

CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
  -DCMAKE_C_COMPILER="$CC"
  -DCMAKE_CXX_COMPILER="$CXX"
  -DCUDA_HOST_COMPILER="$CUDAHOSTCXX"
  -DWITH_CUDA=ON
  -DWITH_CUBLAS=ON
  -DWITH_CUFFT=ON
  -DWITH_CUDNN=OFF
  -DOPENCV_DNN_CUDA=OFF
  -DWITH_NVCUVID=OFF
  -DWITH_NVCUVENC=OFF
  -DCUDA_ARCH_BIN="$CUDA_ARCH_BIN"
  -DCUDA_ARCH_PTX="$CUDA_ARCH_PTX"
  -DCUDA_FAST_MATH=ON
  -DENABLE_FAST_MATH=ON
  -DOPENCV_EXTRA_MODULES_PATH="$SRC_ROOT/opencv_contrib/modules"
  -DBUILD_opencv_cudaoptflow=ON
  -DBUILD_opencv_cudaarithm=ON
  -DBUILD_opencv_cudaimgproc=ON
  -DBUILD_opencv_cudawarping=ON
  -DBUILD_opencv_cudabgsegm=OFF
  -DBUILD_opencv_cudafeatures2d=OFF
  -DBUILD_opencv_cudafilters=ON
  -DBUILD_opencv_cudastereo=OFF
  -DBUILD_opencv_cudacodec=OFF
  -DBUILD_opencv_cudalegacy=OFF
  -DBUILD_opencv_cudaobjdetect=OFF
  -DBUILD_TESTS=OFF
  -DBUILD_PERF_TESTS=OFF
  -DBUILD_EXAMPLES=OFF
  -DBUILD_JAVA=OFF
  -DBUILD_opencv_apps=OFF
  -DBUILD_opencv_python2=OFF
  -DBUILD_opencv_python3=ON
  -DPYTHON3_EXECUTABLE="$PYTHON_EXECUTABLE"
  -DPYTHON3_INCLUDE_DIR="$PYTHON_INCLUDE_DIR"
  -DPYTHON3_NUMPY_INCLUDE_DIRS="$NUMPY_INCLUDE"
  -DPYTHON3_PACKAGES_PATH="$PYTHON_PACKAGES_PATH"
  -DWITH_FFMPEG=ON
  -DWITH_GSTREAMER=OFF
  -DWITH_GTK=OFF
  -DWITH_QT=OFF
  -DWITH_OPENCL=OFF
  -DWITH_IPP=OFF
  -DOPENCV_GENERATE_PKGCONFIG=ON
)

if [[ -d "$CONDA_PREFIX_DETECT" ]]; then
  CMAKE_ARGS+=(-DCMAKE_PREFIX_PATH="$CONDA_PREFIX_DETECT")
fi

if [[ -n "$PYTHON_LIBRARY" && -e "$PYTHON_LIBRARY" ]]; then
  CMAKE_ARGS+=(-DPYTHON3_LIBRARY="$PYTHON_LIBRARY")
fi

cmake "${CMAKE_ARGS[@]}" "$SRC_ROOT/opencv"

echo "==> Building with $JOBS jobs..."
cmake --build . --target install -j"$JOBS"

mkdir -p "$PYTHON_PACKAGES_PATH"
# Some OpenCV installs put cv2 under lib/pythonX.Y/site-packages; normalize a symlink.
shopt -s nullglob
for cand in \
  "$INSTALL_PREFIX/lib/python${PYTHON_VERSION_MAJOR_MINOR}/site-packages" \
  "$INSTALL_PREFIX/lib/python${PYTHON_VERSION_MAJOR_MINOR}/dist-packages" \
  "$INSTALL_PREFIX/lib/python/site-packages"
do
  if [[ -d "$cand/cv2" || -f "$cand/cv2"*.so || -d "$cand/cv2.cpython"* ]]; then
    if [[ "$cand" != "$PYTHON_PACKAGES_PATH" ]]; then
      ln -sfn "$cand" "$INSTALL_PREFIX/lib/python/site-packages-actual"
    fi
    echo "==> Python cv2 package dir: $cand"
  fi
done

cat > "$INSTALL_PREFIX/env.sh" <<EOF
# Source before running motion-detector with CUDA OpenCV:
#   source $INSTALL_PREFIX/env.sh
export OPENCV_CUDA_PREFIX="$INSTALL_PREFIX"
export PATH="\$OPENCV_CUDA_PREFIX/bin:\${PATH}"
export LD_LIBRARY_PATH="\$OPENCV_CUDA_PREFIX/lib:${CONDA_PREFIX_DETECT}/lib:\${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PYTHON_PACKAGES_PATH:\${PYTHONPATH:-}"
# Prefer this build over any pip opencv-python wheel.
EOF

# Also point at alternate site-packages if present.
if [[ -L "$INSTALL_PREFIX/lib/python/site-packages-actual" ]]; then
  ACTUAL="$(readlink -f "$INSTALL_PREFIX/lib/python/site-packages-actual")"
  cat >> "$INSTALL_PREFIX/env.sh" <<EOF
export PYTHONPATH="$ACTUAL:\$PYTHONPATH"
EOF
fi

echo "==> Installed OpenCV CUDA to $INSTALL_PREFIX"
echo "    Activate with: source $INSTALL_PREFIX/env.sh"
echo "    Verify with:   python -c \"import cv2; print(cv2.__version__, cv2.cuda.getCudaEnabledDeviceCount(), hasattr(cv2.cuda,'FarnebackOpticalFlow'))\""
