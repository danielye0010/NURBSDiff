import os

from setuptools import find_packages, setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)


CSRC_DIR = "NURBSDiff/csrc"


def cpp_extensions():
    return [
        CppExtension(
            name="NURBSDiff.curve_eval_cpp",
            sources=[
                f"{CSRC_DIR}/curve_eval.cpp",
                f"{CSRC_DIR}/utils.cpp",
            ],
            include_dirs=[CSRC_DIR],
        ),
        CppExtension(
            name="NURBSDiff.surf_eval_cpp",
            sources=[
                f"{CSRC_DIR}/surf_eval.cpp",
                f"{CSRC_DIR}/utils.cpp",
            ],
            include_dirs=[CSRC_DIR],
        ),
    ]


def cuda_extensions():
    return [
        CUDAExtension(
            name="NURBSDiff.curve_eval_cuda",
            sources=[
                f"{CSRC_DIR}/curve_eval_cuda.cpp",
                f"{CSRC_DIR}/curve_eval_cuda_kernel.cu",
            ],
            include_dirs=[CSRC_DIR],
        ),
        CUDAExtension(
            name="NURBSDiff.surf_eval_cuda",
            sources=[
                f"{CSRC_DIR}/surf_eval_cuda.cpp",
                f"{CSRC_DIR}/surf_eval_cuda_kernel.cu",
            ],
            include_dirs=[CSRC_DIR],
        ),
    ]


# CPU extensions are always built. CUDA extensions are added when PyTorch can
# locate a CUDA toolkit (CUDA_HOME), even on headless build machines without a
# visible GPU. Set NURBSDIFF_FORCE_CPU=1 to explicitly request a CPU-only build.
force_cpu = os.environ.get("NURBSDIFF_FORCE_CPU", "0") == "1"
build_cuda = CUDA_HOME is not None and not force_cpu

extensions = cpp_extensions()
if build_cuda:
    extensions.extend(cuda_extensions())

print(f"Building NURBSDiff with CUDA extensions: {build_cuda}")

setup(
    name="NURBSDiff",
    packages=find_packages(),
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExtension},
)
