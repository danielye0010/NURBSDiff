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


# The validated CPU C++ backend is the default install path. The repository also
# retains its legacy CUDA sources, but they are opt-in until that backend has a
# dedicated CUDA regression environment.
build_cuda = os.environ.get("NURBSDIFF_BUILD_EXPERIMENTAL_CUDA", "0") == "1"
if build_cuda and CUDA_HOME is None:
    raise RuntimeError(
        "NURBSDIFF_BUILD_EXPERIMENTAL_CUDA=1 was requested, but CUDA_HOME "
        "could not be located."
    )

extensions = cpp_extensions()
if build_cuda:
    extensions.extend(cuda_extensions())

print(f"Building NURBSDiff experimental CUDA extensions: {build_cuda}")

setup(
    name="NURBSDiff",
    packages=find_packages(),
    ext_modules=extensions,
    cmdclass={"build_ext": BuildExtension},
)
