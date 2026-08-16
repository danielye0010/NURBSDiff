# NURBSDiff

NURBSDiff provides differentiable NURBS curve and surface evaluation layers for PyTorch, together with examples for fitting NURBS geometry to point-cloud and target geometry data.

The core evaluators combine PyTorch autograd with compiled C++ extensions and optional CUDA extensions.

## Features

- differentiable NURBS curve evaluation
- differentiable NURBS surface evaluation
- CPU C++ extensions
- optional CUDA extensions
- example workflows for NURBS surface fitting and geometry offsetting

## Installation

### 1. Install PyTorch

Install a PyTorch build appropriate for your operating system and, if applicable, CUDA environment using the official PyTorch installation instructions:

https://pytorch.org/get-started/locally/

A working C++ compiler is required to build the native extensions. Building the optional CUDA extensions additionally requires a CUDA-capable PyTorch environment and CUDA build toolchain.

### 2. Install NURBSDiff

From the repository root:

```bash
pip install -e .
```

The package always builds the CPU C++ evaluators. When PyTorch reports CUDA availability, the CUDA curve and surface extensions are built as well.

To explicitly request a CPU-only build on a CUDA-capable machine:

```bash
NURBSDIFF_FORCE_CPU=1 pip install -e .
```

On Windows, set the same environment variable using the syntax appropriate for your shell before running `pip install -e .`.

## Core Usage

### Surface evaluation on CPU

```python
from NURBSDiff.surf_eval import SurfEval

surface = SurfEval(
    m=14,
    n=13,
    dimension=3,
    p=3,
    q=3,
    out_dim_u=128,
    out_dim_v=128,
    dvc="cpp",
)
```

### Surface evaluation with CUDA

If the package was built with CUDA extensions and PyTorch can access a CUDA device:

```python
surface = SurfEval(
    m=14,
    n=13,
    dimension=3,
    p=3,
    q=3,
    out_dim_u=128,
    out_dim_v=128,
    dvc="cuda",
)
```

If `dvc="cuda"` is requested without an available CUDA extension/device, NURBSDiff raises an explicit error rather than silently switching execution back to CPU.

### Curve evaluation

```python
from NURBSDiff.curve_eval import CurveEval

curve = CurveEval(
    m=8,
    dimension=3,
    p=2,
    out_dim=128,
    dvc="cpp",
)
```

`CurveEval` retains CUDA as its historical default device setting; use `dvc="cpp"` explicitly for CPU execution.

## Fitting Examples

The `examples/` directory contains research scripts and geometry/data assets used for fitting and offset experiments. For example:

```text
examples/NURBSSurfaceFitting.py
examples/DuckyNURBSSurfaceFitting.py
examples/Surface_Offset_Aorta.py
examples/Surface_Offset_Stabilizer.py
```

These fitting examples use additional packages beyond the core evaluator, including packages such as:

- `pytorch3d`
- `geomdl`
- `matplotlib`
- `tqdm`

Install those dependencies according to the requirements of the example and your PyTorch/CUDA environment. PyTorch3D installation is platform- and PyTorch-version-dependent, so its official installation instructions should be used rather than assuming one universal pip command.

## Repository Structure

```text
NURBSDiff/
├── NURBSDiff/
│   ├── curve_eval.py
│   ├── surf_eval.py
│   ├── nurbs_eval.py
│   ├── utils.py
│   └── csrc/              # C++ and CUDA extension sources
├── examples/              # fitting/offset examples and research assets
├── setup.py
├── LICENSE
└── README.md
```

## Notes

- Large example assets are currently stored directly in the repository; they are preserved here to avoid changing example reproducibility as part of packaging cleanup.
- `NURBSDiff/nurbs_eval.py` is retained as a legacy/experimental differentiable knot-vector implementation and is not the primary evaluator used by the documented examples.
- The package includes a license file; see `LICENSE` for terms.
