# NURBSDiff

NURBSDiff provides **differentiable NURBS curve and surface evaluation for PyTorch**, with compiled C++ kernels and tensor-native reference implementations for geometry fitting, optimization, and point-cloud workflows.

The package is built around rational NURBS evaluation in homogeneous coordinates and supports gradients with respect to the control points, making classical CAD geometry usable inside gradient-based learning and optimization pipelines.

## Highlights

- differentiable NURBS curve evaluation
- differentiable NURBS surface evaluation
- compiled CPU C++ kernels
- tensor-native PyTorch reference path
- forward **and gradient parity tests** between compiled and tensor-native implementations
- surface-fitting and geometry-offset examples
- legacy CUDA sources retained for continued GPU development

## Installation

### 1. Install PyTorch

Install the PyTorch build appropriate for your platform using the official PyTorch installation instructions:

https://pytorch.org/get-started/locally/

A working C++ compiler is required for the native CPU extensions.

### 2. Install NURBSDiff

From the repository root:

```bash
pip install -e .
```

The default installation builds the validated CPU C++ curve and surface extensions.

## Core usage

NURBSDiff control points use homogeneous coordinates. For a 3D rational NURBS entity, the last control-point dimension is the weight:

```text
[xw, yw, zw, w]
```

### Surface evaluation

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
    method="tc",
    dvc="cpp",
)

points = surface(control_points)
```

### Curve evaluation

```python
from NURBSDiff.curve_eval import CurveEval

curve = CurveEval(
    m=8,
    dimension=3,
    p=2,
    out_dim=128,
    method="tc",
    dvc="cpp",
)

points = curve(control_points)
```

## Two differentiable CPU paths

The evaluators expose two useful implementations under the same interface.

### Tensor-native PyTorch

```python
method="tc"
```

This path performs NURBS accumulation with PyTorch tensor operations and lets PyTorch construct the gradient automatically.

### Compiled C++

```python
method="cpp"
```

This path uses the native C++ forward/backward kernels. The custom backward includes the full rational dehomogenization chain rule, including derivatives with respect to the NURBS weights.

Both paths are checked against one another for forward output and control-point gradients in `tests/test_cpu_consistency.py`.

## Regression test

After installation:

```bash
pip install pytest
pytest tests/test_cpu_consistency.py
```

The test compares compiled C++ and tensor-native evaluation for both curves and surfaces, including batched gradient propagation.

## Experimental CUDA backend

The repository retains the original CUDA extension sources for further GPU development. They are intentionally **opt-in** rather than part of the default install path:

```bash
NURBSDIFF_BUILD_EXPERIMENTAL_CUDA=1 pip install -e .
```

This requires a CUDA-enabled PyTorch environment and a CUDA toolkit discoverable through `CUDA_HOME`. The CPU C++ and tensor-native paths are the validated backends used by the documented workflow.

## Fitting examples

The `examples/` directory contains research scripts and geometry/data assets for surface fitting and offset experiments, including:

```text
examples/NURBSSurfaceFitting.py
examples/DuckyNURBSSurfaceFitting.py
examples/Surface_Offset_Aorta.py
examples/Surface_Offset_Stabilizer.py
```

Several examples use additional packages such as:

- `pytorch3d`
- `geomdl`
- `matplotlib`
- `tqdm`

Install those packages according to the example and local PyTorch environment. PyTorch3D installation is platform- and PyTorch-version-dependent, so its official installation instructions are recommended.

## Repository structure

```text
NURBSDiff/
├── NURBSDiff/
│   ├── curve_eval.py
│   ├── surf_eval.py
│   ├── nurbs_eval.py
│   ├── utils.py
│   └── csrc/              # C++ and experimental CUDA sources
├── examples/              # fitting/offset examples and research assets
├── tests/                 # CPU forward/gradient parity tests
├── setup.py
├── LICENSE
└── README.md
```

## Notes

- The existing large example geometry assets are intentionally preserved for reproducibility.
- `NURBSDiff/nurbs_eval.py` is retained as a legacy/experimental differentiable knot-vector implementation; `curve_eval.py` and `surf_eval.py` are the primary documented evaluators.
- See `LICENSE` for licensing terms.
