import torch

from NURBSDiff.curve_eval import CurveEval
from NURBSDiff.surf_eval import SurfEval


def _homogeneous_control_points(*shape):
    ctrl = torch.rand(*shape, dtype=torch.float32)
    ctrl[..., -1] = 0.2 + 0.8 * ctrl[..., -1]
    return ctrl


def _compare_output_and_gradient(compiled, tensor_native, control_points):
    compiled_ctrl = control_points.clone().requires_grad_(True)
    native_ctrl = control_points.clone().requires_grad_(True)

    compiled_out = compiled(compiled_ctrl)
    native_out = tensor_native(native_ctrl)
    torch.testing.assert_close(compiled_out, native_out, rtol=1e-5, atol=1e-6)

    projection = torch.randn_like(compiled_out)
    (compiled_out * projection).sum().backward()
    (native_out * projection).sum().backward()

    torch.testing.assert_close(
        compiled_ctrl.grad,
        native_ctrl.grad,
        rtol=1e-5,
        atol=1e-6,
    )


def test_curve_cpp_matches_tensor_native_forward_and_gradient():
    torch.manual_seed(7)
    m, p, dimension, out_dim = 6, 3, 3, 11
    compiled = CurveEval(
        m=m,
        p=p,
        dimension=dimension,
        out_dim=out_dim,
        method="cpp",
        dvc="cpp",
    )
    tensor_native = CurveEval(
        m=m,
        p=p,
        dimension=dimension,
        out_dim=out_dim,
        method="tc",
        dvc="cpp",
    )
    control = _homogeneous_control_points(2, m, dimension + 1)
    _compare_output_and_gradient(compiled, tensor_native, control)


def test_surface_cpp_matches_tensor_native_forward_and_gradient():
    torch.manual_seed(11)
    m, n, p, q, dimension = 5, 6, 2, 2, 3
    out_u, out_v = 7, 8
    compiled = SurfEval(
        m=m,
        n=n,
        p=p,
        q=q,
        dimension=dimension,
        out_dim_u=out_u,
        out_dim_v=out_v,
        method="cpp",
        dvc="cpp",
    )
    tensor_native = SurfEval(
        m=m,
        n=n,
        p=p,
        q=q,
        dimension=dimension,
        out_dim_u=out_u,
        out_dim_v=out_v,
        method="tc",
        dvc="cpp",
    )
    control = _homogeneous_control_points(2, m, n, dimension + 1)
    _compare_output_and_gradient(compiled, tensor_native, control)
