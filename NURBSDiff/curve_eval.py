import numpy as np
import torch
from torch.autograd import Variable

from NURBSDiff.curve_eval_cpp import (
    backward as cpp_backward,
    forward as cpp_forward,
    pre_compute_basis as cpp_pre_compute_basis,
)
from .utils import gen_knot_vector

try:
    from NURBSDiff.curve_eval_cuda import (
        backward as cuda_backward,
        forward as cuda_forward,
        pre_compute_basis as cuda_pre_compute_basis,
    )
    CUDA_EXTENSION_AVAILABLE = True
except ImportError:
    CUDA_EXTENSION_AVAILABLE = False
    cuda_backward = None
    cuda_forward = None
    cuda_pre_compute_basis = None


torch.manual_seed(120)


class CurveEval(torch.nn.Module):
    """Differentiable NURBS curve evaluation layer."""

    def __init__(self, m, knot_v=None, dimension=3, p=2, out_dim=32, method="tc", dvc="cpp"):
        super(CurveEval, self).__init__()
        self.m = m
        self._dimension = dimension
        self.p = p

        if knot_v is not None:
            self.U = knot_v
        else:
            self.U = torch.Tensor(np.array(gen_knot_vector(self.p, self.m)))

        self.u = torch.linspace(0.0, 1.0, steps=out_dim, dtype=torch.float32)
        self.method = method
        self.dvc = dvc

        if self.dvc == "cuda":
            if not CUDA_EXTENSION_AVAILABLE:
                raise RuntimeError(
                    "NURBSDiff was installed without the curve CUDA extension. "
                    "Reinstall with CUDA available or use dvc='cpp'."
                )
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested, but PyTorch reports no CUDA device.")

            self.U = self.U.cuda()
            self.u = self.u.cuda()
            self.uspan, self.Nu = cuda_pre_compute_basis(
                self.u, self.U, m, p, out_dim, self._dimension
            )
        else:
            self.uspan, self.Nu = cpp_pre_compute_basis(
                self.u, self.U, m, p, out_dim, self._dimension
            )

    def forward(self, input):
        # input dimensions: (batch_size, no. control points, dimension + weight)
        if self.method == "cpp":
            return CurveEvalFunc.apply(
                input,
                self.uspan,
                self.Nu,
                self.u,
                self.m,
                self.p,
                self._dimension,
                self.dvc,
            )
        if self.method == "tc":
            curves = (
                self.Nu[:, 0].unsqueeze(-1)
                * input[:, (self.uspan - self.p).type(torch.LongTensor), :]
            )
            for j in range(1, self.p + 1):
                curves += (
                    self.Nu[:, j].unsqueeze(-1)
                    * input[:, (self.uspan - self.p + j).type(torch.LongTensor), :]
                )
            return curves[:, :, : self._dimension] / curves[:, :, self._dimension].unsqueeze(-1)

        raise ValueError(f"Unknown evaluation method: {self.method}")


class CurveEvalFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ctrl_pts, uspan, Nu, u, m, p, _dimension, _device):
        ctx.save_for_backward(ctrl_pts)
        ctx.uspan = uspan
        ctx.Nu = Nu
        ctx.u = u
        ctx.m = m
        ctx.p = p
        ctx._dimension = _dimension
        ctx._device = _device

        if _device == "cuda":
            curves = cuda_forward(ctrl_pts, uspan, Nu, u, m, p, _dimension)
        else:
            curves = cpp_forward(
                ctrl_pts.cpu(), uspan.cpu(), Nu.cpu(), u.cpu(), m, p, _dimension
            )

        ctx.curves = curves
        return curves[:, :, :_dimension] / curves[:, :, _dimension].unsqueeze(-1)

    @staticmethod
    def backward(ctx, grad_output):
        (ctrl_pts,) = ctx.saved_tensors
        uspan = ctx.uspan
        Nu = ctx.Nu
        u = ctx.u
        m = ctx.m
        p = ctx.p
        _device = ctx._device
        _dimension = ctx._dimension
        curves = ctx.curves

        # Chain the Cartesian-output gradient through rational
        # dehomogenization: C = Cw_xyz / Cw_w.
        weights = curves[:, :, _dimension]
        grad_cw = torch.zeros_like(curves)
        grad_cw[:, :, :_dimension] = grad_output / weights.unsqueeze(-1)
        grad_cw[:, :, _dimension] = -(
            grad_output * curves[:, :, :_dimension]
        ).sum(dim=-1) / (weights**2)

        if _device == "cuda":
            grad_ctrl_pts = cuda_backward(
                grad_cw, ctrl_pts, uspan, Nu, u, m, p, _dimension
            )
        else:
            grad_ctrl_pts = cpp_backward(
                grad_cw, ctrl_pts, uspan, Nu, u, m, p, _dimension
            )

        return Variable(grad_ctrl_pts[0]), None, None, None, None, None, None, None
