"""
HSIC utilities adapted for HSIC-SPIB.

Reference implementation ideas from the HSIC Bottleneck repo
(source/hsicbt/math/hsic.py), rewritten for standard PyTorch autograd:
- gradients flow through Z (latent), not through X/Y kernels or bandwidth.
"""
import torch


def pairwise_squared_distances(x):
    """Compute pairwise squared Euclidean distances. x: (B, D)."""
    x_norm = (x * x).sum(dim=1, keepdim=True)
    dist = x_norm + x_norm.t() - 2.0 * torch.mm(x, x.t())
    return dist.clamp(min=0.0)


def median_heuristic_bandwidth(x, min_bandwidth=1e-2):
    """Median pairwise distance heuristic; returned value is detached."""
    with torch.no_grad():
        dist = pairwise_squared_distances(x)
        # use upper triangle (exclude diagonal)
        b = dist.size(0)
        if b < 2:
            return torch.tensor(1.0, device=x.device, dtype=x.dtype)
        tri = dist[torch.triu(torch.ones(b, b, device=x.device, dtype=torch.bool), diagonal=1)]
        med = torch.median(torch.sqrt(tri.clamp(min=0.0)))
        if not torch.isfinite(med) or med <= 0:
            med = torch.mean(torch.sqrt(tri.clamp(min=0.0)))
        if (not torch.isfinite(med)) or med < min_bandwidth:
            med = torch.tensor(min_bandwidth, device=x.device, dtype=x.dtype)
        return med.detach()


def rbf_kernel(x, sigma=None, detach_kernel=False):
    """
    RBF / Gaussian kernel matrix.
    If sigma is None, use median heuristic (detached).
    """
    dist = pairwise_squared_distances(x)
    if sigma is None:
        sigma = median_heuristic_bandwidth(x)
    else:
        sigma = torch.as_tensor(sigma, device=x.device, dtype=x.dtype).detach()
        if float(sigma) <= 0:
            sigma = median_heuristic_bandwidth(x)

    # scale by feature dim similar to HSIC-bottleneck variance convention
    dim = max(x.size(1), 1)
    variance = 2.0 * sigma * sigma * dim
    kx = torch.exp(-dist / variance)
    if detach_kernel:
        kx = kx.detach()
    return kx, sigma


def linear_kernel(x, detach_kernel=False):
    kx = torch.mm(x, x.t())
    if detach_kernel:
        kx = kx.detach()
    return kx


def delta_kernel(y, detach_kernel=True):
    """
    Label kernel for one-hot or integer labels.
    For one-hot (B, C): K_ij = 1 if same class else 0.
    Always returns a floating-point kernel for matmul with centering matrix.
    """
    float_dtype = torch.get_default_dtype()
    if y.dim() == 1:
        y_idx = y.long().view(-1, 1)
        kx = (y_idx == y_idx.t()).to(dtype=float_dtype)
    else:
        # one-hot / soft labels: same argmax class, or soft inner product
        y_sum = y.sum(dim=1)
        ones = torch.ones(y.size(0), device=y.device, dtype=y.dtype)
        if torch.allclose(y_sum, ones, atol=1e-3):
            y_idx = y.argmax(dim=1, keepdim=True)
            # bool equality -> float (avoid long/bool matmul with H)
            kx = (y_idx == y_idx.t()).to(dtype=float_dtype)
        else:
            kx = torch.mm(y.float(), y.float().t())
    if detach_kernel:
        kx = kx.detach()
    return kx


def center_kernel(k):
    """Centered kernel: HKH with H = I - 1/n 11^T."""
    # Ensure float dtype: delta/label kernels may arrive as bool/long.
    if not k.is_floating_point():
        k = k.float()
    n = k.size(0)
    unit = torch.ones(n, n, device=k.device, dtype=k.dtype) / n
    h = torch.eye(n, device=k.device, dtype=k.dtype) - unit
    return torch.mm(torch.mm(h, k), h)


def hsic_unnormalized(kx, ky):
    """Empirical HSIC = mean of elementwise product of centered kernels."""
    kxc = center_kernel(kx)
    kyc = center_kernel(ky)
    return torch.mean(kxc * kyc)


def hsic_normalized(kx, ky, eps=1e-6):
    """Normalized HSIC / centered kernel alignment."""
    pxy = hsic_unnormalized(kx, ky)
    px = torch.sqrt(hsic_unnormalized(kx, kx).clamp(min=eps))
    py = torch.sqrt(hsic_unnormalized(ky, ky).clamp(min=eps))
    return pxy / (px * py)


def build_kernel(x, kernel_type="rbf", sigma=None, detach_kernel=False):
    kernel_type = (kernel_type or "rbf").lower()
    if kernel_type == "rbf":
        return rbf_kernel(x, sigma=sigma, detach_kernel=detach_kernel)
    if kernel_type == "linear":
        return linear_kernel(x, detach_kernel=detach_kernel), None
    if kernel_type in ("delta", "label"):
        return delta_kernel(x, detach_kernel=True), None
    raise ValueError("Unknown kernel_type: {}".format(kernel_type))


def compute_hsic(z, other, kernel_z="rbf", kernel_other="rbf",
                 sigma_z=None, sigma_other=None, normalized=True,
                 detach_other=True):
    """
    Compute HSIC(Z, other). Gradients flow through z only when other is detached.

    Parameters
    ----------
    z : (B, Dz) latent (keep grad)
    other : (B, D) input X or labels Y
    """
    if z.size(0) != other.size(0):
        raise ValueError("Batch size mismatch in HSIC: {} vs {}".format(z.size(0), other.size(0)))
    if z.size(0) < 2:
        zero = z.new_zeros(())
        return zero, None, None

    kz, sz = build_kernel(z, kernel_type=kernel_z, sigma=sigma_z, detach_kernel=False)
    ko, so = build_kernel(other, kernel_type=kernel_other, sigma=sigma_other,
                          detach_kernel=detach_other)
    if normalized:
        val = hsic_normalized(kz, ko)
    else:
        val = hsic_unnormalized(kz, ko)
    return val, sz, so
