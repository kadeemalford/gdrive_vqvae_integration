"""Independent subspace VQ, with explicit interleaved symbols and shared codebooks."""
import torch
import torch.nn.functional as F


def validate_codebooks(codebooks):
    if codebooks.ndim != 3 or min(codebooks.shape) < 1 or not torch.isfinite(codebooks).all():
        raise ValueError('Expected finite [groups, alphabet, subspace] codebooks')
    groups, alphabet, width = codebooks.shape
    if alphabet < 2 or alphabet > 32768 or alphabet & (alphabet - 1):
        raise ValueError('Expected a power-of-two alphabet')
    return groups, alphabet, width


def validate_transform(basis, center, width, device, dtype):
    if basis is None and center is None:
        return
    if (basis is None or center is None or basis.shape != (width, width) or center.shape != (width,)
            or basis.device != device or center.device != device or basis.dtype != dtype or center.dtype != dtype
            or not torch.isfinite(basis).all() or not torch.isfinite(center).all()):
        raise ValueError('Invalid product rotation/center')
    if not torch.allclose(basis.T @ basis, torch.eye(width, device=device, dtype=dtype), atol=2e-5, rtol=2e-5):
        raise ValueError('Product basis must be orthogonal, without whitening')


def replay_product(symbols, latent_shape, codebooks, basis=None, center=None):
    groups, alphabet, width = validate_codebooks(codebooks)
    if (len(latent_shape) != 2 or min(latent_shape) < 1 or symbols.ndim != 2
            or symbols.shape != (latent_shape[0] * latent_shape[1], groups)
            or symbols.dtype != torch.long or symbols.device != codebooks.device
            or torch.any(symbols < 0) or torch.any(symbols >= alphabet)):
        raise ValueError('Invalid product indices/geometry')
    flat = torch.cat([F.embedding(symbols[:, g], codebooks[g]) for g in range(groups)], dim=1)
    validate_transform(basis, center, groups * width, codebooks.device, codebooks.dtype)
    if basis is not None:
        flat = flat @ basis.T + center
    return flat.view(1, *latent_shape, groups * width).permute(0, 3, 1, 2).contiguous()


def quantize_product(z, codebooks, basis=None, center=None):
    groups, _, width = validate_codebooks(codebooks)
    if z.ndim != 4 or z.shape[0] != 1 or z.shape[1] != groups * width or z.device != codebooks.device:
        raise ValueError('Product VQ requires one matching latent grid')
    flat = z.permute(0, 2, 3, 1).reshape(-1, groups * width)
    validate_transform(basis, center, groups * width, codebooks.device, codebooks.dtype)
    if basis is not None:
        flat = (flat - center) @ basis
    flat = flat.reshape(-1, groups, width)
    symbols = []
    for group in range(groups):
        values, centers = flat[:, group], codebooks[group]
        distance = values.square().sum(1, keepdim=True) - 2 * values @ centers.T + centers.square().sum(1)
        symbols.append(distance.argmin(1))
    symbols = torch.stack(symbols, dim=1)
    return symbols, replay_product(symbols, tuple(z.shape[-2:]), codebooks, basis, center)
