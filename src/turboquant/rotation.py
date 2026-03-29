"""Random matrix generation for TurboQuant rotation and QJL projection."""

from __future__ import annotations

import torch


def generate_rotation_matrix(
    d: int,
    seed: int | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate a Haar-distributed random orthogonal matrix via QR decomposition.

    The random rotation decorrelates vector coordinates, making each follow
    a predictable distribution (approximately N(0, 1/d) for large d) suitable
    for per-coordinate scalar quantization.
    """
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    G = torch.randn(d, d, generator=gen)
    Q, R = torch.linalg.qr(G)
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0
    Q = Q * diag_sign.unsqueeze(0)
    return Q.to(device)


def generate_qjl_matrix(
    d: int,
    m: int | None = None,
    seed: int | None = None,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate the random projection matrix S for Quantized JL transform.

    S has i.i.d. N(0, 1) entries with shape (m, d). Default m = d.
    """
    if m is None:
        m = d
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    S = torch.randn(m, d, generator=gen)
    return S.to(device)
