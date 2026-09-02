# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark Stiefel retractions in Iso: latency, throughput, and numerical precision."""

from collections.abc import Callable
from typing import Any

import torch
import triton

from emerging_optimizers import utils
from emerging_optimizers.riemannian_optimizers.isospectral import (
    _cayley_retraction,
    _newton_schulz_retraction,
    _polar_retraction,
    _qr_retraction,
)


def max_err(x: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """Return (max abs err, max rel err) of x against ref."""
    ref_f = ref.float()
    assert ref_f.abs().amax() > 0.0, "reference is all zeros"
    diff = (x.float() - ref_f).abs()
    return diff.max().item(), (diff / ref_f.abs().clamp_min(1e-8)).max().item()


def orth_err(q: torch.Tensor) -> float:
    """Return max absolute deviation of Q.mT @ Q from identity."""
    k = q.shape[-1]
    eye = torch.eye(k, device=q.device, dtype=q.dtype)
    return (q.mT @ q - eye).abs().max().item()


def bench(fn: Callable[[], Any], warmup: int = 25, rep: int = 100) -> float:
    """Time fn with triton.testing.do_bench (returns execution time in ms)."""
    return triton.testing.do_bench(fn, warmup=warmup, rep=rep)


def run_case(
    m: int,
    n: int,
    lr: float = 1e-3,
    device: str = "cuda",
    fp32_prec: utils.FP32MatmulPrecT = "high",
) -> None:
    """Benchmark retractions for a single parameter tensor of shape (M, N)."""
    torch.manual_seed(0)
    k = min(m, n)

    # Initialize factor matrices U (M, K) and V (N, K) on the Stiefel manifold
    raw_u = torch.randn(m, k, device=device, dtype=torch.float32)
    raw_v = torch.randn(n, k, device=device, dtype=torch.float32)
    u_init, _ = torch.linalg.qr(raw_u, mode="reduced")
    v_init, _ = torch.linalg.qr(raw_v, mode="reduced")

    # Synthetic momentum updates
    mom_u = torch.randn_like(u_init)
    mom_v = torch.randn_like(v_init)

    # --- Correctness & Numerical Precision ---
    # Reference polar retraction (exact SVD: argmin_{Q in St} ||X - Q||_F)
    ref_u = _polar_retraction(u_init, mom_u, lr)
    ref_v = _polar_retraction(v_init, mom_v, lr)

    qr_u = _qr_retraction(u_init, mom_u, lr)
    cayley_u = _cayley_retraction(u_init, mom_u, lr)
    ns5_u = _newton_schulz_retraction(u_init, mom_u, lr, num_ns_steps=5)
    ns8_u = _newton_schulz_retraction(u_init, mom_u, lr, num_ns_steps=8)

    # Orthogonality deviation: ||Q^T Q - I||_max
    orth_polar = orth_err(ref_u)
    orth_qr = orth_err(qr_u)
    orth_cayley = orth_err(cayley_u)
    orth_ns5 = orth_err(ns5_u)
    orth_ns8 = orth_err(ns8_u)

    # Error vs exact polar factor
    err_ns5_abs, err_ns5_rel = max_err(ns5_u, ref_u)
    err_ns8_abs, err_ns8_rel = max_err(ns8_u, ref_u)

    # --- Latency Benchmarking (Factor U + Factor V) ---
    with utils.fp32_matmul_precision(fp32_prec):
        t_polar = bench(lambda: (_polar_retraction(u_init, mom_u, lr), _polar_retraction(v_init, mom_v, lr)))
        t_qr = bench(lambda: (_qr_retraction(u_init, mom_u, lr), _qr_retraction(v_init, mom_v, lr)))
        t_cayley = bench(lambda: (_cayley_retraction(u_init, mom_u, lr), _cayley_retraction(v_init, mom_v, lr)))
        t_ns5 = bench(
            lambda: (
                _newton_schulz_retraction(u_init, mom_u, lr, num_ns_steps=5),
                _newton_schulz_retraction(v_init, mom_v, lr, num_ns_steps=5),
            )
        )
        t_ns8 = bench(
            lambda: (
                _newton_schulz_retraction(u_init, mom_u, lr, num_ns_steps=8),
                _newton_schulz_retraction(v_init, mom_v, lr, num_ns_steps=8),
            )
        )

    tag = f"M={m:<5d} N={n:<5d} K={k:<5d}"
    speedup_ns5_vs_polar = t_polar / t_ns5

    print(
        f"{tag} | polar(svd) {t_polar:8.3f} ms | qr {t_qr:7.3f} ms | cayley {t_cayley:7.3f} ms | "
        f"ns(steps=5) {t_ns5:7.3f} ms | ns(steps=8) {t_ns8:7.3f} ms | "
        f"ns5 speedup vs svd: {speedup_ns5_vs_polar:6.2f}x"
    )
    print(
        f"{'':>4}orth drift ||Q^T Q - I||: svd={orth_polar:.2e} | qr={orth_qr:.2e} | "
        f"cayley={orth_cayley:.2e} | ns5={orth_ns5:.2e} | ns8={orth_ns8:.2e}"
    )
    print(
        f"{'':>4}ns vs polar factor: ns5 abs={err_ns5_abs:.2e} rel={err_ns5_rel:.2e} | "
        f"ns8 abs={err_ns8_abs:.2e} rel={err_ns8_rel:.2e}\n"
    )


def main() -> None:
    """Run benchmark across square and rectangular transformer projection dimensions."""
    torch.cuda.init()
    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}")

    # Shapes representative of LLM attention and MLP layers:
    # (1024, 1024) - Small/Medium Hidden Dim
    # (2048, 2048) - 7B/8B Attention Projection
    # (4096, 4096) - 8B/70B Attention Projection
    # (4096, 2048) - Non-Square Feed-Forward / Bottleneck Projection
    # (8192, 2048) - Non-Square SwiGLU MLP Projection
    cases = [
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
        (4096, 2048),
        (8192, 2048),
    ]

    print("=== Iso Stiefel Retraction Benchmark Suite ===")
    for m, n in cases:
        run_case(m, n)


if __name__ == "__main__":
    main()
