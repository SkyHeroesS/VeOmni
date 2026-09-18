# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek-V4 sparse MQA on AMD via Primus' Triton-v2 sparse-MLA kernels.

The ROCm counterpart to ``tilelang_sparse_mla``: same compact index-list
contract, same per-head learnable sink, but MFMA Triton kernels that run on any
MFMA-capable arch instead of requiring NVIDIA SM90+. Primus is an external
dependency and is imported lazily, so importing VeOmni never needs it.

Primus' kernels take one flat latent pool and a token-major index list, which is
a different layout from the ``[B, S, ...]`` tensors DeepSeek-V4 attention holds.
This module owns that translation:

* ``d_qk = kv_lora_rank + rope_rank`` is the kernel's contract. DeepSeek-V4
  carries no separate rope rank here, so ``ROPE_PAD`` zero channels are appended
  and sliced back off in the backward.
* the index list is rebased from per-sample to pool-global rows, preserving
  ``-1`` for invalid slots.
* ``topk`` is padded to ``TOPK_ALIGN`` with ``-1``.
"""

from __future__ import annotations

import os

import torch


# The kernel derives ``rope_rank = d_qk - kv_lora_rank`` and indexes the rope
# slice unconditionally, so the operands carry a zero rope block of this width.
ROPE_PAD = 64
# The backward's inverted-topk gather is built per ``R_CHUNK``; an unpadded topk
# leaves a partial chunk the CSR builder rejects.
TOPK_ALIGN = 64
# Primus' backward stages its LDS operand tiles across Triton pipeline stages,
# which needs 160 KiB at the default schedule -- fine on CDNA4, over budget on
# the 64 KiB of CDNA3. Both knobs are Primus' own; see ``_apply_lds_budget``.
_LDS_BUDGET_KNOBS = {
    "PRIMUS_DSA_BWD_NUM_STAGES": "1",
    "PRIMUS_DSA_DKV_SAFE": "1",
}
_SMALL_LDS_BYTES = 64 * 1024


def _apply_lds_budget(device: torch.device) -> None:
    """Pick Primus' small-LDS backward schedule when the device needs it.

    Measured on MI308X (gfx942, 64 KiB LDS) at S=4096 / H=64 / topk=640, the
    default schedule asks for 163840 B and ``PRIMUS_DSA_DKV_SAFE`` alone still
    asks for 73728 B; both raise ``triton.OutOfResources``. Disabling the
    pipeline staging is what brings it under budget (27.59 ms), and adding the
    narrow dKV tile is a further 12% (24.18 ms).

    ``setdefault`` rather than assignment: these are Primus' documented knobs,
    so an explicitly exported value wins. Only devices that need it are
    touched, leaving the faster default schedule in place on larger LDS.
    """
    properties = torch.cuda.get_device_properties(device)
    if getattr(properties, "shared_memory_per_block", 0) > _SMALL_LDS_BYTES:
        return
    for name, value in _LDS_BUDGET_KNOBS.items():
        os.environ.setdefault(name, value)


def _load_kernels():
    try:
        from primus.backends.megatron.core.transformer.v4_attention_kernels._triton_v2 import (
            sparse_mla_bwd_v4_triton,
            sparse_mla_fwd_v4_triton,
        )
    except ImportError as exc:
        raise ImportError(
            "dsa_attention_implementation='primus_triton_v2' needs the Primus source tree on "
            "PYTHONPATH for its DeepSeek-V4 sparse-MLA kernels (primus.backends.megatron.core."
            "transformer.v4_attention_kernels._triton_v2)."
        ) from exc
    return sparse_mla_fwd_v4_triton, sparse_mla_bwd_v4_triton


def _pad_rope(x: torch.Tensor) -> torch.Tensor:
    """Append ``ROPE_PAD`` zero channels to a ``[..., d]`` operand."""
    pad = x.new_zeros((*x.shape[:-1], ROPE_PAD))
    return torch.cat([x, pad], dim=-1).contiguous()


def _to_pool_indices(topk_indices: torch.Tensor, kv_len: int) -> torch.Tensor:
    """Rebase ``[B, S, K]`` per-sample indices onto one ``B * kv_len`` pool."""
    batch = topk_indices.shape[0]
    offsets = torch.arange(batch, device=topk_indices.device, dtype=topk_indices.dtype).view(batch, 1, 1) * kv_len
    global_indices = torch.where(topk_indices >= 0, topk_indices + offsets, torch.full_like(topk_indices, -1))
    global_indices = global_indices.reshape(-1, global_indices.shape[-1]).to(torch.int32)
    pad = (-global_indices.shape[-1]) % TOPK_ALIGN
    if pad:
        global_indices = torch.cat(
            [
                global_indices,
                global_indices.new_full((global_indices.shape[0], pad), -1),
            ],
            dim=-1,
        )
    return global_indices.contiguous()


class _PrimusSparseMLA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, kv, attn_sink, topk_indices, scaling):
        batch, seq_len, num_heads, head_dim = query.shape
        kv_len = kv.shape[1]
        forward, backward = _load_kernels()
        _apply_lds_budget(query.device)

        q_kernel = _pad_rope(query.reshape(batch * seq_len, num_heads, head_dim))
        kv_kernel = _pad_rope(kv.reshape(batch * kv_len, 1, head_dim))
        pool_indices = _to_pool_indices(topk_indices, kv_len)

        output, lse = forward(
            q_kernel,
            kv_kernel,
            pool_indices,
            attn_sink=attn_sink,
            kv_lora_rank=head_dim,
            scale=float(scaling),
        )
        ctx.save_for_backward(q_kernel, kv_kernel, output, lse, pool_indices, attn_sink)
        ctx.backward_fn = backward
        ctx.shape = (batch, seq_len, num_heads, head_dim, kv_len)
        ctx.scaling = float(scaling)
        return output.reshape(batch, seq_len, num_heads, head_dim)

    @staticmethod
    def backward(ctx, grad_output):
        q_kernel, kv_kernel, output, lse, pool_indices, attn_sink = ctx.saved_tensors
        batch, seq_len, num_heads, head_dim, kv_len = ctx.shape
        grad = grad_output.reshape(batch * seq_len, num_heads, head_dim).contiguous()
        d_query, d_kv, d_sink = ctx.backward_fn(
            q_kernel,
            kv_kernel,
            output,
            grad,
            pool_indices,
            lse,
            attn_sink=attn_sink,
            kv_lora_rank=head_dim,
            scale=ctx.scaling,
        )
        # Drop the zero rope block the operands were padded with.
        d_query = d_query[..., :head_dim].reshape(batch, seq_len, num_heads, head_dim)
        d_kv = d_kv[:, 0, :head_dim].reshape(batch, kv_len, head_dim)
        return d_query, d_kv, d_sink, None, None


def sparse_attn_primus_triton_v2(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    sm_scale: float,
) -> torch.Tensor:
    """DeepSeek-V4 sparse MQA over a compact index list, on AMD MFMA hardware.

    Args:
        q: ``[B, S, H, D]`` bfloat16 queries.
        kv: ``[B, S_kv, D]`` bfloat16 shared latent; K and V are the same tensor.
        attn_sink: ``[H]`` float32 per-head learnable sink.
        topk_idxs: ``[B, S, K]`` int32 candidate rows into ``kv``, ``-1`` for
            invalid slots. Spans the sliding window and the compressed entries.
        sm_scale: softmax scale.

    Returns:
        ``[B, S, H, D]`` attention output in ``q``'s dtype.
    """
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise ValueError(f"primus_triton_v2 requires bfloat16 q/kv, got q={q.dtype}, kv={kv.dtype}")
    if q.shape[-1] != kv.shape[-1]:
        raise ValueError(f"primus_triton_v2 requires matching head dims, got q={q.shape[-1]}, kv={kv.shape[-1]}")
    if attn_sink.dtype != torch.float32 or attn_sink.shape != (q.shape[-2],):
        raise ValueError(
            "primus_triton_v2 requires a float32 sink with one value per query head, got "
            f"dtype={attn_sink.dtype}, shape={tuple(attn_sink.shape)}"
        )
    return _PrimusSparseMLA.apply(q.contiguous(), kv.contiguous(), attn_sink.contiguous(), topk_idxs, float(sm_scale))


__all__ = ["sparse_attn_primus_triton_v2"]
