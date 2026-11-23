"""Pure-NumPy neural inference engines matching PyTorch 2026 baseline networks.

Allows ROS 2 / Gazebo environments without PyTorch (e.g., WSL2 Humble) and
strict provenance audits to load packaged `.npz`/`.pt` weights and execute
bit-compatible eval-mode forward passes for Choi et al. (IJCAI-26 PGMPO) and
Wu et al. (EJOR 2026 GRU-Attention).
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Optional

import numpy as np

_SHA256_CACHE: dict[str, str] = {}
_WEIGHTS_CACHE: dict[str, dict[str, np.ndarray]] = {}


def file_sha256(path: Path) -> str:
    """Return cached hex SHA-256 digest for an existing file."""
    p = Path(path).resolve()
    key = f"{p}:{p.stat().st_mtime_ns}:{p.stat().st_size}"
    if key not in _SHA256_CACHE:
        _SHA256_CACHE[key] = hashlib.sha256(p.read_bytes()).hexdigest()
    return _SHA256_CACHE[key]


def elu(x: np.ndarray) -> np.ndarray:
    """Pure-NumPy ELU activation matching PyTorch `nn.ELU(alpha=1.0)`."""
    return np.where(x > 0.0, x, np.expm1(np.clip(x, -20.0, 0.0))).astype(np.float32, copy=False)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    x_clip = np.clip(x, -40.0, 40.0)
    return (1.0 / (1.0 + np.exp(-x_clip))).astype(np.float32, copy=False)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along `axis`."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(shifted)
    return (ex / np.maximum(1e-12, np.sum(ex, axis=axis, keepdims=True))).astype(np.float32, copy=False)


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Pure-NumPy LayerNorm over the trailing feature dimension."""
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    return (((x - mu) / np.sqrt(var + eps)) * w + b).astype(np.float32, copy=False)


def mha_batch(
    q_in: np.ndarray,
    kv_in: np.ndarray,
    in_proj_w: np.ndarray,
    in_proj_b: np.ndarray,
    out_proj_w: np.ndarray,
    out_proj_b: np.ndarray,
    n_heads: int = 4,
) -> np.ndarray:
    """Exact pure-NumPy batched Multi-Head Attention matching `nn.MultiheadAttention(batch_first=True)`."""
    B, Lq, D = q_in.shape
    Lk = kv_in.shape[1]
    dh = D // n_heads
    wq, wk, wv = in_proj_w[:D], in_proj_w[D : 2 * D], in_proj_w[2 * D :]
    bq, bk, bv = in_proj_b[:D], in_proj_b[D : 2 * D], in_proj_b[2 * D :]

    q = (q_in @ wq.T + bq).reshape(B, Lq, n_heads, dh).transpose(0, 2, 1, 3)
    k = (kv_in @ wk.T + bk).reshape(B, Lk, n_heads, dh).transpose(0, 2, 1, 3)
    v = (kv_in @ wv.T + bv).reshape(B, Lk, n_heads, dh).transpose(0, 2, 1, 3)

    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / math.sqrt(float(dh))
    att = softmax(scores, axis=-1)
    out = np.matmul(att, v).transpose(0, 2, 1, 3).reshape(B, Lq, D)
    return (out @ out_proj_w.T + out_proj_b).astype(np.float32, copy=False)


def resolve_numpy_checkpoint_path(ckpt_path: Path, *, prefer_npz: bool = True) -> Path:
    """Return `.npz` path first (when `prefer_npz=True`) if present directly or as sibling of `.pt`."""
    p = Path(ckpt_path)
    if p.suffix.lower() == ".npz":
        return p
    npz_sib = p.with_suffix(".npz")
    if prefer_npz and npz_sib.is_file():
        return npz_sib
    if p.is_file():
        return p
    return p


def load_numpy_state_dict(
    ckpt_path: Path,
    *,
    prefer_npz: bool = True,
) -> tuple[dict[str, np.ndarray], Path, str]:
    """Load state dict into float32 NumPy arrays; prefer matching `.npz` to avoid importing PyTorch."""
    p = Path(ckpt_path)
    npz_sib = p.with_suffix(".npz")
    if p.suffix.lower() == ".npz":
        load_p = p
    elif prefer_npz and npz_sib.is_file():
        load_p = npz_sib
    else:
        load_p = p

    if not load_p.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    resolved = load_p.resolve()
    sha = file_sha256(resolved)
    cache_key = f"{resolved}:{sha}"
    if cache_key in _WEIGHTS_CACHE:
        return _WEIGHTS_CACHE[cache_key], resolved, sha

    if resolved.suffix.lower() == ".npz":
        data = np.load(str(resolved), allow_pickle=False)
        weights = {str(k): np.asarray(data[k], dtype=np.float32) for k in data.files}
    else:
        import torch

        sd = torch.load(resolved, map_location="cpu", weights_only=True)
        raw = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
        weights = {str(k): v.detach().cpu().numpy().astype(np.float32) for k, v in raw.items()}

    _WEIGHTS_CACHE[cache_key] = weights
    return weights, resolved, sha


class NumpyChoiIJCAI26PGMPONet:
    """Exact pure-NumPy inference model matching `ChoiIJCAI26PGMPONet.eval()`."""

    def __init__(self, weights: dict[str, np.ndarray], *, hidden_dim: int = 64, bit_dim: int = 4, n_heads: int = 4) -> None:
        self.w = weights
        self.hidden_dim = int(hidden_dim)
        self.bit_dim = int(bit_dim)
        self.n_heads = int(n_heads)
        self.n_layers = sum(1 for k in weights if k.startswith("omb_layers.") and k.endswith(".out_proj.weight"))

    @staticmethod
    def bit_vectors(c_policies: int = 8, bit_dim: int = 4) -> np.ndarray:
        """Build deterministic [C, bit_dim] binary policy conditioning vectors."""
        bits = [[(c >> b) & 1 for b in range(bit_dim)] for c in range(c_policies)]
        return np.asarray(bits, dtype=np.float32)

    def encode_attention(
        self,
        op_x: np.ndarray,
        res_x: np.ndarray,
        z_bit: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run L layers of Bit-Conditioned OMB + MMB Dual Attention returning `(h_o, h_m, h_g)`."""
        w = self.w
        op_f = np.asarray(op_x, dtype=np.float32)
        res_f = np.asarray(res_x, dtype=np.float32)
        bit_f = np.asarray(z_bit, dtype=np.float32)
        h_o = (op_f @ w["W1.weight"].T + w["W1.bias"]) + (bit_f @ w["W3.weight"].T)[:, None, :]
        h_m = (res_f @ w["W2.weight"].T + w["W2.bias"]) + (bit_f @ w["W4.weight"].T)[:, None, :]
        for l_idx in range(self.n_layers):
            o_att = mha_batch(
                h_o,
                h_o,
                w[f"omb_layers.{l_idx}.in_proj_weight"],
                w[f"omb_layers.{l_idx}.in_proj_bias"],
                w[f"omb_layers.{l_idx}.out_proj.weight"],
                w[f"omb_layers.{l_idx}.out_proj.bias"],
                n_heads=self.n_heads,
            )
            h_o = layer_norm(h_o + o_att, w[f"omb_norms.{l_idx}.weight"], w[f"omb_norms.{l_idx}.bias"])
            m_att = mha_batch(
                h_m,
                h_o,
                w[f"mmb_layers.{l_idx}.in_proj_weight"],
                w[f"mmb_layers.{l_idx}.in_proj_bias"],
                w[f"mmb_layers.{l_idx}.out_proj.weight"],
                w[f"mmb_layers.{l_idx}.out_proj.bias"],
                n_heads=self.n_heads,
            )
            h_m = layer_norm(h_m + m_att, w[f"mmb_norms.{l_idx}.weight"], w[f"mmb_norms.{l_idx}.bias"])
        h_g = np.concatenate([h_o.mean(axis=1), h_m.mean(axis=1)], axis=-1)
        return h_o, h_m, h_g

    def forward(
        self,
        op_x: np.ndarray,
        res_x: np.ndarray,
        pair_x: np.ndarray,
        z_bit: np.ndarray,
    ) -> np.ndarray:
        """Compute [B, N_t, N_a] candidate logits matching PyTorch `ChoiIJCAI26PGMPONet.forward`."""
        w = self.w
        pair_f = np.asarray(pair_x, dtype=np.float32)
        B, N_t, N_a, _ = pair_f.shape
        h_o, h_m, h_g = self.encode_attention(op_x, res_x, z_bit)
        ho_e = np.broadcast_to(h_o[:, :, None, :], (B, N_t, N_a, self.hidden_dim))
        hm_e = np.broadcast_to(h_m[:, None, :, :], (B, N_t, N_a, self.hidden_dim))
        hg_e = np.broadcast_to(h_g[:, None, None, :], (B, N_t, N_a, self.hidden_dim * 2))
        feat = np.concatenate([ho_e, hm_e, hg_e, pair_f], axis=-1)
        h1 = elu(feat @ w["pair_mlp.0.weight"].T + w["pair_mlp.0.bias"])
        h2 = elu(h1 @ w["pair_mlp.2.weight"].T + w["pair_mlp.2.bias"])
        return (h2 @ w["pair_mlp.4.weight"].T + w["pair_mlp.4.bias"])[..., 0].astype(np.float32, copy=False)


class NumpyWuEJOR26GRUAttentionNet:
    """Exact pure-NumPy inference model matching `WuEJOR26GRUAttentionNet.eval()`."""

    def __init__(self, weights: dict[str, np.ndarray], *, static_dim: int = 12, hidden_dim: int = 128) -> None:
        self.w = weights
        self.static_dim = int(static_dim)
        self.hidden_dim = int(hidden_dim)

    def encode_static(self, static_x: np.ndarray) -> np.ndarray:
        """Compute [B, H, L] static candidate encoding from [B, 12, L]."""
        w = self.w
        sx = np.asarray(static_x, dtype=np.float32)
        return (np.matmul(w["static_enc.weight"][:, :, 0], sx) + w["static_enc.bias"][None, :, None]).astype(
            np.float32, copy=False
        )

    def estimate_value(self, static_x: np.ndarray, dynamic_x: np.ndarray) -> np.ndarray:
        """Compute [B] critic value matching `WuEJOR26GRUAttentionNet.estimate_value`."""
        w = self.w
        hs = self.encode_static(static_x)
        dx = np.asarray(dynamic_x, dtype=np.float32)
        hd = np.matmul(w["dynamic_enc.weight"][:, :, 0], dx) + w["dynamic_enc.bias"][None, :, None]
        cat = np.concatenate([hs, hd], axis=1)
        x = np.maximum(0.0, np.matmul(w["critic_c1.weight"][:, :, 0], cat) + w["critic_c1.bias"][None, :, None])
        x = np.maximum(0.0, np.matmul(w["critic_c2.weight"][:, :, 0], x) + w["critic_c2.bias"][None, :, None])
        x = np.matmul(w["critic_c3.weight"][:, :, 0], x) + w["critic_c3.bias"][None, :, None]
        return x.sum(axis=(1, 2)).astype(np.float32, copy=False)

    def step_pointer(
        self,
        hs: np.ndarray,
        dynamic_x: np.ndarray,
        dec_in: np.ndarray,
        h_prev: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single decoding step outputting masked logits [B, L] and new GRU state [B, H]."""
        w = self.w
        H = self.hidden_dim
        B, _, L = hs.shape
        dx = np.asarray(dynamic_x, dtype=np.float32)
        hd = np.matmul(w["dynamic_enc.weight"][:, :, 0], dx) + w["dynamic_enc.bias"][None, :, None]
        di = np.asarray(dec_in, dtype=np.float32)
        h_dec = di @ w["dec_enc.weight"][:, :, 0].T + w["dec_enc.bias"]

        hp = np.asarray(h_prev, dtype=np.float32)
        gi = h_dec @ w["gru.weight_ih"].T + w["gru.bias_ih"]
        gh = hp @ w["gru.weight_hh"].T + w["gru.bias_hh"]
        r = sigmoid(gi[:, :H] + gh[:, :H])
        z = sigmoid(gi[:, H : 2 * H] + gh[:, H : 2 * H])
        n = np.tanh(gi[:, 2 * H :] + r * gh[:, 2 * H :])
        h_t = ((1.0 - z) * n + z * hp).astype(np.float32, copy=False)

        h_exp = np.broadcast_to(h_t[:, :, None], (B, H, L))
        cat_e = np.concatenate([hs, hd, h_exp], axis=1).transpose(0, 2, 1)
        e = (np.tanh(cat_e @ w["w_attn.weight"].T + w["w_attn.bias"]) @ w["v_attn.weight"].T)[..., 0]
        e_masked = np.where(np.asarray(mask, dtype=np.float32) <= 0.0, -1e9, e)
        attns = softmax(e_masked, axis=-1)[:, None, :]
        context = np.broadcast_to(np.matmul(attns, hs.transpose(0, 2, 1)), (B, L, H))
        cat_p = np.concatenate([hs.transpose(0, 2, 1), hd.transpose(0, 2, 1), context], axis=-1)
        logits = (np.tanh(cat_p @ w["w_ptr.weight"].T + w["w_ptr.bias"]) @ w["v_ptr.weight"].T)[..., 0]
        logits = np.where(np.asarray(mask, dtype=np.float32) <= 0.0, -1e9, logits).astype(np.float32, copy=False)
        return logits, h_t
