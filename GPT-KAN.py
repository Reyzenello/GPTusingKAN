"""
GPT with KAN-FFN (safe integration):
- Self-attention projections remain linear (preserve attention geometry).
- LM head is linear and weight-tied to token embeddings (classic GPT).
- FFN (MLP) replaced by two KAN layers.

KAN layer here uses simple piecewise-linear "hat" basis functions over fixed knots.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------
# Utility / config
# ------------------------------

@dataclass
class GPTConfig:
    # Either set model_type OR (n_layer, n_head, n_embd).
    model_type: str | None = "gpt-nano"
    n_layer: int | None = None
    n_head: int | None = None
    n_embd: int | None = None

    # Must set these
    vocab_size: int = 50257
    block_size: int = 1024

    # Dropouts
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1

    # FFN size multiplier
    ff_mult: int = 4

    # KAN hyperparams
    kan_knots: int = 8
    kan_range: tuple[float, float] = (-3.0, 3.0)

    # Weight tying
    tie_weights: bool = True


def _maybe_fill_from_model_type(cfg: GPTConfig):
    """Populate n_layer/n_head/n_embd from model_type if provided."""
    if cfg.model_type is None:
        assert cfg.n_layer and cfg.n_head and cfg.n_embd
        return cfg

    presets = {
        # HF-ish sizes
        "openai-gpt": dict(n_layer=12, n_head=12, n_embd=768),
        "gpt2":        dict(n_layer=12, n_head=12, n_embd=768),
        "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
        "gpt2-large":  dict(n_layer=36, n_head=20, n_embd=1280),
        "gpt2-xl":     dict(n_layer=48, n_head=25, n_embd=1600),
        # Smaller toys
        "gpt-mini":  dict(n_layer=6, n_head=6, n_embd=192),
        "gpt-micro": dict(n_layer=4, n_head=4, n_embd=128),
        "gpt-nano":  dict(n_layer=3, n_head=3, n_embd=48),
    }
    assert cfg.model_type in presets, f"unknown model_type {cfg.model_type}"
    preset = presets[cfg.model_type]
    cfg.n_layer = preset["n_layer"]; cfg.n_head = preset["n_head"]; cfg.n_embd = preset["n_embd"]
    return cfg


# ------------------------------
# Building blocks
# ------------------------------

class NewGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class CausalSelfAttention(nn.Module):
    """Standard masked multi-head self-attention with linear QKV and output proj."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=True)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=True)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # Causal mask (buffer so it moves with device)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)  # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # (B, heads, T, head_dim)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Attention: (B, h, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B, h, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


# ------------------------------
# KAN layer (hat basis)
# ------------------------------

class KANLayer(nn.Module):
    """
    y_j = b_j + sum_i [ A_{i,j} * x_i  +  sum_k C_{i,j,k} * B_k(x_i) ]
    where B_k are 'hat' (triangular) basis functions over fixed knots.
    Operates on last dimension of (B, T, D_in) -> (B, T, D_out).
    """

    def __init__(self, in_dim: int, out_dim: int, n_knots: int = 8, knot_range: tuple[float, float] = (-3.0, 3.0), bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_knots = n_knots

        # Linear term A and spline coeffs C
        self.A = nn.Parameter(torch.empty(in_dim, out_dim))                  # (I, O)
        self.C = nn.Parameter(torch.empty(in_dim, out_dim, n_knots))         # (I, O, K)
        self.b = nn.Parameter(torch.zeros(out_dim)) if bias else None

        # Fixed, uniform knots as buffer
        knots = torch.linspace(knot_range[0], knot_range[1], n_knots)
        self.register_buffer("knots", knots, persistent=False)
        dx = (self.knots[-1] - self.knots[0]) / (n_knots - 1)
        self.register_buffer("dx", torch.tensor(float(dx)), persistent=False)

        self.reset_parameters()

    def reset_parameters(self):
        # Small-ish init: linear like Linear, spline small
        nn.init.xavier_uniform_(self.A)
        nn.init.zeros_(self.C)  # start near-linear; model will bend as needed
        if self.b is not None:
            nn.init.zeros_(self.b)

    def _hat_basis(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,I) -> basis: (B,T,I,K)
        basis_k(x) = max(1 - |(x - knot_k)/dx|, 0)
        """
        # Align dtype/device
        knots = self.knots.to(dtype=x.dtype, device=x.device)
        dx = self.dx.to(dtype=x.dtype, device=x.device).clamp(min=1e-8)

        # (B,T,I,1) - (K,) -> (B,T,I,K)
        diff = x.unsqueeze(-1) - knots
        basis = 1.0 - (diff.abs() / dx)
        return basis.clamp_min(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,I) -> y: (B,T,O)
        """
        assert x.size(-1) == self.in_dim, f"KANLayer got x last dim {x.size(-1)} != in_dim {self.in_dim}"

        # Linear: (B,T,I) @ (I,O) -> (B,T,O)
        lin = torch.einsum("bti,io->bto", x, self.A)

        # Spline: tensordot over (i,k)
        # basis: (B,T,I,K), C: (I,O,K) -> (B,T,O)
        basis = self._hat_basis(x)
        spline = torch.einsum("btik,iok->bto", basis, self.C)

        y = lin + spline
        if self.b is not None:
            y = y + self.b
        return y


class KANFFN(nn.Module):
    """Two KAN layers with GELU in between: d -> 4d -> d (like GPT-FFN)."""

    def __init__(self, d_model: int, mult: int = 4, n_knots: int = 8, knot_range: tuple[float, float] = (-3.0, 3.0), resid_pdrop: float = 0.1):
        super().__init__()
        hidden = mult * d_model
        self.fc1 = KANLayer(d_model, hidden, n_knots=n_knots, knot_range=knot_range, bias=True)
        self.act = NewGELU()
        self.fc2 = KANLayer(hidden, d_model, n_knots=n_knots, knot_range=knot_range, bias=True)
        self.drop = nn.Dropout(resid_pdrop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    """Pre-LN Transformer block: x = x + Attn(LN(x)); x = x + FFN(LN(x))."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = KANFFN(
            d_model=config.n_embd,
            mult=config.ff_mult,
            n_knots=config.kan_knots,
            knot_range=config.kan_range,
            resid_pdrop=config.resid_pdrop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# ------------------------------
# GPT model
# ------------------------------

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = _maybe_fill_from_model_type(config)

        assert self.config.vocab_size is not None
        assert self.config.block_size is not None
        assert self.config.n_layer and self.config.n_head and self.config.n_embd

        # Embeddings
        self.wte = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.wpe = nn.Embedding(self.config.block_size, self.config.n_embd)
        self.drop = nn.Dropout(self.config.embd_pdrop)

        # Blocks
        self.blocks = nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)])
        self.ln_f = nn.LayerNorm(self.config.n_embd)

        # LM head (linear, weight-tied)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size, bias=False)
        if self.config.tie_weights:
            self.lm_head.weight = self.wte.weight  # weight tying

        # Init
        self.apply(self._init_weights)
        # Scale residual projections in attention output as in GPT-2
        for m in self.modules():
            if isinstance(m, CausalSelfAttention):
                nn.init.normal_(m.c_proj.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Number of parameters: {n_params/1e6:.2f}M")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight); nn.init.zeros_(module.bias)
        elif isinstance(module, KANLayer):
            # KANLayer has its own reset_parameters, already called in __init__
            pass

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx: (B, T) token indices
        targets: (B, T) or None
        """
        B, T = idx.size()
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.wte(idx) + self.wpe(pos)  # (B, T, C)
        x = self.drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, do_sample: bool = False, top_k: int | None = None):
        """
        Autoregressive generation.
        """
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(probs, dim=-1, keepdim=True)

            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    def configure_optimizers(self, weight_decay: float = 0.1, learning_rate: float = 3e-4, betas=(0.9, 0.95)):
        """
        Create AdamW with proper weight decay groups:
        - Decay: Linear/KAN weights (matrices/tensors with ndim >= 2).
        - No decay: biases + LayerNorm + Embeddings.
        """
        decay, no_decay = set(), set()
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters(recurse=False):
                if pn.endswith("bias"):
                    no_decay.add(f"{mn}.{pn}" if mn else pn)
                elif isinstance(m, (nn.Linear,)) and p.dim() >= 2:
                    decay.add(f"{mn}.{pn}" if mn else pn)
                elif isinstance(m, (nn.Embedding, nn.LayerNorm)):
                    no_decay.add(f"{mn}.{pn}" if mn else pn)
                elif isinstance(m, KANLayer):
                    # Decay A and C, don't decay b
                    if pn in ("A", "C"):
                        decay.add(f"{mn}.{pn}" if mn else pn)
                    elif pn in ("b",):
                        no_decay.add(f"{mn}.{pn}" if mn else pn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter = decay & no_decay
        assert len(inter) == 0, f"params in both decay/no_decay: {inter}"
        missing = set(param_dict.keys()) - (decay | no_decay)
        assert len(missing) == 0, f"some params not in any group: {missing}"

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(decay)], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(no_decay)], "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)


# ------------------------------
# Tiny smoke test
# ------------------------------

if __name__ == "__main__":
    torch.manual_seed(27) #to define
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = GPTConfig(
        model_type="gpt-nano",
        vocab_size=128,     # toy vocab
        block_size=64,      # toy context
        ff_mult=4,
        kan_knots=8,
        kan_range=(-3.0, 3.0),
    )
    model = GPT(cfg).to(device)

    B, T = 2, 16
    x = torch.randint(0, cfg.vocab_size, (B, T), device=device)
    y = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    logits, loss = model(x, y)
    print("logits:", tuple(logits.shape), "loss:", float(loss))

    # quick generation
    start = torch.randint(0, cfg.vocab_size, (1, 4), device=device)
    out = model.generate(start, max_new_tokens=8, temperature=1.0, do_sample=False)
    print("gen:", out.tolist())
message.txt
15 KB
