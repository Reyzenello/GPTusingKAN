"""
Full definition of a GPT Language Model.

References:
1) OpenAI's GPT-2 TensorFlow implementation:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) Hugging Face Transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers\
/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from kan_gpt.efficient_kan.model import KAN as EFFICIENT_KAN
from kan_gpt.kan.KAN import KAN as ORIGINAL_KAN
from kan_gpt.mingpt.utils import CfgNode as CN
from kan_gpt.settings import settings


# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    GELU activation function from Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """
    Multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, config):
        super().__init__()
        self.kan_implementation = config.kan_implementation
        self.KAN = self._get_KAN()

        assert config.n_embd % config.n_head == 0
        # Key, query, value projections for all heads in a batch
        self.c_attn = self.KAN(width=[config.n_embd, 3 * config.n_embd])
        # Output projection
        self.c_proj = self.KAN(width=[config.n_embd, config.n_embd])
        # Regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # Causal mask to ensure attention is only applied to the left
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def _get_KAN(self):
        if self.kan_implementation == "EFFICIENT_KAN":
            return EFFICIENT_KAN
        elif self.kan_implementation == "ORIGINAL_KAN":
            return ORIGINAL_KAN
        raise NotImplementedError(f"Unknown KAN implementation: {self.kan_implementation}")

    def forward(self, x):
        B, T, C = x.size()  # Batch size, sequence length, embedding dimensionality

        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Re-assemble all head outputs

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, config):
        super().__init__()
        self.kan_implementation = config.kan_implementation
        self.KAN = self._get_KAN()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            self.KAN(width=[config.n_embd, 4 * config.n_embd]),
            NewGELU(),
            self.KAN(width=[4 * config.n_embd, config.n_embd]),
            nn.Dropout(config.resid_pdrop),
        )

    def _get_KAN(self):
        if self.kan_implementation == "EFFICIENT_KAN":
            return EFFICIENT_KAN
        elif self.kan_implementation == "ORIGINAL_KAN":
            return ORIGINAL_KAN
        raise NotImplementedError(f"Unknown KAN implementation: {self.kan_implementation}")

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    """GPT Language Model."""

    @staticmethod
    def get_default_config():
        C = CN()
        # Either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = "gpt"
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        # These options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # Dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        # KAN Implementation
        C.kan_implementation = settings.kan.KAN_IMPLEMENTATION
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.kan_implementation = config.kan_implementation
        self.KAN = self._get_KAN()

        type_given = config.model_type is not None
        params_given = all(
            [
                config.n_layer is not None,
                config.n_head is not None,
                config.n_embd is not None,
            ]
        )
        assert type_given ^ params_given  # Exactly one of these (XOR)
        if type_given:
            # Translate from model_type to detailed configuration
            config.merge_from_dict(
                {
                    # Names follow Hugging Face naming conventions
                    # GPT-1
                    "openai-gpt": dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                    # GPT-2 configs
                    "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                    "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
                    "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
                    "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
                    # Gophers
                    "gopher-44m": dict(n_layer=8, n_head=16, n_embd=512),
                    # ...
                    # Tiny models
                    "gpt-mini": dict(n_layer=6, n_head=6, n_embd=192),
                    "gpt-micro": dict(n_layer=4, n_head=4, n_embd=128),
                    "gpt-nano": dict(n_layer=3, n_head=3, n_embd=48),
                    "gpt-pico": dict(n_layer=1, n_head=1, n_embd=1),
                }[config.model_type]
            )

        self.transformer = nn.Sequential(
            nn.Embedding(config.vocab_size, config.n_embd),
            nn.Embedding(config.block_size, config.n_embd),
            nn.Dropout(config.embd_pdrop),
            *[Block(config) for _ in range(config.n_layer)],
            nn.LayerNorm(config.n_embd),
        )
        self.lm_head = self.KAN(width=[config.n_embd, config.vocab_size], bias_trainable=False)

        # Initialize weights, apply scaled init to residual projections
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # Report number of parameters
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("Number of parameters: %.2fM" % (n_params / 1e6,))

    def _get_KAN(self):
        if self.kan_implementation == "EFFICIENT_KAN":
            return EFFICIENT_KAN
        elif self.kan_implementation == "ORIGINAL_KAN":
            return ORIGINAL_KAN
        raise NotImplementedError(f"Unknown KAN implementation: {self.kan_implementation}")

    def _kan_loss(self, x: torch.Tensor, lamb_l1=1.0, lamb_entropy=2.0, lamb_coef=0.0, lamb_coefdiff=0.0,
                   small_mag_threshold=1e-16, small_reg_factor=1.0):

        def _reg(mod):
            def _nonlinear(x, th=small_mag_threshold, factor=small_reg_factor):
                return (x < th) * x * factor + (x > th) * (x + (factor - 1) * th)

            reg_ = 0.0
            for i in range(len(mod.acts_scale)):
                vec = mod.acts_scale[i].reshape(-1)
                p = vec / torch.sum(vec)
                l1 = torch.sum(_nonlinear(vec))
                entropy = -torch.sum(p * torch.log2(p + 1e-4))
                reg_ += lamb_l1 * l1 + lamb_entropy * entropy  # L1 and entropy regularization

            # Regularize coefficient to encourage spline to be zero
            for i in range(len(mod.act_fun)):
                coeff_l1 = torch.sum(torch.mean(torch.abs(mod.act_fun[i].coef), dim=1))
                coeff_diff_l1 = torch.sum(torch.mean(torch.abs(torch.diff(mod.act_fun[i].coef)), dim=1))
                reg_ += lamb_coef * coeff_l1 + lamb_coefdiff * coeff_diff_l1

            return reg_

        total_reg = torch.tensor(0.0).to(device=x.device, dtype=torch.float32)
        size = 0
        for mod in self.modules():
            if isinstance(mod, self.KAN):
                total_reg += _reg(mod)
                size += 1

        mean_reg = total_reg / size
        return mean_reg

    def _init_weights(self, module):
        if isinstance(module, self.KAN):
            # TODO: init weights for KAN
            pass
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying weights from a Hugging Face checkpoint.
        """
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        from transformers import GPT2LMHeadModel

        # Create a from-scratch initialized GPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257  # OpenAI's model vocabulary
        config.block_size = 1024  # OpenAI's model block_size
        model = GPT(config)
        sd = model.state_dict()

        # Initialize a Hugging Face model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy weights, ensuring alignment and matching names/shapes
        keys = [k for k in sd_hf if not k.endswith("attn.masked_bias")]  # Ignore these
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.0.weight", "mlp.2.weight"]
        # OpenAI checkpoints use a "Conv1D" module, which needs to be transposed for nn.Linear
        assert len(keys) == len(sd)
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # Transpose Conv1D weights
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Copy other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, train_config):
        """
        Separates model parameters into weight decay and no-decay groups for optimization.
        """

        decay = set()
        no_decay = set()
        whitelist_weight_modules = (self.KAN,)
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # Full parameter name

                if pn.endswith("bias"):
                    # All biases will not be decayed
                    no_decay.add(fpn)
                elif isinstance(m, whitelist_weight_modules):
                    # Weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # Weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # Validate that all parameters are considered
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {str(inter_params)} are in both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, (
            f"Parameters {str(param_dict.keys() - union_params)} were not separated into decay/no_decay sets!"
        )

        # Create PyTorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None, lamb=0.01, lamb_l1=1.0, lamb_entropy=2.0, lamb_coef=0.0,
                lamb_coefdiff=0.0, small_mag_threshold=1e-16, small_reg_factor=1.0):
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # Shape (1, t)

        # Forward the GPT model
        tok_emb = self.transformer[0](idx)  # Token embeddings (b, t, n_embd)
        pos_emb = self.transformer[1](pos)  # Position embeddings (1, t, n_embd)
        x = self.transformer[2](tok_emb + pos_emb)
        for i in range(3, 3 + len(self.transformer) - 3):
            x = self.transformer[i](x)
        x = self.transformer[-1](x)
        logits = self.lm_head(x)

        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            if settings.kan.KAN_IMPLEMENTATION == "ORIGINAL_KAN":
                reg = self._kan_loss(
                    x=idx,
                    lamb_l1=lamb_l1,
                    lamb_entropy=lamb_entropy,
                    lamb_coef=lamb_coef,
                    lamb_coefdiff=lamb_coefdiff,
                    small_mag_threshold=small_mag_threshold,
                    small_reg_factor=small_reg_factor,
                )
                loss = loss + lamb * reg

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Generate text from a conditioning sequence.
        """
        for _ in range(max_new_tokens):
            # Crop context if it exceeds block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # Get logits for the next token
            logits, _ = self(idx_cond)
            # Scale logits by temperature
            logits = logits[:, -1, :] / temperature
            # Apply top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample or take the most likely token
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # Append sampled token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


if __name__ == "__main__":
    model_config = GPT.get_default_config()
    model_config.model_type = "gpt2"
    model_config.vocab_size = 5
    model_config.block_size = 10
    model = GPT(model_config)

    x = torch.zeros((1, 10), dtype=torch.long)
    y = torch.zeros((1, 10), dtype=torch.long)

    logits, loss = model(x, y)

    print(logits.shape)
