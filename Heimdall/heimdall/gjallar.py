"""
gjallar.py — Gjallar Surprise-Gated Retrieval Attention (SGRA).

Extends Heimdall/Bifrost hybrid architecture with:
1. Zero-Cost Surprise Gating: dynamically branches between Fast-Path (recurrent state + window)
   and Slow-Path (exact host-RAM sparse KV retrieval).
2. Two-Level Superblock Bounds: maintains 4096-token superblock summaries on GPU (4 MB at 1M context)
   and streams 64-token block summaries + KV blocks over PCIe only when surprise gate opens.
3. Fast-Path / Slow-Path execution with budget-aware inference stats.
"""

from typing import Dict, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GjallarGate(nn.Module):
    """Surprise gate computing read/no-read decision from zero-cost recurrent & window observables."""

    def __init__(self, in_features: int = 2) -> None:
        super().__init__()
        self.in_features = in_features
        self.proj = nn.Linear(in_features, 1)
        nn.init.constant_(self.proj.bias, -1.0)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(
        self,
        omem_norm: torch.Tensor,
        lse_win: Optional[torch.Tensor] = None,
        res_feat: Optional[torch.Tensor] = None,
        hard: bool = False,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Returns gating score g [B, H, t, 1]. Hard=True applies binary thresholding for inference."""
        if lse_win is None:
            feat = omem_norm  # pre-assembled feature tensor [B, H, t, in_features]
        elif res_feat is not None and self.in_features == 3:
            feat = torch.stack([omem_norm, lse_win, res_feat], dim=-1)
        else:
            feat = torch.stack([omem_norm, lse_win], dim=-1)  # [B, H, t, 2]
        logits = self.proj(feat)                         # [B, H, t, 1]
        if hard:
            return (torch.sigmoid(logits) > threshold).float()
        return torch.sigmoid(logits)


class TwoLevelBlockIndex:
    """Two-level hierarchical block index for constant-GPU-memory exact KV bounds and quantized centroid selectors."""

    def __init__(self, block_size: int = 64, super_size: int = 64) -> None:
        self.block_size = block_size   # L = 64 tokens per block
        self.super_size = super_size   # M = 64 blocks (4096 tokens) per superblock

    @staticmethod
    def block_upper_bounds(q: torch.Tensor, kmin: torch.Tensor, kmax: torch.Tensor) -> torch.Tensor:
        """
        Computes maximum potential dot-product q·k for any vector inside the bounded box:
            max_{k} q·k = Σ_d [ max(0, q_d) * kmax_d + min(0, q_d) * kmin_d ]
        q: [B, G, R, t, D]
        kmin/kmax: [B, G, 1, 1, num_blocks, D]
        """
        # Align shapes for dot product over D
        q_pos = F.relu(q).unsqueeze(-2)      # [B, G, R, t, 1, D]
        q_neg = (-F.relu(-q)).unsqueeze(-2)  # [B, G, R, t, 1, D]
        ub_pos = (q_pos * kmax).sum(dim=-1)  # [B, G, R, t, num_blocks]
        ub_neg = (q_neg * kmin).sum(dim=-1)  # [B, G, R, t, num_blocks]
        return (ub_pos + ub_neg).amax(dim=2) # Max over R query heads -> [B, G, t, num_blocks]

    @staticmethod
    def quantize_c4(x: torch.Tensor, group_size: int = 32) -> torch.Tensor:
        """
        Symmetric int4 quantization in groups of 32 with per-group FP16 scale (7 levels: [-7, 7]).
        Matches Vulkan C4 shader FP16 scale precision.
        Returns dequantized float tensor.
        """
        D = x.shape[-1]
        g_size = group_size if D % group_size == 0 else D
        orig_shape = x.shape
        x_g = x.view(*orig_shape[:-1], -1, g_size)
        scale = torch.clamp(x_g.abs().amax(dim=-1, keepdim=True), min=1e-12) / 7.0
        scale = scale.to(torch.float16).to(x.dtype)
        x_q = torch.clamp(torch.round(x_g / scale), -7.0, 7.0) * scale
        return x_q.view(orig_shape)

    @staticmethod
    def centroid_mass_feature(
        q_gqa: torch.Tensor,       # [B, G, R, t, D]
        cen_c4: torch.Tensor,      # [B, G, nb, D]
        scale: float,
        lse_win: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Estimates archive attention mass from all C4 block centroids:
            LSE_cen(q) = logsumexp(q_gqa @ cen_c4 * scale, dim=-1)  # over nb blocks
        If lse_win is provided, returns relative archive mass: LSE_cen - lse_win [B, H, t].
        Otherwise returns LSE_cen [B, H, t].
        """
        B, G, R, t, D = q_gqa.shape
        H = G * R
        c4_dots = torch.einsum("bgrtd,bgnd->bgrtn", q_gqa, cen_c4) * scale
        lse_cen = torch.logsumexp(c4_dots, dim=-1).view(B, H, t)
        if lse_win is not None:
            return lse_cen - lse_win
        return lse_cen

    @staticmethod
    def centroid_c4_select(
        q_gqa: torch.Tensor,       # [B, G, R, t, D]
        kb: torch.Tensor,          # [B, G, nb, block_size, D]
        scale: float,
        top_k: int = 16,
        shortlist_size: int = 64,
        group_size: int = 32,
        cen_c4: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Centroid C4 selector with exact token-max reranking:
        1. Computes or receives cached per-block centroids kb.mean(dim=3).
        2. Quantizes centroids to int4 (groups of 32 values).
        3. Scores all blocks via q_gqa @ cen_c4 and takes shortlist of top-min(shortlist_size, nb) blocks.
        4. Evaluates exact token-max dot product for candidate keys within the shortlisted blocks.
        5. Reranks to final top_k blocks. Returns block indices [B, G, t, kk].
        """
        B, G, nb, L, D = kb.shape
        t = q_gqa.size(3)
        effective_shortlist = max(shortlist_size, top_k)
        short_k = min(effective_shortlist, nb)
        kk = min(top_k, short_k)

        if nb <= kk:
            return torch.arange(nb, device=kb.device).view(1, 1, 1, nb).expand(B, G, t, nb)

        # 1 & 2. Centroid computation and int4 quantization (or use precomputed / cached)
        if cen_c4 is None:
            cen = kb.mean(dim=3)  # [B, G, nb, D]
            cen_c4 = TwoLevelBlockIndex.quantize_c4(cen, group_size=group_size)

        # 3. Score all blocks by C4 centroid dot product
        c4_scores = torch.einsum("bgrtd,bgnd->bgrtn", q_gqa, cen_c4) * scale  # [B, G, R, t, nb]
        sc = c4_scores.amax(dim=2)  # Max over query heads R -> [B, G, t, nb]

        # Top shortlist candidates
        short_idx = sc.topk(short_k, dim=-1).indices  # [B, G, t, short_k]

        # 4. Exact token-max rerank on the shortlist
        # Gather candidate keys for shortlisted blocks: [B, G, t, short_k, L, D]
        if t == 1:
            gather_idx = short_idx[:, :, 0].unsqueeze(-1).unsqueeze(-1).expand(B, G, short_k, L, D)
            kb_short = torch.gather(kb, 2, gather_idx).unsqueeze(2)  # [B, G, 1, short_k, L, D]
        else:
            idx_exp = short_idx.unsqueeze(-1).unsqueeze(-1).expand(B, G, t, short_k, L, D)
            kb_exp = kb.unsqueeze(2).expand(B, G, t, nb, L, D)
            kb_short = torch.gather(kb_exp, 3, idx_exp)  # [B, G, t, short_k, L, D]

        # Exact dot products against shortlisted tokens
        dot_short = torch.einsum("bgrtd,bgtmld->bgrtml", q_gqa, kb_short) * scale  # [B, G, R, t, short_k, L]
        tok_max = dot_short.amax(dim=-1).amax(dim=2)  # [B, G, t, short_k]

        # 5. Rerank top_k within the shortlist
        pick = tok_max.topk(kk, dim=-1).indices  # [B, G, t, kk]
        top_blocks = torch.gather(short_idx, dim=-1, index=pick)  # [B, G, t, kk]
        return top_blocks


class GjallarAttention(nn.Module):
    """
    Gjallar Surprise-Gated Retrieval Attention Layer with proper GQA [B, G, R, t, D] grouped attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        window: int = 2048,
        sinks: int = 4,
        block_size: int = 64,
        super_size: int = 64,
        top_k_blocks: int = 16,
        top_k_supers: int = 8,
        selector: str = "box_bound",
        shortlist_size: int = 64,
        gate_in_features: int = 2,
    ) -> None:
        super().__init__()
        if selector not in ("box_bound", "centroid_c4"):
            raise ValueError(f"Unknown selector '{selector}'; choose 'box_bound' or 'centroid_c4'")
        self.selector = selector
        self.shortlist_size = shortlist_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        self.window = window
        self.sinks = sinks
        self.block_size = block_size
        self.super_size = super_size
        self.top_k_blocks = top_k_blocks
        self.top_k_supers = top_k_supers

        self.index = TwoLevelBlockIndex(block_size, super_size)
        self.gate = GjallarGate(in_features=gate_in_features)

        self.write_gate = nn.Linear(d_model, n_kv_heads)
        self.mix_mem = nn.Parameter(torch.ones(1, n_heads, 1, 1))

        self.fast_path_count = 0
        self.slow_path_count = 0
        self.enable_index_cache = False
        self.cached_cen_c4: Optional[torch.Tensor] = None

    def reset_stats(self) -> None:
        self.fast_path_count = 0
        self.slow_path_count = 0
        self.cached_cen_c4 = None

    def _get_or_update_centroid_c4(self, kb: torch.Tensor, B: int, G: int, nb: int) -> torch.Tensor:
        """
        Retrieves or incrementally maintains the C4 quantized block centroids.
        Avoids per-token descriptor rebuild: updates only newly arrived candidate blocks.
        """
        if not self.enable_index_cache:
            return TwoLevelBlockIndex.quantize_c4(kb.mean(dim=3))

        if self.cached_cen_c4 is not None and self.cached_cen_c4.shape[:3] == (B, G, nb):
            return self.cached_cen_c4
        elif self.cached_cen_c4 is not None and self.cached_cen_c4.shape[:2] == (B, G) and self.cached_cen_c4.size(2) == nb - 1:
            new_cen = kb[:, :, -1:].mean(dim=3)
            new_c4 = TwoLevelBlockIndex.quantize_c4(new_cen)
            self.cached_cen_c4 = torch.cat([self.cached_cen_c4, new_c4], dim=2)
            return self.cached_cen_c4
        else:
            cen = kb.mean(dim=3)
            self.cached_cen_c4 = TwoLevelBlockIndex.quantize_c4(cen)
            return self.cached_cen_c4

    def get_stats(self) -> Dict[str, float]:
        total = self.fast_path_count + self.slow_path_count
        rate = (self.slow_path_count / total) if total > 0 else 0.0
        return {
            "fast_path_tokens": self.fast_path_count,
            "slow_path_tokens": self.slow_path_count,
            "retrieval_rate": round(rate, 4),
        }

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        S_prev: Optional[torch.Tensor] = None,
        hard_gate: bool = True,
        gate_threshold: float = 0.5,
        beta: Optional[torch.Tensor] = None,
        res_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        q: [B, H, t, D]
        k: [B, G, n, D]
        v: [B, G, n, D]
        """
        B, H, t, D = q.shape
        if t != 1:
            # The reference has no causal mask and applies one memory write per call, so chunked prefill would let early
            # queries see future values. Decode (t=1) only; a chunked version needs a causal mask and a per-token scan.
            raise ValueError(f"GjallarAttention reference supports single-token decode only (t=1), got t={t}")
        G = self.n_kv_heads
        R = H // G
        n = k.size(2)
        dev = q.device

        # Reshape queries into GQA groups: [B, G, R, t, D]
        q_gqa = q.view(B, G, R, t, D)

        # 1. Sliding Window Attention (last window tokens + initial sinks)
        if self.sinks > 0 and n > self.window:
            s_count = min(self.sinks, n)
            w_count = min(self.window, n - s_count)
            if w_count == 0:
                k_win, v_win = k[:, :, :s_count], v[:, :, :s_count]
            else:
                k_win = torch.cat([k[:, :, :s_count], k[:, :, n - w_count:]], dim=2)
                v_win = torch.cat([v[:, :, :s_count], v[:, :, n - w_count:]], dim=2)
        else:
            win_start = max(0, n - self.window)
            k_win, v_win = k[:, :, win_start:], v[:, :, win_start:]
        # Grouped dot-product: [B, G, R, t, D] x [B, G, 1, w, D] -> [B, G, R, t, w]
        scores_win = (q_gqa @ k_win.unsqueeze(2).transpose(-1, -2)) * self.scale
        lse_win = torch.logsumexp(scores_win, dim=-1).view(B, H, t)  # [B, H, t]
        attn_win = F.softmax(scores_win, dim=-1)
        o_win = (attn_win @ v_win.unsqueeze(2)).view(B, H, t, D)

        # 2. Delta-rule associative memory read
        if S_prev is None:
            S_prev = q.new_zeros(B, G, D, D)
        # S_prev: [B, G, D, D], q_gqa: [B, G, R, t, D]
        o_mem = torch.einsum("bgij,bgrtj->bgrti", S_prev, q_gqa).view(B, H, t, D)
        omem_norm = o_mem.norm(dim=-1) # [B, H, t]

        # 3. Surprise Gating
        nc_tokens = max(0, n - self.window - self.sinks)
        nb = nc_tokens // self.block_size

        if res_feat is not None:
            g = self.gate(omem_norm, lse_win, res_feat=res_feat, hard=hard_gate, threshold=gate_threshold)
        elif getattr(self.gate, "in_features", 2) == 3:
            if nb > 0:
                k_cand = k[:, :, self.sinks:self.sinks + nb * self.block_size]
                kb_cand = k_cand.view(B, G, nb, self.block_size, D)
                cen_for_mass = self._get_or_update_centroid_c4(kb_cand, B, G, nb)
                mass_feat = TwoLevelBlockIndex.centroid_mass_feature(q_gqa, cen_for_mass, self.scale, lse_win)
            else:
                mass_feat = q.new_zeros(B, H, t)
            g = self.gate(omem_norm, lse_win, res_feat=mass_feat, hard=hard_gate, threshold=gate_threshold)
        else:
            g = self.gate(omem_norm, lse_win, hard=hard_gate, threshold=gate_threshold) # [B, H, t, 1]

        # 4. Fast-Path vs Slow-Path Routing
        token_retrieves = (g > 0).any(dim=1).squeeze(-1)  # [B, t] per-token retrieval flag
        num_slow = int(token_retrieves.sum().item())
        num_fast = (B * t) - num_slow
        read_triggered = num_slow > 0

        if not read_triggered or nb == 0:
            # FAST-PATH: Zero PCIe traffic, pure constant memory
            self.fast_path_count += (B * t)
            o_final = o_win + self.mix_mem * o_mem
        else:
            # SLOW-PATH: Hierarchical block retrieval (tracks exact per-token batch counts)
            self.slow_path_count += num_slow
            self.fast_path_count += num_fast

            # Block summaries
            k_cand = k[:, :, self.sinks:self.sinks + nb * self.block_size]
            v_cand = v[:, :, self.sinks:self.sinks + nb * self.block_size]
            kb = k_cand.view(B, G, nb, self.block_size, D)
            vb = v_cand.view(B, G, nb, self.block_size, D)
            # Select top-k blocks per KV head
            kk = min(self.top_k_blocks, nb)
            if self.selector == "centroid_c4":
                cen_c4_arg = self._get_or_update_centroid_c4(kb, B, G, nb) if self.enable_index_cache else None
                top_blocks = self.index.centroid_c4_select(
                    q_gqa, kb, scale=self.scale, top_k=self.top_k_blocks, shortlist_size=self.shortlist_size, cen_c4=cen_c4_arg
                )
            else:
                kmin = kb.amin(dim=3).view(B, G, 1, 1, nb, D)
                kmax = kb.amax(dim=3).view(B, G, 1, 1, nb, D)
                ub = self.index.block_upper_bounds(q_gqa, kmin, kmax)
                top_blocks = ub.topk(kk, dim=-1).indices # [B, G, t, kk]

            # Exact attention over retrieved blocks: true O(1) memory and compute
            # Gather strictly the picked blocks from candidate tensor into [B, G, kk * block_size, D]
            blk_gather_idx = top_blocks[:, :, 0].unsqueeze(-1).unsqueeze(-1).expand(B, G, kk, self.block_size, D)
            k_ret = torch.gather(kb, 2, blk_gather_idx).view(B, G, kk * self.block_size, D)
            v_ret = torch.gather(vb, 2, blk_gather_idx).view(B, G, kk * self.block_size, D)

            # Attention computed strictly over retrieved tokens (O(kk * block_size))
            scores_ret = (q_gqa @ k_ret.unsqueeze(2).transpose(-1, -2)) * self.scale  # [B, G, R, t, kk * block_size]
            attn_ret = F.softmax(scores_ret, dim=-1)
            o_ret = (attn_ret @ v_ret.unsqueeze(2)).view(B, H, t, D)

            o_final = o_win + self.mix_mem * o_mem + g * o_ret

        # 5. Delta-rule associative memory update for the latest token leaving window
        # Sinks (0..sinks-1) are permanently pinned in the window and never enter delta memory
        if n > self.window + self.sinks:
            k_leave = k[:, :, n - self.window - 1] # [B, G, D]
            v_leave = v[:, :, n - self.window - 1] # [B, G, D]
            pred_v = torch.einsum("bgij,bgj->bgi", S_prev, k_leave)
            err = v_leave - pred_v
            if beta is not None:
                err = err * beta.view(B, G, 1)
            decay = 0.999
            S_next = decay * S_prev + torch.einsum("bgi,bgj->bgij", err, k_leave)
        else:
            S_next = S_prev

        return o_final, S_next, g


class GjallarLayer(nn.Module):
    """
    A single layer in the Gjallar hybrid architecture.
    Can be configured as:
    - 'gdn': Pure recurrent Gated DeltaNet layer (constant memory O(1), no KV storage).
    - 'sgra': Surprise-Gated Retrieval Attention layer (local window + recurrent S + host-RAM block retrieval).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        layer_type: str = "gdn",
        window: int = 2048,
        sinks: int = 4,
        block_size: int = 64,
        top_k_blocks: int = 16,
        selector: str = "centroid_c4",
        shortlist_size: int = 64,
        gate_in_features: int = 2,
    ) -> None:
        super().__init__()
        self.layer_type = layer_type
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)

        if layer_type == "sgra":
            self.attn = GjallarAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                window=window,
                sinks=sinks,
                block_size=block_size,
                top_k_blocks=top_k_blocks,
                selector=selector,
                shortlist_size=shortlist_size,
                gate_in_features=gate_in_features,
            )
        elif layer_type == "gdn":
            self.write_gate = nn.Linear(d_model, n_kv_heads)
            self.decay = 0.999
        else:
            raise ValueError(f"Unknown layer_type '{layer_type}', choose 'gdn' or 'sgra'")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        S_prev: Optional[torch.Tensor] = None,
        hard_gate: bool = True,
        gate_threshold: float = 0.5,
        res_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.layer_type == "sgra":
            return self.attn(q, k, v, S_prev=S_prev, hard_gate=hard_gate, gate_threshold=gate_threshold, res_feat=res_feat)
        else:
            # Pure GDN layer: zero host KV memory, flat O(1) state
            B, H, t, D = q.shape
            G = self.n_kv_heads
            R = H // G
            q_gqa = q.view(B, G, R, t, D)
            if S_prev is None:
                S_prev = q.new_zeros(B, G, D, D)
            # Associative readout o = S q
            o = torch.einsum("bgij,bgrtj->bgrti", S_prev, q_gqa).view(B, H, t, D)
            # Recurrent delta update with latest token
            k_last = k[:, :, -1]
            v_last = v[:, :, -1]
            pred_v = torch.einsum("bgij,bgj->bgi", S_prev, k_last)
            err = v_last - pred_v
            S_next = self.decay * S_prev + torch.einsum("bgi,bgj->bgij", err, k_last)
            return o, S_next, None


class GjallarStack(nn.Module):
    """
    Multi-layer reference stack implementing the 3:1 hybrid ratio:
    For every 4 layers: 3 GDN layers + 1 anchor SGRA layer.
    Pure structural reference model; makes no hardware performance claims.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        num_layers: int = 4,
        window: int = 64,
        sinks: int = 4,
        block_size: int = 16,
        top_k_blocks: int = 4,
        selector: str = "centroid_c4",
        shortlist_size: int = 64,
        gate_in_features: int = 2,
        layer_types: Optional[list] = None,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers if layer_types is None else len(layer_types)
        self.layers = nn.ModuleList()
        for l_idx in range(self.num_layers):
            if layer_types is not None:
                l_type = layer_types[l_idx]
            else:
                # 3:1 ratio: anchor SGRA on layers where (l_idx + 1) % 4 == 0 (e.g. layer 3, 7, 11...)
                l_type = "sgra" if (l_idx + 1) % 4 == 0 else "gdn"
            self.layers.append(
                GjallarLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    layer_type=l_type,
                    window=window,
                    sinks=sinks,
                    block_size=block_size,
                    top_k_blocks=top_k_blocks,
                    selector=selector,
                    shortlist_size=shortlist_size,
                    gate_in_features=gate_in_features,
                )
            )

    def forward(
        self,
        q: torch.Tensor,
        k: Union[torch.Tensor, list, tuple],
        v: Union[torch.Tensor, list, tuple],
        states: Optional[list] = None,
        hard_gate: bool = True,
        gate_threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, list, list]:
        """
        Processes single-token decode step across all layers sequentially.
        Accepts either shared k, v tensors [B, G, n, D] or per-layer lists/tuples of tensors.
        Returns:
            out: output of final layer [B, H, t, D]
            next_states: list of S matrices for each layer
            gates: list of gating scores for SGRA layers (or None for GDN layers)
        """
        B, H, t, D = q.shape
        if states is None:
            states = [None] * self.num_layers
        next_states = []
        gates = []
        x = q
        for l_idx, layer in enumerate(self.layers):
            k_l = k[l_idx] if isinstance(k, (list, tuple)) else k
            v_l = v[l_idx] if isinstance(v, (list, tuple)) else v
            out_l, S_next, g = layer(
                x, k_l, v_l, S_prev=states[l_idx], hard_gate=hard_gate, gate_threshold=gate_threshold
            )
            next_states.append(S_next)
            gates.append(g)
            x = x + out_l  # residual connection
        return x, next_states, gates


if __name__ == "__main__":
    print("=" * 60)
    print("Gjallar Attention: Synthetic Reference Demo (CPU, toy scale)")
    print("=" * 60)
    torch.manual_seed(42)
    B, H, G, D = 1, 4, 2, 64
    window, sinks, block_size = 64, 4, 16
    top_k = 4
    n_ctx = 256  # 256 tokens total
    n_steps = 10 # 10 decode steps

    print(f"Configuration: B={B}, H={H}, G={G}, D={D}")
    print(f"Context: {n_ctx} tokens, window: {window}, sinks: {sinks}, block_size: {block_size}")
    print(f"Retrieved blocks top-k: {top_k}")
    print("\nSimulating 10 decode steps with synthetic random tokens...")

    layer_box = GjallarAttention(
        d_model=H * D, n_heads=H, n_kv_heads=G, head_dim=D,
        window=window, sinks=sinks, block_size=block_size, top_k_blocks=top_k,
        selector="box_bound"
    )
    layer_c4 = GjallarAttention(
        d_model=H * D, n_heads=H, n_kv_heads=G, head_dim=D,
        window=window, sinks=sinks, block_size=block_size, top_k_blocks=top_k,
        selector="centroid_c4"
    )
    layer_c4.gate.load_state_dict(layer_box.gate.state_dict())

    k = torch.randn(B, G, n_ctx, D)
    v = torch.randn(B, G, n_ctx, D)
    S_box = None
    S_c4 = None

    for step in range(n_steps):
        q = torch.randn(B, H, 1, D)
        thresh = 0.5 if step % 2 == 0 else 0.0
        out_box, S_box, g_box = layer_box(q, k, v, S_prev=S_box, hard_gate=True, gate_threshold=thresh)
        out_c4, S_c4, g_c4 = layer_c4(q, k, v, S_prev=S_c4, hard_gate=True, gate_threshold=thresh)

    stats_box = layer_box.get_stats()
    stats_c4 = layer_c4.get_stats()

    print("\nExecution Counters (Computed over 10 steps):")
    print(f"  [box_bound]   Fast-path tokens: {stats_box['fast_path_tokens']}, Slow-path tokens: {stats_box['slow_path_tokens']}, Retrieval rate: {stats_box['retrieval_rate']:.1%}")
    print(f"  [centroid_c4] Fast-path tokens: {stats_c4['fast_path_tokens']}, Slow-path tokens: {stats_c4['slow_path_tokens']}, Retrieval rate: {stats_c4['retrieval_rate']:.1%}")

    print("\nDemonstrating 4-layer 3:1 Hybrid Stack (3 GDN + 1 SGRA):")
    stack = GjallarStack(
        d_model=H * D, n_heads=H, n_kv_heads=G, head_dim=D,
        num_layers=4, window=window, sinks=sinks, block_size=block_size, top_k_blocks=top_k,
        selector="centroid_c4"
    )
    out_stack, states_stack, gates_stack = stack(torch.randn(B, H, 1, D), k, v)
    gate_val = gates_stack[3].mean().item()
    print(f"  Stack output shape: {out_stack.shape}, States: {len(states_stack)} layers, Anchor gate mean: {gate_val:.3f}")
    print("=" * 60)
