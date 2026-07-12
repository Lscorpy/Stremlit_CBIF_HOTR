# =============================================================================
# Cross-Branch Attention Fusion (CBAF) with per-query confidence gating
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossBranchFusion(nn.Module):
    """
    Bidirectional, confidence-gated cross-attention between the last decoder
    layer of HOI and HHI.

    Parameters
    ----------
    hidden_dim   : int    – transformer d_model (same for both branches)
    nhead        : int    – number of attention heads
    dropout      : float  – dropout probability inside MHA
    use_gate     : bool   – learned global scalar gate per direction
                            (initialised near 0 so fusion starts gently)
    ffn          : bool   – two-layer FFN after cross-attn
    use_conf_gate: bool   – per-query confidence gating (NEW, default True)
    """

    def __init__(
        self,
        hidden_dim: int,
        nhead: int = 8,
        dropout: float = 0.1,
        use_gate: bool = True,
        ffn: bool = True,
        use_conf_gate: bool = True,
    ):
        super().__init__()

        self.hoi_from_har = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.hoi_norm1 = nn.LayerNorm(hidden_dim)

        self.har_from_hoi = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=nhead,
            dropout=dropout, batch_first=True,
        )
        self.har_norm1 = nn.LayerNorm(hidden_dim)

        self.ffn = ffn
        if ffn:
            self.hoi_ffn = _FFN(hidden_dim, dropout=dropout)
            self.har_ffn = _FFN(hidden_dim, dropout=dropout)
            self.hoi_norm2 = nn.LayerNorm(hidden_dim)
            self.har_norm2 = nn.LayerNorm(hidden_dim)

        self.use_gate = use_gate
        if use_gate:
            # self.alpha_hoi = nn.Parameter(torch.tensor(-3.0))  # sigmoid(-3)≈0.05
            # self.alpha_har = nn.Parameter(torch.tensor(-3.0))
            self.alpha_hoi = nn.Parameter(torch.tensor(-1.0))  # sigmoid(-1)≈0.27
            self.alpha_har = nn.Parameter(torch.tensor(-1.0))

        self.use_conf_gate = use_conf_gate

        self._reset_parameters()

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    @staticmethod
    @torch.no_grad()
    def _query_confidence(logits: torch.Tensor) -> torch.Tensor:
        """
        Compute a per-query confidence score in [0, 1]: probability that
        this query is NOT background / NOT "no-interaction".

        logits : (B, N, C+1) – last channel assumed to be the background /
                 no-object / no-action class (matches this codebase's
                 convention, e.g. pred_actions, pred_action_logits).

        Returns
        -------
        conf : (B, N, 1) – detached confidence in [0, 1], no gradient.
        """
        probs = F.softmax(logits, dim=-1)
        bg_prob = probs[..., -1:]            # (B, N, 1)
        conf = (1.0 - bg_prob).clamp(0.0, 1.0)
        return conf  # already detached via no_grad

    def forward(
        self,
        hoi_feat: torch.Tensor,            # (B, N_hoi, D)
        HHI_feat: torch.Tensor,            # (B, N_HHI, D)
        hoi_action_logits: torch.Tensor = None,  # (B, N_hoi, A+1) optional
        HHI_action_logits: torch.Tensor = None,  # (B, N_HHI, HA+1) optional
    ):
        """
        Returns
        -------
        hoi_out : (B, N_hoi, D)  – HOI features enriched with HHI context
        HHI_out : (B, N_HHI, D)  – HHI features enriched with HOI context
        """

        # ── Per-query confidence (routing signal, no gradient) ───────────────
        HHI_conf = None
        hoi_conf = None
        if self.use_conf_gate:
            if HHI_action_logits is not None:
                HHI_conf = self._query_confidence(HHI_action_logits)  # (B, N_HHI, 1)
            if hoi_action_logits is not None:
                hoi_conf = self._query_confidence(hoi_action_logits)  # (B, N_hoi, 1)

        # ── HOI queries attend to HHI keys/values ─────────────────────────────
        hoi_ctx, _ = self.hoi_from_har(query=hoi_feat, key=HHI_feat, value=HHI_feat)

        if HHI_conf is not None:

            HHI_global_conf = HHI_conf.mean(dim=1, keepdim=True)  # (B, 1, 1)
            hoi_ctx = hoi_ctx * HHI_global_conf

        if self.use_gate:
            hoi_ctx = torch.sigmoid(self.alpha_hoi) * hoi_ctx

        hoi_out = self.hoi_norm1(hoi_feat + hoi_ctx)
        if self.ffn:
            hoi_out = self.hoi_norm2(hoi_out + self.hoi_ffn(hoi_out))

        # ── HHI queries attend to HOI keys/values ─────────────────────────────
        HHI_ctx, _ = self.har_from_hoi(query=HHI_feat, key=hoi_feat, value=hoi_feat)

        if hoi_conf is not None:
            hoi_global_conf = hoi_conf.mean(dim=1, keepdim=True)  # (B, 1, 1)
            HHI_ctx = HHI_ctx * hoi_global_conf

        if self.use_gate:
            HHI_ctx = torch.sigmoid(self.alpha_har) * HHI_ctx

        HHI_out = self.har_norm1(HHI_feat + HHI_ctx)
        if self.ffn:
            HHI_out = self.har_norm2(HHI_out + self.har_ffn(HHI_out))

        return hoi_out, HHI_out


class _FFN(nn.Module):
    """Two-layer position-wise feed-forward network (inner_dim = 4 × hidden_dim)."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        inner = hidden_dim * 4
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
