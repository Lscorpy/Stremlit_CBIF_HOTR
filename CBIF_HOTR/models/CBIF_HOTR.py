# ------------------------------------------------------------------------
# Modified from HOTR (https://github.com/kakaobrain/hotr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------



#  =============================================================================
# CBIF_HOTR.py
#
# Merged CBIF_HOTR model:
#   Branch 1 — Tool triplets  (HOTR-style)
#              ⟨Human A, Swinging, Knife⟩  ⟨Human A, Holding, Bottle⟩
#              Decoder : interaction_transformer  (num_hoi_queries)
#              Heads   : H_Pointer, O_Pointer, action_embed, violence_action_embed
#
#   Branch 2 — Action triplets  (HHI-style)
#              ⟨Human A, Attacking, Human B⟩
#              Decoder : har_transformer  (num_HHI_queries)
#              Heads   : A_Pointer, V_Pointer, null_victim_embed,
#                        action_cls_embed, victim_visible
#
#   Shared   : ResNet backbone  +  DETR encoder  (when share_enc=True)
#              DETR decoder (100 object queries) for instance representations
#
# Output    : TWO separate dicts  →  out_hoi  and  out_HHI
#             wrapped in a top-level dict:
#               { "hoi": out_hoi, "HHI": out_HHI,
#                 "pred_logits": ..., "pred_boxes": ... }
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
from CBIF_HOTR.models.cross_branch_fusion import CrossBranchFusion
from CBIF_HOTR.util.misc import NestedTensor, nested_tensor_from_tensor_list
from .feed_forward import MLP


class DualBranchModel(nn.Module):
    """
    Unified model that runs both the HOTR tool-triplet branch and the
    HHI action-triplet branch on top of a shared DETR backbone + encoder.

    Parameters
    ----------
    detr                : DETR  – the base object-detection module
    # ── Branch 1 (HOI / tool triplets) ─────────────────────────────────────
    num_hoi_queries     : int   – number of pair-slot queries for Branch 1
    num_actions         : int   – number of HOI/VCOCO action classes
    num_violence_actions: int   – number of violence-verb classes (Branch 1 head)
    interaction_transformer     – transformer decoder for Branch 1
    # ── Branch 2 (HHI / action triplets) ────────────────────────────────────
    num_HHI_queries     : int   – number of pair-slot queries for Branch 2
    num_HHI_actions     : int   – number of HHI violence-action classes
    har_transformer             – transformer decoder for Branch 2
    # ── Shared options ───────────────────────────────────────────────────────
    freeze_detr         : bool  – freeze the DETR object-detection weights
    share_enc           : bool  – share the DETR encoder across both branches
    pretrained_dec      : bool  – initialise both branch decoders from DETR decoder
    temperature         : float – softmax temperature for pointer dot-products
    hoi_aux_loss        : bool  – auxiliary decoder-layer losses for Branch 1
    HHI_aux_loss        : bool  – auxiliary decoder-layer losses for Branch 2
    """
 
    def __init__(
        self,
        detr,
        # Branch 1
        num_hoi_queries,
        num_actions,
        num_violence_actions,
        interaction_transformer,
        # Branch 2
        num_HHI_queries,
        num_HHI_actions,
        har_transformer,
        # Shared
        freeze_detr,
        share_enc,
        pretrained_dec,
        temperature,
        aux_loss,
        hoi_aux_loss,
        HHI_aux_loss,
        # CBAF
        cbaf_nhead=8,
        cbaf_dropout=0.1,
        cbaf_use_gate=True,
        cbaf_ffn=True,
        cbaf_use_conf_gate=True,
        anchor_loss_weight=0.5,
    ):
        super().__init__()

        # ── DETR backbone / encoder / decoder ────────────────────────────────
        self.detr = detr
        if freeze_detr:
            for p in self.detr.parameters():
                p.requires_grad_(False)

        hidden_dim = detr.transformer.d_model
        self.aux_loss=aux_loss
        # ── Shared encoder (optional) ─────────────────────────────────────────
        # Both branch decoders will cross-attend to the SAME encoder memory.
        if share_enc:
            interaction_transformer.encoder = detr.transformer.encoder
            har_transformer.encoder         = detr.transformer.encoder

        # ── Optional: initialise branch decoders from pretrained DETR decoder ─
        if pretrained_dec:
            interaction_transformer.decoder = copy.deepcopy(detr.transformer.decoder)
            for p in interaction_transformer.decoder.parameters():
                p.requires_grad_(True)

            har_transformer.decoder = copy.deepcopy(detr.transformer.decoder)
            for p in har_transformer.decoder.parameters():
                p.requires_grad_(True)

        # =====================================================================
        # Branch 1  –  HOTR-style tool / interaction triplets
        # =====================================================================
        self.interaction_transformer = interaction_transformer
        self.hoi_query_embed = nn.Embedding(num_hoi_queries, hidden_dim)

        # Pointer heads (which DETR query is the human / which is the object)
        self.H_Pointer_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.O_Pointer_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)

        # Action head  (VCOCO / tool actions)
        self.action_embed = nn.Linear(hidden_dim, num_actions + 1)

        # Violence-verb head within Branch 1
        self.num_violence_actions = num_violence_actions
        self.violence_adapter = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.violence_action_embed = nn.Linear(hidden_dim, num_violence_actions + 1)

        self.hoi_aux_loss = hoi_aux_loss

        # =====================================================================
        # Branch 2  –  HHI-style action / aggression triplets
        # =====================================================================
        self.har_transformer = har_transformer
        self.har_query_embed = nn.Embedding(num_HHI_queries, hidden_dim)

        # Pointer heads (which DETR query is aggressor / which is victim)
        self.A_Pointer_embed   = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        self.V_Pointer_embed   = MLP(hidden_dim, hidden_dim, hidden_dim, 2)

        # Null-victim slot: lets Branch 2 predict "victim not visible"
        self.null_victim_embed = nn.Linear(hidden_dim, 1)

        # Action and visibility heads
        self.action_cls_embed  = nn.Linear(hidden_dim, num_HHI_actions + 1)
        self.victim_visible    = nn.Linear(hidden_dim, 2)

        self.HHI_aux_loss = HHI_aux_loss

        # ── Shared temperature ────────────────────────────────────────────────
        self.tau = temperature

        # =====================================================================
        # Cross-Branch Attention Fusion  (NEW)
        # =====================================================================

        self.cbaf = CrossBranchFusion(
            hidden_dim    = hidden_dim,
            nhead         = cbaf_nhead,
            dropout       = cbaf_dropout,
            use_gate      = cbaf_use_gate,
            ffn           = cbaf_ffn,
            use_conf_gate = cbaf_use_conf_gate,
        )

        self.anchor_loss_weight = anchor_loss_weight

    # =========================================================================
    # Forward
    # =========================================================================
    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # ── Backbone ──────────────────────────────────────────────────────────
        features, pos = self.detr.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        # ── DETR object-detection decoder (shared instance representations) ───
        t0 = time.time()
        hs, _ = self.detr.transformer(
            self.detr.input_proj(src), mask,
            self.detr.query_embed.weight, pos[-1]
        )
        # Normalised instance representations for dot-product pointers
        inst_repr = F.normalize(hs[-1], p=2, dim=2)   # (B, N_det, D)

        outputs_class = self.detr.class_embed(hs)      # (L, B, N_det, C)
        outputs_coord = self.detr.bbox_embed(hs).sigmoid()  # (L, B, N_det, 4)
        detr_time = time.time() - t0

        # Projected image features (reused by both branch decoders)
        img_feat = self.detr.input_proj(src)

        # =====================================================================
        # Branch 1 & 2 decoders (independent, as before)
        # =====================================================================
        t1 = time.time()
        hoi_hs = self.interaction_transformer(
            img_feat, mask, self.hoi_query_embed.weight, pos[-1]
        )[0]  # (L, B, N_hoi, D)

        t2 = time.time()
        HHI_hs = self.har_transformer(
            img_feat, mask, self.har_query_embed.weight, pos[-1]
        )[0]  # (L, B, N_HHI, D)

        # =====================================================================
        # Anchor predictions on UN-FUSED last-layer features
        # =====================================================================

        hoi_last_raw = hoi_hs[-1]   # (B, N_hoi, D)
        HHI_last_raw = HHI_hs[-1]   # (B, N_HHI, D)

        H_raw = F.normalize(self.H_Pointer_embed(hoi_last_raw), p=2, dim=-1)
        O_raw = F.normalize(self.O_Pointer_embed(hoi_last_raw), p=2, dim=-1)
        hidx_anchor_last_layer = torch.bmm(H_raw, inst_repr.transpose(1, 2)) / self.tau
        oidx_anchor_last_layer  = torch.bmm(O_raw, inst_repr.transpose(1, 2)) / self.tau

        action_anchor_hoi_last_layer    = self.action_embed(hoi_last_raw)                       # (B, N_hoi, A+1)
        violence_anchor_last_layer   = self.violence_action_embed(self.violence_adapter(hoi_last_raw))

        A_raw = F.normalize(self.A_Pointer_embed(HHI_last_raw), p=2, dim=-1)
        V_raw = F.normalize(self.V_Pointer_embed(HHI_last_raw), p=2, dim=-1)
        agg_anchor_last_layer  = torch.bmm(A_raw, inst_repr.transpose(1, 2)) / self.tau
        vic_logits_raw = torch.bmm(V_raw, inst_repr.transpose(1, 2)) / self.tau
        null_raw       = self.null_victim_embed(HHI_last_raw)
        vic_anchor_last_layer      = torch.cat([vic_logits_raw, null_raw], dim=-1)

        action_cls_anchor_HHI_last_layer = self.action_cls_embed(HHI_last_raw)
        vic_visible_anchor_last_layer  = self.victim_visible(HHI_last_raw)

        # =====================================================================
        # Confidence-gated Cross-Branch Fusion
        # =====================================================================

        hoi_last_fused, HHI_last_fused = self.cbaf(
            hoi_feat=hoi_last_raw,
            HHI_feat=HHI_last_raw,
            hoi_action_logits=violence_anchor_last_layer,
            HHI_action_logits=action_cls_anchor_HHI_last_layer,
        )

        hoi_time = time.time() - t1
        HHI_time = time.time() - t2

        # =====================================================================
        # Fused predictions (main output)
        # =====================================================================
        H_fused = F.normalize(self.H_Pointer_embed(hoi_last_fused), p=2, dim=-1)
        O_fused = F.normalize(self.O_Pointer_embed(hoi_last_fused), p=2, dim=-1)
        hidx_fused = torch.bmm(H_fused, inst_repr.transpose(1, 2)) / self.tau
        oidx_fused = torch.bmm(O_fused, inst_repr.transpose(1, 2)) / self.tau

        action_fused   = self.action_embed(hoi_last_fused)
        violence_fused = self.violence_action_embed(self.violence_adapter(hoi_last_fused))

        A_fused = F.normalize(self.A_Pointer_embed(HHI_last_fused), p=2, dim=-1)
        V_fused = F.normalize(self.V_Pointer_embed(HHI_last_fused), p=2, dim=-1)
        agg_fused = torch.bmm(A_fused, inst_repr.transpose(1, 2)) / self.tau
        vic_logits_fused = torch.bmm(V_fused, inst_repr.transpose(1, 2)) / self.tau
        null_fused       = self.null_victim_embed(HHI_last_fused)
        vic_fused        = torch.cat([vic_logits_fused, null_fused], dim=-1)

        action_cls_fused  = self.action_cls_embed(HHI_last_fused)
        vic_visible_fused = self.victim_visible(HHI_last_fused)

        # =====================================================================
        # Build per-layer pointer/action lists for AUX losses
        # =====================================================================
        H_ori_all = F.normalize(self.H_Pointer_embed(hoi_hs), p=2, dim=-1)
        O_ori_all = F.normalize(self.O_Pointer_embed(hoi_hs), p=2, dim=-1)
        outputs_hidx_ori_all = [torch.bmm(h, inst_repr.transpose(1, 2)) / self.tau for h in H_ori_all]
        outputs_oidx_ori_all = [torch.bmm(o, inst_repr.transpose(1, 2)) / self.tau for o in O_ori_all]
        outputs_action_ori_all    = self.action_embed(hoi_hs)
        outputs_violence_ori_all  = self.violence_action_embed(self.violence_adapter(hoi_hs))

        A_ori_all = F.normalize(self.A_Pointer_embed(HHI_hs), p=2, dim=-1)
        V_ori_all = F.normalize(self.V_Pointer_embed(HHI_hs), p=2, dim=-1)
        agg_ori_all_list = [torch.bmm(a, inst_repr.transpose(1, 2)) / self.tau for a in A_ori_all]
        vic_logits_ori_all_list = [torch.bmm(v, inst_repr.transpose(1, 2)) / self.tau for v in V_ori_all]
        null_ori_all = self.null_victim_embed(HHI_hs)
        vic_ori_all_list = [
            torch.cat([v_logit, n_logit], dim=-1)
            for v_logit, n_logit in zip(vic_logits_ori_all_list, null_ori_all.unbind(0))
        ]
        action_cls_ori_all  = self.action_cls_embed(HHI_hs)
        vic_visible_ori_all = self.victim_visible(HHI_hs)

        # =====================================================================
        # Output dicts
        # =====================================================================

        # ── Branch 1 (HOI) ────────────────────────────────────────────────────
        out_hoi = {
            "pred_logits": outputs_class[-1],
            "pred_boxes":  outputs_coord[-1],

            # Reported / main predictions = FUSED
            "pred_hidx":             hidx_fused,
            "pred_oidx":             oidx_fused,
            "pred_actions":          action_fused,
            "pred_violence_actions": violence_fused,

            # Anchor predictions (un-fused)

            "pred_hidx_anchor":             hidx_anchor_last_layer,
            "pred_oidx_anchor":             oidx_anchor_last_layer,
            "pred_actions_anchor":          action_anchor_hoi_last_layer,
            "pred_violence_actions_anchor": violence_anchor_last_layer,

            "hoi_recognition_time": max(hoi_time - detr_time, 0),
        }

        # ── Branch 2 (HHI) ────────────────────────────────────────────────────
        out_HHI= {
            "pred_logits":     outputs_class[-1],
            "pred_boxes":      outputs_coord[-1],
            "pred_human_box":  outputs_coord[-1],

            # Reported / main predictions = FUSED
            "pred_aggressor_idx":  agg_fused,
            "pred_victim_idx":     vic_fused,
            "pred_action_logits":  action_cls_fused,
            "pred_victim_visible": vic_visible_fused,

            # Anchor predictions (un-fused)
            "pred_aggressor_idx_anchor":  agg_anchor_last_layer,
            "pred_victim_idx_anchor":     vic_anchor_last_layer,
            "pred_action_logits_anchor":  action_cls_anchor_HHI_last_layer,
            "pred_victim_visible_anchor": vic_visible_anchor_last_layer,

            "HHI_recognition_time": max(HHI_time - detr_time, 0),
        }

        # ── Auxiliary losses ──────────────────────────────────────────────────
        if self.hoi_aux_loss:
            out_hoi["hoi_aux_outputs"] = self._hoi_aux_loss(
                outputs_class, outputs_coord,
                outputs_hidx_ori_all, outputs_oidx_ori_all,
                outputs_action_ori_all, outputs_violence_ori_all,
            )

        if self.HHI_aux_loss:
            out_HHI["HHI_aux_outputs"] = self._HHI_aux_loss(
                outputs_class, outputs_coord,
                agg_ori_all_list, vic_ori_all_list,
                action_cls_ori_all, vic_visible_ori_all,
            )


        # ── Top-level combined dict ───────────────────────────────────────────
        # Criterion and post-processors receive the branch-specific sub-dict.
        # Callers can also inspect the top-level dict for shared DETR outputs.
        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes":  outputs_coord[-1],
            "hoi": out_hoi,
            "hhi": out_HHI,
        }

        if self.aux_loss:
            out["aux_outputs"]=self._set_aux_loss(outputs_class, outputs_coord)

        # # ── Debug: Print Shapes and Key-Value Summary ───────────────────────────────
        # print("\n" + "="*40 + " DEBUG: OUT DICT STRUCTURE " + "="*40)

        # def debug_print_dict(d, indent=0):
        #     spacing = "  " * indent
        #     for key, value in d.items():
        #         # Case 1: Nested Dictionaries (like 'hoi', 'HHI', 'aux_outputs')
        #         if isinstance(value, dict):
        #             print(f"{spacing}📂 Key: '{key}' (dict)")
        #             debug_print_dict(value, indent + 1)
                    
        #         # Case 2: PyTorch Tensors
        #         elif hasattr(value, "shape"):
        #             print(f"{spacing}📄 Key: '{key}' | Type: Tensor | Shape: {list(value.shape)} | Device: {value.device}")

        # debug_print_dict(out)
        # print("="*107 + "\n")
        return out
    
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):

        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
    # =========================================================================
    # Auxiliary-loss helpers  –  Branch 1 (HOI)
    # =========================================================================
    @torch.jit.unused
    def _hoi_aux_loss(
        self,
        outputs_class, outputs_coord,
        outputs_hidx, outputs_oidx,
        outputs_action, outputs_violence,
    ):
        return [
            {
                "pred_logits":           a,
                "pred_boxes":            b,
                "pred_hidx":             c,
                "pred_oidx":             d,
                "pred_actions":          e,
                "pred_violence_actions": f,
            }
            for a, b, c, d, e, f in zip(
                outputs_class[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                outputs_coord[-1:].repeat((outputs_action.shape[0], 1, 1, 1)),
                outputs_hidx[:-1],
                outputs_oidx[:-1],
                outputs_action[:-1],
                outputs_violence[:-1],
            )
        ]


    # =========================================================================
    # Auxiliary-loss helpers  –  Branch 2 (HHI)
    # =========================================================================
    @torch.jit.unused
    def _HHI_aux_loss(
        self,
        outputs_class, outputs_coord,
        aggressor_outputs_idx, victim_outputs_idx,
        action_outputs_class, vic_visible,
    ):
        return [
            {
                "pred_logits":         a,
                "pred_boxes":          b,
                "pred_human_box":      c,
                "pred_aggressor_idx":  d,
                "pred_victim_idx":     e,
                "pred_action_logits":  f,
                "pred_victim_visible": g,
            }
            for a, b, c, d, e, f, g in zip(
                outputs_class[-1:].repeat((action_outputs_class.shape[0], 1, 1, 1)),
                outputs_coord[-1:].repeat((action_outputs_class.shape[0], 1, 1, 1)),
                outputs_coord[-1:].repeat((action_outputs_class.shape[0], 1, 1, 1)),
                aggressor_outputs_idx[:-1],
                victim_outputs_idx[:-1],
                action_outputs_class[:-1],
                vic_visible[:-1],
            )
        ]
