# =============================================================================
# unified_postprocess.py
#
# Single post-processor for CBIF_HOTR.
#
# Output format (per image, same dict for both branches)
# -------------------------------------------------------
# {
#   # ── Shared DETR detections ────────────────────────────────────────────
#   'scores'  : (K,)          – per-query max class confidence
#   'labels'  : (K,)          – per-query predicted class id
#   'boxes'   : (K, 4)        – all query boxes in pixel xyxy
#   'h_box'   : (n_h, 4)      – boxes of detected humans
#   'h_cat'   : (n_h,)        – confidence of detected humans
#   'o_box'   : (n_o+1, 4)    – boxes of detected objects (+1 null slot)
#   'o_cat'   : (n_o+1,)      – confidence of detected objects
#
#   # ── Branch 1  (HOI / tool triplets) ──────────────────────────────────
#   'pair_score_vcoco'    : (n_act_vcoco, n_h, n_o+1)
#   'pair_score_violence' : (n_act_violence, n_h, n_o+1)
#
#   # ── Branch 2  (HHI / action-aggression triplets) ─────────────────────
#   # Same (n_act, K, K+1) grid convention so evaluators are interchangeable.
#   # Null column (index K) = "victim not visible".
#   'pair_score_HHI'      : (n_act_HHI, n_h, n_o+1)
#   'HHI_victim_visible'  : (n_h, n_o+1) bool – True where victim is visible
#
#   'hoi_recognition_time': float
# }
# =============================================================================

import time
import torch
import torch.nn.functional as F
from torch import nn
from CBIF_HOTR.util import box_ops


class UnifiedPostProcess(nn.Module):
    """
    Unified post-processor for DualBranchModel.

    Parameters
    ----------
    HOIDet   : bool  – enables Branch 1 (tool triplet) scoring
    HHIDet   : bool  – enables Branch 2 (action triplet) scoring
    threshold: float – minimum DETR query confidence to be considered
                       a human (label==1) or object candidate
    """



    def __init__(self, HOIDet: bool, HHIDet: bool, threshold: float = 0.0):
        super().__init__()
        self.HOIDet    = HOIDet
        self.HHIDet    = HHIDet
        self.threshold = threshold

    # =========================================================================
    @torch.no_grad()
    def forward(self, outputs: dict, target_sizes: torch.Tensor, **kwargs):
        """
        Parameters
        ----------
        outputs      : top-level dict from DualBranchModel.forward()
                       Must contain keys "hoi", "HHI", "pred_logits", "pred_boxes"
        target_sizes : (B, 2) tensor – (H, W) for each image in pixels

        Returns
        -------
        list[dict]  – one result dict per image (see module docstring for keys)
        """
        # ── Shared DETR detections ────────────────────────────────────────────
        # Use top-level pred_logits / pred_boxes (identical across both branches)
        out_logits = outputs["pred_logits"]   # (B, K, C)
        out_bbox   = outputs["pred_boxes"]    # (B, K, 4)  normalised cxcywh

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob         = F.softmax(out_logits, dim=-1)
        scores, labels = prob[..., :-1].max(-1)   # (B, K)

        # Convert boxes to pixel xyxy
        boxes      = box_ops.box_cxcywh_to_xyxy(out_bbox)          # (B, K, 4)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct  = torch.stack([img_w, img_h, img_w, img_h], dim=1)  # (B, 4)
        boxes      = boxes * scale_fct[:, None, :]                  # (B, K, 4)

        K = boxes.shape[1]

        # ── Branch outputs ────────────────────────────────────────────────────
        out_hoi = outputs.get("hoi", {})
        out_HHI = outputs.get("hhi", {})

        # Branch 1 timing baseline
        t_hoi = out_hoi.get("hoi_recognition_time", 0.0)
        t_HHI = out_HHI.get("hhi_recognition_time", 0.0)
        out_time = max(t_hoi, t_HHI)

        # ── Branch 1: pre-compute across batch ───────────────────────────────
        if self.HOIDet and out_hoi:
            pair_act_vcoco    = torch.sigmoid(out_hoi["pred_actions"])           # (B, Q1, A+1)
            pair_act_violence = torch.sigmoid(out_hoi["pred_violence_actions"])  # (B, Q1, V+1)
            h_prob_hoi = F.softmax(out_hoi["pred_hidx"], dim=-1)  # (B, Q1, K)
            o_prob_hoi = F.softmax(out_hoi["pred_oidx"], dim=-1)  # (B, Q1, K)
            h_idx_score_hoi, h_indices_hoi = h_prob_hoi.max(-1)   # (B, Q1)
            o_idx_score_hoi, o_indices_hoi = o_prob_hoi.max(-1)   # (B, Q1)

        # ── Branch 2: pre-compute across batch ───────────────────────────────
        if self.HHIDet and out_HHI:
            pair_act_HHI = torch.sigmoid(out_HHI["pred_action_logits"])  # (B, Q2, HA+1)
            a_prob = F.softmax(out_HHI["pred_aggressor_idx"], dim=-1)    # (B, Q2, K)

            v_prob = F.softmax(out_HHI["pred_victim_idx"],    dim=-1)    # (B, Q2, K+1)
            a_idx_score, a_indices = a_prob.max(-1)                      # (B, Q2)
            v_idx_score, v_indices = v_prob.max(-1)                      # (B, Q2)  may equal K

            vis_prob   = F.softmax(out_HHI["pred_victim_visible"], dim=-1)  # (B, Q2, 2)
            is_visible = vis_prob.argmax(-1) == 1                           # (B, Q2) bool

        # =====================================================================
        # Per-image assembly
        # =====================================================================
        start_time = time.time()
        results    = []

        for b in range(boxes.shape[0]):
            s = scores[b]   # (K,)
            l = labels[b]   # (K,)
            bx = boxes[b]   # (K, 4)

            # Human / object masks
            h_inds = (l == 1) & (s > self.threshold)   # person queries
            o_inds = (s > self.threshold)               # any confident query

            h_box = bx[h_inds]
            h_cat = s[h_inds]
            o_box = bx[o_inds]
            o_cat = s[o_inds]

            # Append null object slot (V-COCO scenario-1 / HHI invisible-victim)
            o_inds = torch.cat((o_inds, torch.ones(1, dtype=torch.bool, device=o_inds.device)))
            o_box  = torch.cat((o_box,  torch.zeros(1, 4, device=o_box.device)))
            o_cat  = torch.cat((o_cat,  torch.zeros(1,    device=o_cat.device)))

            result_dict = {
                "scores": s,
                "labels": l,
                "boxes":  bx,
                "h_box":  h_box,
                "h_cat":  h_cat,
                "o_box":  o_box,
                "o_cat":  o_cat,
            }

            # ── Branch 1 scoring ──────────────────────────────────────────────
            if self.HOIDet and out_hoi:
                n_act   = pair_act_vcoco[b][:, :-1].shape[-1]
                n_act_V = pair_act_violence[b][:, :-1].shape[-1]

                # Accumulation grids  (n_act, K, K+1)
                score    = torch.zeros(n_act,   K, K + 1, device=s.device)
                score_V  = torch.zeros(n_act_V, K, K + 1, device=s.device)
                id_score   = torch.zeros(K, K + 1, device=s.device)
                id_score_V = torch.zeros(K, K + 1, device=s.device)
                sorted_score   = torch.zeros_like(score)
                sorted_score_V = torch.zeros_like(score_V)

                for ( h_idx, o_idx,
                     pa_vcoco, pa_violence) in zip(
                         h_indices_hoi[b],
                         o_indices_hoi[b],
                        pair_act_vcoco[b],  pair_act_violence[b]):

                    match_v  = 1 - pa_vcoco[-1]    # "interaction" confidence
                    match_vv = 1 - pa_violence[-1]

                    o_slot = o_idx if h_idx != o_idx else torch.tensor(-1, device=s.device)

                    # VCOCO scoring
                    if match_v > id_score[h_idx, o_slot]:
                        id_score[h_idx, o_slot] = match_v
                        sorted_score[:, h_idx, o_slot] = match_v * pa_vcoco[:-1]
                    score[:, h_idx, o_slot] += match_v * pa_vcoco[:-1]

                    # Violence scoring
                    if match_vv > id_score_V[h_idx, o_slot]:
                        id_score_V[h_idx, o_slot] = match_vv
                        sorted_score_V[:, h_idx, o_slot] = match_vv * pa_violence[:-1]
                    score_V[:, h_idx, o_slot] += match_vv * pa_violence[:-1]


                score   = (score + sorted_score)[:, h_inds, :][:, :, o_inds]
                score_V = (score_V + sorted_score_V)[:, h_inds, :][:, :, o_inds]

                result_dict.update({
                    "pair_score_vcoco":    score,    # (n_act_vcoco,  n_h, n_o+1)
                    "pair_score_violence": score_V,  # (n_act_violence, n_h, n_o+1)
                })

            # ── Branch 2 scoring ──────────────────────────────────────────────

            if self.HHIDet and out_HHI:
                n_act_H = pair_act_HHI[b][:, :-1].shape[-1]

                score_H    = torch.zeros(n_act_H, K, K + 1, device=s.device)
                id_score_H = torch.zeros(K, K + 1,          device=s.device)
                sorted_H   = torch.zeros_like(score_H)

                # Visibility grid (bool) – aligned with score_H
                vis_grid   = torch.zeros(K, K + 1, dtype=torch.bool, device=s.device)

                for (ag_s, ag_idx,
                     vi_s, vi_idx,
                     vis_flag,
                     pa_HHI) in zip(
                        a_idx_score[b], a_indices[b],
                        v_idx_score[b], v_indices[b],
                        is_visible[b],
                        pair_act_HHI[b]):

                    match_H = 1 - pa_HHI[-1]   # "interaction" confidence

                    # ── Case A: genuine null-victim prediction 
                    is_null = (vi_idx >= K)

                    # ── Case B: pointer collision (NOT a visibility judgment) 
                    self_interaction = (not is_null) and (ag_idx == vi_idx)
                    if self_interaction:
                        continue
                    # ── Real victim slot (only reached for non-collision cases)
                    v_slot = torch.tensor(-1, device=s.device) if is_null else vi_idx
                    joint = ag_s * (torch.tensor(1.0, device=s.device) if is_null else vi_s)

                    if match_H > id_score_H[ag_idx, v_slot]:
                        id_score_H[ag_idx, v_slot]  = match_H
                        sorted_H[:, ag_idx, v_slot] = match_H * joint * pa_HHI[:-1]

                        vis_grid[ag_idx, v_slot]    = False if is_null else vis_flag

                    score_H[:, ag_idx, v_slot] += match_H * joint * pa_HHI[:-1]

                # Branch 2
                score_H  = (score_H + sorted_H)[:, h_inds, :][:, :, o_inds]
                vis_grid = vis_grid[h_inds, :][:, o_inds]

                result_dict.update({
                    "pair_score_HHI":     score_H,    # (n_act_HHI, n_h, n_o+1)
                    "HHI_victim_visible": vis_grid,   # (n_h, n_o+1)  bool
                })

            result_dict["hoi_recognition_time"] = (time.time() - start_time) + out_time
            results.append(result_dict)

        return results
