# ------------------------------------------------------------------------
# Modified from HOTR (https://github.com/kakaobrain/hotr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

# =============================================================================
# CBIF_HOTR.py
#
# Merged criterion for the DualBranchModel.
#
# Design
# ------
#   • One forward() call covers BOTH branches.
#   • Losses from Branch 1 (HOI) and Branch 2 (HHI) are accumulated into a
#     single `losses` dict with non-overlapping key prefixes:
#       hoi_*   →  Branch 1 losses  (loss_hidx, loss_oidx, loss_act,
#                                    loss_violence, loss_hidx_v, loss_oidx_v)
#       hhi_*   →  Branch 2 losses  (loss_aggressor, loss_victim,
#                                    loss_action, loss_visibility)
#       loss_ce, loss_bbox, loss_giou  →  shared DETR losses
#
#   • A single scalar `loss_total` is assembled using `combined_weight_dict`
#     (= hoi_weight_dict ∪ HHI_weight_dict ∪ detr_weight_dict).
#     `.backward()` is called once on that scalar by the training loop.
#
#   • Per-branch sub-totals (loss_hoi_total, loss_HHI_total) are logged but
#     do NOT need separate backward passes.
#
# Matchers required
# -----------------
#   detr_matcher        – standard Hungarian matcher for object detection
#   human_matcher       – DETR matcher restricted to person queries (for HHI)
#   hoi_matcher         – HOTR Hungarian matcher (Branch 1 pairs)
#   HHI_matcher         – HHI Hungarian matcher (Branch 2 pairs)
# =============================================================================

import copy
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from CBIF_HOTR.util import box_ops
from CBIF_HOTR.util.misc import (accuracy, get_world_size, is_dist_avail_and_initialized)


class DualBranchCriterion(nn.Module):
    """
    Parameters
    ----------
    num_classes         : int   – number of DETR object classes (excl. background)
    detr_matcher        : nn.Module  – matches DETR predictions → GT boxes
    human_matcher       : nn.Module  – matches DETR predictions → GT *human* boxes
    hoi_matcher         : nn.Module  – matches HOI pair queries  → GT HOI pairs
    HHI_matcher         : nn.Module  – matches HHI pair queries  → GT HHI triplets
    hoi_weight_dict     : dict  – loss weights for Branch 1 (keys: loss_hidx, …)
    HHI_weight_dict     : dict  – loss weights for Branch 2 (keys: loss_aggressor, …)
    detr_weight_dict    : dict  – loss weights for sHHIed DETR losses
    eos_coef            : float – background weight for DETR class loss
    detr_losses         : list  – which DETR losses to compute ('labels','boxes',…)
    hoi_losses          : list  – which HOI losses to compute  ('pair_labels',…)
    HHI_losses          : list  – which HHI losses to compute  ('action_vHHI',…)
    args                : argparse.Namespace

        anchor_loss_weight  : float – weight applied to the un-fused anchor losses
                                  relative to the fused-prediction losses
                                  (default 0.5; can also be read from
                                  args.anchor_loss_weight if present)
    """

    def __init__(
        self,
        num_classes,
        detr_matcher,
        human_matcher,
        hoi_matcher,
        HHI_matcher,
        hoi_weight_dict,
        HHI_weight_dict,
        detr_weight_dict,
        eos_coef,
        detr_losses,
        hoi_losses,
        HHI_losses,
        anchor_loss_weight=0.5,
        args=None,
    ):
        super().__init__()

        self.num_classes      = num_classes
        self.matcher          = detr_matcher
        self.human_matcher    = human_matcher
        self.hoi_matcher      = hoi_matcher
        self.HHI_matcher      = HHI_matcher

        self.hoi_weight_dict  = hoi_weight_dict
        self.HHI_weight_dict  = HHI_weight_dict
        self.detr_weight_dict = detr_weight_dict
        
        self.eos_coef         = eos_coef
        self.losses           = detr_losses
        self.HOI_losses       = hoi_losses
        self.HHI_losses       = HHI_losses

        self.anchor_loss_weight= anchor_loss_weight

        # Combined weight dict used to compute loss_total
        self.combined_weight_dict = {}
        self.combined_weight_dict.update(detr_weight_dict)
        self.combined_weight_dict.update(hoi_weight_dict)
        self.combined_weight_dict.update(HHI_weight_dict)

        # DETR class-weight buffer
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef

        # Optional per-class tuning (mirrors original criterion)
        if num_classes >= 93:
            empty_weight[91] = 0.5   # gun  – reduce FP reward
            empty_weight[92] = 2.0   # stick – rare class boost
        self.register_buffer("empty_weight", empty_weight)

        # Dataset-specific ids read from args
        self.invalid_ids_vcoco = args.invalid_ids_vcoco
        self.valid_ids_vcoco   = np.concatenate(
            (args.valid_ids_vcoco, [-1]), axis=0
        )
        self.valid_ids_violence = args.valid_ids_violence


    # =========================================================================
    # Utility helpers
    # =========================================================================
    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx   = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _remap_anchor_keys(outputs: dict, key_pairs) -> dict:
        """
        Build a shallow-copied dict where each `plain_key` is overwritten by
        the value currently stored under `anchor_key`. Used so the EXISTING
        loss functions (which look for e.g. "pred_hidx") can be reused
        unmodified to compute the anchor loss, just by feeding them the
        anchor tensor under the name they expect.

        key_pairs : list of (plain_key, anchor_key) tuples.
        """
        remapped = dict(outputs)  # shallow copy is enough; we only swap top-level keys
        for plain_key, anchor_key in key_pairs:
            if anchor_key in outputs:
                remapped[plain_key] = outputs[anchor_key]
        return remapped
    # =========================================================================
    # ── Shared DETR losses ────────────────────────────────────────────────────
    # =========================================================================
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes   = torch.full(
            src_logits.shape[:2], self.num_classes,
            dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses  = {"loss_ce": loss_ce}
        if log:
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs["pred_logits"]
        device      = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        card_pred   = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err    = F.l1_loss(card_pred.float(), tgt_lengths.float())
        return {"cardinality_error": card_err}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        assert "pred_boxes" in outputs
        idx        = self._get_src_permutation_idx(indices)
        src_boxes  = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox  = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses     = {"loss_bbox": loss_bbox.sum() / num_boxes}
        loss_giou  = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes),
        ))
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels":      self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes":       self.loss_boxes,
        }
        assert loss in loss_map, f"Unknown DETR loss: {loss}"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    # =========================================================================
    # ── Branch 1 (HOI) losses ──────────────────────────────────────────────── 
    # =========================================================================

    # ── Pointer loss (human + object slots) ───────────────────────────────────
    def loss_pair_labels(self, outputs, targets, hoi_indices, log=False):
        """Cross-entropy on which DETR query each HOI slot points to."""
        assert "pred_hidx" in outputs and "pred_oidx" in outputs
        src_hidx = outputs["pred_hidx"]
        src_oidx = outputs["pred_oidx"]
        device   = src_hidx.device
        idx      = self._get_src_permutation_idx(hoi_indices)

        target_hidx = torch.full(src_hidx.shape[:2], -1, dtype=torch.int64, device=device)
        target_oidx = torch.full(src_oidx.shape[:2], -1, dtype=torch.int64, device=device)

        target_classes_h = torch.cat([t["h_labels"][J]  for t, (_, J) in zip(targets, hoi_indices)])
        target_classes_o = torch.cat([t["o_labels"][J]  for t, (_, J) in zip(targets, hoi_indices)])
        targets_vcoco    = torch.cat([t["has_vcoco_labels"][J] for t, (_, J) in zip(targets, hoi_indices)])

        target_hidx[idx] = target_classes_h
        target_oidx[idx] = target_classes_o

        # Masked pointer loss (zero-out pairs with no VCOCO label)
        loss_h = F.cross_entropy(src_hidx.transpose(1, 2), target_hidx, ignore_index=-1, reduction="none")
        loss_o = F.cross_entropy(src_oidx.transpose(1, 2), target_oidx, ignore_index=-1, reduction="none")

        matched_h = loss_h[idx] * targets_vcoco
        matched_o = loss_o[idx] * targets_vcoco
        denom     = targets_vcoco.sum()
        if denom > 0:
            loss_h = matched_h.sum() / denom
            loss_o = matched_o.sum() / denom
        else:
            loss_h = src_hidx.sum() * 0.0
            loss_o = src_oidx.sum() * 0.0

        return {"loss_hidx": loss_h, "loss_oidx": loss_o}

    # ── Action classification loss (VCOCO/tool verbs) ─────────────────────────
    def loss_pair_actions(self, outputs, targets, hoi_indices):
        assert "pred_actions" in outputs
        src_actions = outputs["pred_actions"]
        idx         = self._get_src_permutation_idx(hoi_indices)

        target_classes_o = torch.cat([t["pair_actions"][J] for t, (_, J) in zip(targets, hoi_indices)])
        target_classes   = torch.full(src_actions.shape, 0, dtype=torch.float32, device=src_actions.device)
        target_classes[..., -1] = 1  # background slot default

        pos_classes = torch.zeros(target_classes[idx].shape, dtype=torch.float32, device=src_actions.device)
        pos_classes[:, :-1] = target_classes_o.float()
        target_classes[idx] = pos_classes

        # Focal BCE
        logits    = src_actions.sigmoid()
        loss_bce  = F.binary_cross_entropy(
            logits[..., self.valid_ids_vcoco],
            target_classes[..., self.valid_ids_vcoco],
            reduction="none",
        )
        p_t       = (logits[..., self.valid_ids_vcoco] * target_classes[..., self.valid_ids_vcoco]
                     + (1 - logits[..., self.valid_ids_vcoco]) * (1 - target_classes[..., self.valid_ids_vcoco]))
        alpha_t   = (0.25 * target_classes[..., self.valid_ids_vcoco]
                     + 0.75 * (1 - target_classes[..., self.valid_ids_vcoco]))
        loss_focal = alpha_t * (1 - p_t) ** 2 * loss_bce
        denom      = max(target_classes[..., self.valid_ids_vcoco[:-1]].sum(), 1)
        return {"loss_act": loss_focal.sum() / denom}

    # ── Violence-verb loss (Branch 1 sub-head) ────────────────────────────────
    def loss_pair_violence(self, outputs, targets, hoi_indices):
        assert "pred_violence_actions" in outputs
        src_violence = outputs["pred_violence_actions"]
        idx          = self._get_src_permutation_idx(hoi_indices)

        target_violence_o = torch.cat([t["pair_violence"][J] for t, (_, J) in zip(targets, hoi_indices)], dim=0)
        target_violence   = torch.full(src_violence.shape, 0, dtype=torch.float32, device=src_violence.device)
        target_violence[..., -1] = 1

        pos_classes = torch.zeros(target_violence[idx].shape, dtype=torch.float32, device=src_violence.device)
        pos_classes[:, :-1] = target_violence_o.float()
        target_violence[idx] = pos_classes

        logits   = src_violence.sigmoid()
        # loss_bce = F.binary_cross_entropy_with_logits(logits, target_violence, reduction="none")
        loss_bce = F.binary_cross_entropy(logits, target_violence, reduction="none")
        p_t      = logits * target_violence + (1 - logits) * (1 - target_violence)
        alpha_t  = 0.25 * target_violence + 0.75 * (1 - target_violence)
        loss_focal = alpha_t * (1 - p_t) ** 2 * loss_bce
        denom      = max(target_violence[..., :-1].sum(), 1)
        return {"loss_violence": loss_focal.sum() / denom}

    # ── Violence pointer loss (separate Hungarian assignment) ─────────────────
    def loss_pair_violence_labels(self, outputs, targets, hoi_indices_V, log=False):
        assert "pred_hidx" in outputs and "pred_oidx" in outputs
        src_hidx = outputs["pred_hidx"]
        src_oidx = outputs["pred_oidx"]
        device   = src_hidx.device
        idx      = self._get_src_permutation_idx(hoi_indices_V)

        target_hidx = torch.full(src_hidx.shape[:2], -1, dtype=torch.int64, device=device)
        target_oidx = torch.full(src_oidx.shape[:2], -1, dtype=torch.int64, device=device)

        target_classes_h = torch.cat([t["h_labels_v"][J] for t, (_, J) in zip(targets, hoi_indices_V)])
        target_classes_o = torch.cat([t["o_labels_v"][J] for t, (_, J) in zip(targets, hoi_indices_V)])

        target_hidx[idx] = target_classes_h
        target_oidx[idx] = target_classes_o

        loss_h_v = F.cross_entropy(src_hidx.transpose(1, 2), target_hidx, ignore_index=-1)
        loss_o_v = F.cross_entropy(src_oidx.transpose(1, 2), target_oidx, ignore_index=-1)
        return {"loss_hidx_v": loss_h_v, "loss_oidx_v": loss_o_v}

    def get_hoi_loss(self, loss, outputs, targets, indices, **kwargs):
        """Dispatcher for Branch 1 HOI losses."""
        # Guard: if no VCOCO pairs in batch, short-circuit action loss
        any_vcoco = any(
            (t["has_vcoco_labels"] == 1).any().item() if "has_vcoco_labels" in t else True
            for t in targets
        )

        loss_map = {
            "pair_labels":           self.loss_pair_labels,
            "pair_actions":          self.loss_pair_actions,
            "pair_violence":         self.loss_pair_violence,
            "pair_violence_labels":  self.loss_pair_violence_labels,
        }
        if not any_vcoco:
            loss_map["pair_actions"] = lambda *a, **kw: {
                "loss_act": torch.tensor(0.0, device=outputs["pred_actions"].device, requires_grad=True)
            }

        assert loss in loss_map, f"Unknown HOI loss: {loss}"
        return loss_map[loss](outputs, targets, indices, **kwargs)

    # =========================================================================
    # ── Branch 2 (HHI) losses ────────────────────────────────────────────────
    # =========================================================================

    # ── Human-box regression loss (HHI uses its own GT boxes) ────────────────
    def loss_human_boxes(self, outputs, targets, detr_human_indices, num_boxes):
        assert "pred_human_box" in outputs
        idx        = self._get_src_permutation_idx(detr_human_indices)
        src_boxes  = outputs["pred_human_box"][idx]
        target_boxes = torch.cat(
            [t["human_boxes"][i] for t, (_, i) in zip(targets, detr_human_indices)], dim=0
        )
        loss_bbox  = F.l1_loss(src_boxes, target_boxes, reduction="none")
        losses     = {"loss_human_bbox": loss_bbox.sum() / num_boxes}
        loss_giou  = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes),
        ))
        losses["loss_human_giou"] = loss_giou.sum() / num_boxes
        return losses

    # ── Violence action classification (focal BCE, HHI head) ──────────────────
    def loss_action_HHI(self, outputs, targets, indices):
        assert "pred_action_logits" in outputs
        pred_action_logits = outputs["pred_action_logits"]
        idx    = self._get_src_permutation_idx(indices)
        device = pred_action_logits.device

        gt_action_vectors = torch.cat([t["violence_actions"][J] for t, (_, J) in zip(targets, indices)])
        target_classes    = torch.full(pred_action_logits.shape, 0, dtype=torch.float32, device=device)
        target_classes[..., -1] = 1  # background default

        if gt_action_vectors.shape[0] > 0:
            pos_classes = torch.zeros(target_classes[idx].shape, dtype=torch.float32, device=device)
            pos_classes[:, :-1] = gt_action_vectors.float()
            target_classes[idx] = pos_classes

        logits     = pred_action_logits.sigmoid()
        loss_bce   = F.binary_cross_entropy(logits, target_classes, reduction="none")
        p_t        = logits * target_classes + (1 - logits) * (1 - target_classes)
        alpha_t    = 0.25 * target_classes + 0.75 * (1 - target_classes)
        loss_focal = alpha_t * (1 - p_t) ** 2 * loss_bce
        denom      = max(target_classes[..., :-1].sum(), 1)
        return {"loss_action": loss_focal.sum() / denom}

    # ── Aggressor / victim pointer loss (HHI) ─────────────────────────────────
    def loss_pointer_HHI(self, outputs, targets, indices):
        assert "pred_aggressor_idx" in outputs and "pred_victim_idx" in outputs
        agg_logits = outputs["pred_aggressor_idx"]
        vic_logits = outputs["pred_victim_idx"]
        device     = agg_logits.device
        idx        = self._get_src_permutation_idx(indices)


        target_agg = torch.full(agg_logits.shape[:2], -1, dtype=torch.int64, device=device)
        target_vic = torch.full(vic_logits.shape[:2], -1, dtype=torch.int64, device=device)

        gt_agg  = torch.cat([t["a_labels"][J]             for t, (_, J) in zip(targets, indices)])
        gt_vic  = torch.cat([t["v_labels"][J]             for t, (_, J) in zip(targets, indices)])

        target_agg[idx] = gt_agg
        target_vic[idx] = gt_vic 

        active_agg = (target_agg != -1).sum()
        loss_agg   = (
            F.cross_entropy(agg_logits.transpose(1, 2), target_agg, ignore_index=-1)
            if active_agg > 0
            else torch.tensor(0.0, device=device, requires_grad=True)
        )

        active_vic = (target_vic != -1).sum()
        loss_vic   = (
            F.cross_entropy(vic_logits.transpose(1, 2), target_vic, ignore_index=-1)
            if active_vic > 0
            else torch.tensor(0.0, device=device, requires_grad=True)
        )

        return {"loss_aggressor": loss_agg, "loss_victim": loss_vic}

    # ── Victim-visibility classification loss (HHI) ───────────────────────────
    def loss_visibility_HHI(self, outputs, targets, indices):
        assert "pred_victim_visible" in outputs
        pred_vis = outputs["pred_victim_visible"]
        idx      = self._get_src_permutation_idx(indices)

        target_vis = torch.full(pred_vis.shape[:2], -1, dtype=torch.int64, device=pred_vis.device)
        gt_visible = torch.cat([t["has_target_visible"][J].long() for t, (_, J) in zip(targets, indices)])
        target_vis[idx] = gt_visible

        active = (target_vis != -1).sum()
        loss_vis = (
            F.cross_entropy(pred_vis.transpose(1, 2), target_vis, ignore_index=-1)
            if active > 0
            else torch.tensor(0.0, device=pred_vis.device, requires_grad=True)
        )
        return {"loss_visibility": loss_vis}

    def get_HHI_loss(self, loss, outputs, targets, indices, **kwargs):
        """Dispatcher for Branch 2 HHI losses."""
        loss_map = {
            "action_HHI":     self.loss_action_HHI,
            "pointer_HHI":    self.loss_pointer_HHI,
            "visibility_HHI": self.loss_visibility_HHI,
        }
        assert loss in loss_map, f"Unknown HHI loss: {loss}"
        return loss_map[loss](outputs, targets, indices, **kwargs)

    # =========================================================================
    # Forward
    # =========================================================================
    def forward(self, outputs, targets, log=False):
        """
        Parameters
        ----------
        outputs : dict returned by DualBranchModel.forward()
                  Expected top-level keys: "hoi", "HHI", "pred_logits", "pred_boxes"
        targets : list of per-image GT dicts
        log     : bool – passed to matchers for logging

        Returns
        -------
        losses  : dict containing all individual losses AND
                  loss_hoi_total, loss_HHI_total, loss_total
        """
        out_hoi = outputs["hoi"]
        out_HHI = outputs["hhi"]

        # Strips auxiliary outputs from the main prediction dict before matching
        out_hoi_without_aux = {k: v for k, v in out_hoi.items() if k not in ['aux_outputs', 'hoi_aux_outputs']}
        out_HHI_without_aux = {k: v for k, v in out_HHI.items() if k not in ['aux_outputs', 'HHI_aux_outputs']}

        # ── DETR matching (shared) ────────────────────────────────────────────
        # Use the HOI branch's pred_logits/pred_boxes (identical to HHI branch)
        detr_indices       = self.matcher(out_hoi_without_aux, targets)
        detr_human_indices = self.human_matcher(out_HHI_without_aux, targets)

        # ── HOI matching (Branch 1) ───────────────────────────────────────────
        hoi_indices    = None
        hoi_indices_V  = None
        hoi_targets    = None

        if self.HOI_losses is not None:
            hoi_indices, hoi_indices_V, hoi_targets = self.hoi_matcher(
                out_hoi_without_aux, targets, detr_indices
            )

        # ── HHI matching (Branch 2) ───────────────────────────────────────────
        HHI_indices  = None
        HHI_targets  = None

        if self.HHI_losses and self.HHI_matcher is not None:
            HHI_indices, HHI_targets = self.HHI_matcher(out_HHI_without_aux, targets, detr_human_indices)

        # ── Normalisation ─────────────────────────────────────────────────────
        num_boxes_detr = sum(len(t["labels"]) for t in targets)
        num_boxes_detr = torch.as_tensor(
            [num_boxes_detr], dtype=torch.float,
            device=outputs["pred_logits"].device,
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes_detr)
        num_boxes_detr = torch.clamp(num_boxes_detr / get_world_size(), min=1).item()


        num_boxes_HHI = max(sum(len(t.get("human_boxes", [])) for t in targets), 1)
        num_boxes_HHI = torch.as_tensor(
            [num_boxes_HHI], dtype=torch.float, device=outputs["pred_logits"].device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes_HHI)
        num_boxes_HHI = torch.clamp(num_boxes_HHI / get_world_size(), min=1).item()

        # =====================================================================
        # Compute all losses
        # =====================================================================
        losses = {}

        # ── Shared DETR losses ────────────────────────────────────────────────
        for loss in self.losses:
            losses.update(self.get_loss(loss, out_hoi, targets, detr_indices, num_boxes_detr))

        # DETR auxiliary losses
        if "aux_outputs" in outputs:
            for i, aux in enumerate(outputs["aux_outputs"]):

                aux_idx = self.matcher(aux, targets)
                for loss in self.losses:

                    l_dict = self.get_loss(loss, aux, targets, aux_idx, num_boxes_detr)
                    losses.update({f"{k}_{i}": v for k, v in l_dict.items()})
        
        

        # ── Branch 1 HOI losses ───────────────────────────────────────────────
        hoi_loss_dict = {}
        if self.HOI_losses and hoi_indices is not None:
            for loss in self.HOI_losses:
                if loss in ("pair_labels", "pair_actions"):
                    l_dict = self.get_hoi_loss(loss, out_hoi, hoi_targets, hoi_indices)
                elif loss in ("pair_violence", "pair_violence_labels"):
                    # Violence branch uses its own Hungarian assignment
                    l_dict = self.get_hoi_loss(loss, out_hoi, hoi_targets, hoi_indices_V)
                else:
                    print(f"[DualBranchCriterion] Unknown HOI loss: {loss}")
                    sys.exit(1)
                hoi_loss_dict.update(l_dict)
            losses.update(hoi_loss_dict)

            # HOI auxiliary losses
            if "hoi_aux_outputs" in out_hoi:
                for i, aux in enumerate(out_hoi["hoi_aux_outputs"]):
                    input_targets = [copy.deepcopy(t) for t in targets]
                    aux_hoi_idx, aux_hoi_idx_V,_= self.hoi_matcher(aux, input_targets, detr_indices)

                    for loss in self.HOI_losses:
                        if loss in ("pair_labels", "pair_actions"):
                            l_dict = self.get_hoi_loss(loss, aux, hoi_targets, aux_hoi_idx)
                        elif loss in ("pair_violence", "pair_violence_labels"):
                            # Violence branch uses its own Hungarian assignment
                            l_dict = self.get_hoi_loss(loss, aux, hoi_targets, aux_hoi_idx_V)
                        else:
                            print(f"[DualBranchCriterion] Unknown HOI loss: {loss}")
                            sys.exit(1)
                        losses.update({f"{k}_aux_{i}": v for k, v in l_dict.items()})


            
            
            # ── Branch 1 ANCHOR losses (un-fused predictions, dual supervision)

            hoi_anchor_remapped = self._remap_anchor_keys(
                out_hoi,
                key_pairs=[
                    ("pred_hidx",             "pred_hidx_anchor"),
                    ("pred_oidx",             "pred_oidx_anchor"),
                    ("pred_actions",          "pred_actions_anchor"),
                    ("pred_violence_actions", "pred_violence_actions_anchor"),
                ],
            )
            hoi_anchor_loss_dict = {}
            for loss in self.HOI_losses:
                if loss in ("pair_labels", "pair_actions"):
                    l_dict = self.get_hoi_loss(loss, hoi_anchor_remapped, hoi_targets, hoi_indices)
                elif loss in ("pair_violence", "pair_violence_labels"):
                    l_dict = self.get_hoi_loss(loss, hoi_anchor_remapped, hoi_targets, hoi_indices_V)
                else:
                    continue
                hoi_anchor_loss_dict.update(l_dict)
            # Scale and log under "_anchor" suffix
            hoi_anchor_loss_dict = {
                f"{k}_anchor": self.anchor_loss_weight * v for k, v in hoi_anchor_loss_dict.items()
            }
            losses.update(hoi_anchor_loss_dict)


        

        # ── Branch 2 HHI losses ───────────────────────────────────────────────
        HHI_loss_dict = {}
        if self.HHI_losses and HHI_indices is not None:
            # Human-box regression (uses DETR matcher restricted to humans)
            losses.update(self.loss_human_boxes(out_HHI, targets, detr_human_indices, num_boxes_HHI))

            for loss in self.HHI_losses:
                l_dict = self.get_HHI_loss(loss, out_HHI, HHI_targets, HHI_indices)
                HHI_loss_dict.update(l_dict)
            losses.update(HHI_loss_dict)

            # HHI auxiliary losses
            if "HHI_aux_outputs" in out_HHI:
                for i, aux in enumerate(out_HHI["HHI_aux_outputs"]):
                    input_targets = [copy.deepcopy(target) for target in targets]
                    aux_human_idx = self.human_matcher(aux, input_targets)
                    aux_HHI_idx, _ = self.HHI_matcher(aux, input_targets, aux_human_idx)


                    for loss in self.HHI_losses:
                        l_dict = self.get_HHI_loss(loss, aux, HHI_targets, aux_HHI_idx)
                        l_dict.update(self.loss_human_boxes(aux, targets, detr_human_indices, num_boxes_HHI))
                        l_dict = {k + f'_aux_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)


            # ── Branch 2 ANCHOR losses (un-fused predictions, dual supervision)
            HHI_anchor_remapped = self._remap_anchor_keys(
                out_HHI,
                key_pairs=[
                    ("pred_aggressor_idx",  "pred_aggressor_idx_anchor"),
                    ("pred_victim_idx",     "pred_victim_idx_anchor"),
                    ("pred_action_logits",  "pred_action_logits_anchor"),
                    ("pred_victim_visible", "pred_victim_visible_anchor"),
                ],
            )
            HHI_anchor_loss_dict = {}
            for loss in self.HHI_losses:
                l_dict = self.get_HHI_loss(loss, HHI_anchor_remapped, HHI_targets, HHI_indices)
                HHI_anchor_loss_dict.update(l_dict)
            HHI_anchor_loss_dict = {
                f"{k}_anchor": self.anchor_loss_weight * v for k, v in HHI_anchor_loss_dict.items()
            }
            losses.update(HHI_anchor_loss_dict)
        
        

        # =====================================================================
        # Sub-totals and grand total (for logging + single backward pass)
        # =====================================================================

        violence_keys = {"loss_violence", "loss_hidx_v", "loss_oidx_v"}
 
        all_hoi_for_total = {**hoi_loss_dict, **hoi_anchor_loss_dict} if self.HOI_losses and hoi_indices is not None else {}
        loss_hoi_total = sum(
            v for k, v in all_hoi_for_total.items()
            if not any(vk in k for vk in violence_keys)
        )
        loss_violence_total = sum(
            v for k, v in all_hoi_for_total.items()
            if any(vk in k for vk in violence_keys)
        )
        losses["loss_hoi_total"]      = loss_hoi_total
        losses["loss_violence_total"] = loss_violence_total
 
        # Branch 2: fused + anchor sub-totals combined into loss_HHI_total
        all_HHI_for_total = {**HHI_loss_dict, **HHI_anchor_loss_dict} if self.HHI_losses and HHI_indices is not None else {}
        losses["loss_HHI_total"] = sum(all_HHI_for_total.values()) if all_HHI_for_total else torch.tensor(0.0)

        return losses
