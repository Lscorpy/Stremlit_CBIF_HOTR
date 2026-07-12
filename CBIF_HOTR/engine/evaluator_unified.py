# =============================================================================
# evaluator_unified.py
#
# Single evaluation engine for DualBranchModel.
#
# Produces three mAP numbers per validation run:
#   1. HOI V-COCO      (Branch 1, pair_score_vcoco)
#   2. HOI Violence    (Branch 1, pair_score_violence)
#   3. HHI             (Branch 2, pair_score_HHI)
#
# Requires unified_postprocess.py (UnifiedPostProcess) as the postprocessor.
# =============================================================================

import time
import datetime

import numpy as np
import torch

import CBIF_HOTR.util.misc as utils
import CBIF_HOTR.util.logger as loggers
from CBIF_HOTR.util.box_ops import rescale_bboxes, rescale_pairs
from CBIF_HOTR.metrics.utils import compute_overlap
# Branch-1 metric (shared for vcoco + violence columns)
from CBIF_HOTR.metrics.ap_role import APRole
# Branch-2 metric (HHI aggressor/victim pairs)
from CBIF_HOTR.metrics.ap_role_HHI import HHI_APRole   # fixed version
from CBIF_HOTR.metrics.detection_recall import UnaryDetectionRecallHOI,UnaryDetectionRecallHHI   # fixed version
from CBIF_HOTR.metrics.iou_quality import IoUQualityHOI,IoUQualityHHI

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# Shared IoU threshold for the lightweight matching used by the
# IoU-quality helpers below — matches APRole/APHHIRole's default.
iou_eval_threshold = 0.5

ADD_VAL_PRINT=False
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_b1_evaluators(action_names):
    """Return (scenario1, scenario2) APRole instances for a Branch-1 action list."""
    e1 = APRole(act_name=action_names, scenario_flag=True,  iou_threshold=0.5)
    e2 = APRole(act_name=action_names, scenario_flag=False, iou_threshold=0.5)
    return e1, e2


def _make_b2_evaluators(action_names):
    """Return (scenario1, scenario2) APHHIRole instances for Branch-2 HHI actions."""
    e1 = HHI_APRole(human_actions=action_names, scenario_flag=True,  iou_threshold=0.5)
    e2 = HHI_APRole(human_actions=action_names, scenario_flag=False, iou_threshold=0.5)
    return e1, e2


# ---------------------------------------------------------------------------
# Branch-1 update helper  (vcoco OR violence — same APRole class, same format)
# ---------------------------------------------------------------------------

def _update_b1(role_eval1, role_eval2,
               hbox, obox, score_grid,
               target, action_idx, num_human_act,
               gt_action_key='pair_actions'):
    """
        Feed one image into a pair of APRole evaluators.

    score_grid : (n_act_full, n_h, n_o+1)  — FULL action dim from postprocessor
    action_idx : list[int]                  — which action columns to keep
    gt_action_key : str — which target field holds the per-pair action GT
                    ('pair_actions' for vcoco, 'pair_violence' for violence)
    """
    score = score_grid[action_idx, :, :]                 # (n_act, n_h, n_o+1)

    gt_h_inds = (target['labels'] == 1)
    gt_h_box  = target['boxes'][gt_h_inds, :4].cpu().numpy()
    gt_h_act  = target['inst_actions'][gt_h_inds, :num_human_act].cpu().numpy()

    # Guard: pad to at least 1 column so APRole indexing doesn't crash on shape (N,0)
    if gt_h_act.shape[1] == 0:
        gt_h_act = np.full((gt_h_act.shape[0], 1), -1, dtype=np.float32)

    gt_p_box = target['pair_boxes'].cpu().numpy()
    gt_p_act = target[gt_action_key].cpu().numpy()[:, action_idx]   # <-- fixed

    role_eval1.add_data(hbox, obox, score, gt_h_box, gt_h_act, gt_p_box, gt_p_act)
    role_eval2.add_data(hbox, obox, score, gt_h_box, gt_h_act, gt_p_box, gt_p_act)
# ---------------------------------------------------------------------------
# Branch-2 update helper
# ---------------------------------------------------------------------------

def _unpack_HHI_grid(pred, target):
    """
    Convert UnifiedPostProcess Branch-2 grid output into the flat
    (pred_agg_box, pred_vic_box, action_score, pred_no_victim) arrays
    that APHHIRole.add_data() expects.

    UnifiedPostProcess outputs
    --------------------------
    pred['pair_score_HHI']     : (n_act_HHI, n_h, n_o+1)  torch tensor
    pred['HHI_victim_visible'] : (n_h, n_o+1)             bool tensor
    pred['h_box']              : (n_h, 4)
    pred['o_box']              : (n_o+1, 4)   last row = null/zero slot
 
    APHHIRole.add_data() expects
    ----------------------------
    pred_agg_box  : (n_p, 4)              — one row per (aggressor, victim) pair
    pred_vic_box  : (n_p, 4)             — matched victim box (zeros if invisible)
    action_score  : (num_actions, n_p)
    pred_no_victim: (n_p,)  bool/int
    """
    score_grid  = pred['pair_score_HHI']       # (A, n_h, n_o+1)
    vis_grid    = pred['HHI_victim_visible']   # (n_h, n_o+1)  bool
    h_box       = pred['h_box']                # (n_h, 4)
    o_box       = pred['o_box']                # (n_o+1, 4)

    n_act, n_h, n_o1 = score_grid.shape       # n_o1 = n_o + 1

    if n_h == 0:
        empty4 = np.zeros((0, 4), dtype=np.float32)
        empty_act = np.zeros((n_act, 0), dtype=np.float32)
        return empty4, empty4, empty_act, np.zeros(0, dtype=np.int32)

    # Flatten the (n_h, n_o+1) grid into individual candidate pairs
    # We enumerate every (aggressor_i, victim_j) cell.
    agg_boxes   = []
    vic_boxes   = []
    act_scores  = []   # list of (n_act,) vectors
    no_victim   = []

    h_box_np  = h_box.cpu().numpy()   if torch.is_tensor(h_box)  else h_box
    o_box_np  = o_box.cpu().numpy()   if torch.is_tensor(o_box)  else o_box
    sg_np     = score_grid.cpu().numpy() if torch.is_tensor(score_grid) else score_grid
    vis_np    = vis_grid.cpu().numpy()   if torch.is_tensor(vis_grid)   else vis_grid

    null_col  = n_o1 - 1   # last column = "no visible victim" slot

    for hi in range(n_h):
        for oi in range(n_o1):
            cell_score = sg_np[:, hi, oi]          # (n_act,)
            if cell_score.max() == 0.0:
                continue                            # empty cell, skip

            agg_boxes.append(h_box_np[hi])          # aggressor box

            is_null = (oi == null_col)
            if is_null:
                vic_boxes.append(np.zeros(4, dtype=np.float32))
                no_victim.append(1)
            else:
                vic_boxes.append(o_box_np[oi])
                no_victim.append(0 if vis_np[hi, oi] else 1)

            act_scores.append(cell_score)

    if len(agg_boxes) == 0:
        empty4 = np.zeros((0, 4), dtype=np.float32)
        return empty4, empty4, np.zeros((n_act, 0), dtype=np.float32), np.zeros(0, dtype=np.int32)

    pred_agg_box  = np.stack(agg_boxes,  axis=0)          # (n_p, 4)
    pred_vic_box  = np.stack(vic_boxes,  axis=0)          # (n_p, 4)
    action_score  = np.stack(act_scores, axis=0).T        # (n_act, n_p)
    pred_no_victim = np.array(no_victim, dtype=np.int32)  # (n_p,)

    return pred_agg_box, pred_vic_box, action_score, pred_no_victim

# ---------------------------------------------------------------------------
# Branch-1 (HOI) IoU
# ---------------------------------------------------------------------------
def _update_iou_quality_b1(iou_eval, hbox, obox, target):

    if 'pair_boxes' not in target or target['pair_boxes'].shape[0] == 0:
        return
    if hbox.shape[0] == 0:
        return
 
    pair_boxes = target['pair_boxes']
    if torch.is_tensor(pair_boxes):
        pair_boxes = pair_boxes.cpu().numpy()
 
    gt_h_box = pair_boxes[:, :4]
    gt_o_box = pair_boxes[:, 4:8]
    has_object = (pair_boxes[:, 4] != -1)
 
    # Best-matching predicted human box per GT human box.
    h_overlaps = compute_overlap(gt_h_box, hbox)          # (n_g, n_h)
    best_h_idx = np.argmax(h_overlaps, axis=1)
    best_h_iou = h_overlaps[np.arange(len(gt_h_box)), best_h_idx]
    h_matched = best_h_iou > iou_eval_threshold
    pred_h_matched = hbox[best_h_idx][h_matched]
    gt_h_matched   = gt_h_box[h_matched]
 
    pred_o_matched = np.zeros((0, 4))
    gt_o_matched   = np.zeros((0, 4))
    has_object_matched = np.zeros((0,), dtype=bool)
    if obox.shape[0] > 0:
        o_overlaps = compute_overlap(gt_o_box, obox)
        best_o_idx = np.argmax(o_overlaps, axis=1)
        best_o_iou = o_overlaps[np.arange(len(gt_o_box)), best_o_idx]
        o_matched = (best_o_iou > iou_eval_threshold) & h_matched
        pred_o_matched = obox[best_o_idx][o_matched]
        gt_o_matched   = gt_o_box[o_matched]
        has_object_matched = has_object[o_matched]
 
    iou_eval.add_data(pred_h_matched, gt_h_matched,
                       pred_o_matched, gt_o_matched, has_object_matched)
 
# ---------------------------------------------------------------------------
# Branch-2 (HHI) IoU
# ---------------------------------------------------------------------------
def _update_iou_quality_b2(iou_eval, hbox, target):

    agg_idx = target['aggressor_index']
    vic_idx = target['victim_index']
    if torch.is_tensor(agg_idx):
        agg_idx = agg_idx.cpu().numpy()
    if torch.is_tensor(vic_idx):
        vic_idx = vic_idx.cpu().numpy()
 
    if len(agg_idx) == 0 or hbox.shape[0] == 0:
        return
 
    human_boxes = target['human_boxes']
    if torch.is_tensor(human_boxes):
        human_boxes = human_boxes.cpu().numpy()
 
    gt_agg_box = human_boxes[agg_idx]
    vic_visible = (vic_idx >= 0)
    gt_vic_box = np.zeros((len(vic_idx), 4), dtype=np.float32)
    if vic_visible.any():
        gt_vic_box[vic_visible] = human_boxes[vic_idx[vic_visible]]
 
    agg_overlaps = compute_overlap(gt_agg_box, hbox)
    best_agg_idx = np.argmax(agg_overlaps, axis=1)
    best_agg_iou = agg_overlaps[np.arange(len(gt_agg_box)), best_agg_idx]
    agg_matched = best_agg_iou > iou_eval_threshold
    pred_agg_matched = hbox[best_agg_idx][agg_matched]
    gt_agg_matched   = gt_agg_box[agg_matched]
 
    pred_vic_matched = np.zeros((0, 4))
    gt_vic_matched    = np.zeros((0, 4))
    vic_visible_matched = np.zeros((0,), dtype=bool)
    if vic_visible.any():
        vic_overlaps = compute_overlap(gt_vic_box, hbox)
        best_vic_idx = np.argmax(vic_overlaps, axis=1)
        best_vic_iou = vic_overlaps[np.arange(len(gt_vic_box)), best_vic_idx]
        vic_matched = (best_vic_iou > iou_eval_threshold) & agg_matched & vic_visible
        pred_vic_matched = hbox[best_vic_idx][vic_matched]
        gt_vic_matched    = gt_vic_box[vic_matched]
        vic_visible_matched = np.ones(vic_matched.sum(), dtype=bool)
 
    iou_eval.add_data(pred_agg_matched, gt_agg_matched,
                       pred_vic_matched, gt_vic_matched, vic_visible_matched)
    

# ---------------------------------------------------------------------------
# Branch-1 (HOI) detection recall (pairing)
# ---------------------------------------------------------------------------
def _update_detection_recall_b1(det_eval, det_boxes_full, target):

    if 'pair_boxes' not in target or target['pair_boxes'].shape[0] == 0:
        return  # no HOI pairs annotated for this image — nothing to check
 
    pair_boxes = target['pair_boxes']
    if torch.is_tensor(pair_boxes):
        pair_boxes = pair_boxes.cpu().numpy()
 
    gt_h_box = pair_boxes[:, :4]
    gt_o_box = pair_boxes[:, 4:8]
    has_object = (pair_boxes[:, 4] != -1)
 
    det_eval.add_data(
        det_boxes_full, gt_h_box, gt_o_box, has_object,
        image_name=target.get('imgae_name_for_track', None),
    )

# ---------------------------------------------------------------------------
#  Branch-2 (HHI) detection recall (pairing)
# ---------------------------------------------------------------------------
def _update_detection_recall_b2(role_eval1, role_eval2, pred, target, action_idx,det_eval, det_boxes_full):

    agg_idx = target['aggressor_index']
    vic_idx = target['victim_index']
    if torch.is_tensor(agg_idx):
        agg_idx = agg_idx.cpu().numpy()
    if torch.is_tensor(vic_idx):
        vic_idx = vic_idx.cpu().numpy()
 
    if len(agg_idx) == 0:
        return  # no HHI pairs annotated for this image
 
    human_boxes = target['human_boxes']
    if torch.is_tensor(human_boxes):
        human_boxes = human_boxes.cpu().numpy()
 
    gt_agg_box = human_boxes[agg_idx]                     # (n_g, 4)
    vic_visible = (vic_idx >= 0)                            # (n_g,) bool
 
    gt_vic_box = np.zeros((len(vic_idx), 4), dtype=np.float32)
    if vic_visible.any():
        gt_vic_box[vic_visible] = human_boxes[vic_idx[vic_visible]]
 
    det_eval.add_data(
        det_boxes_full, gt_agg_box, gt_vic_box, vic_visible,
        image_name=target.get('imgae_name_for_track', None),
    )
 
 
 
    """Feed one image into a pair of APHHIRole evaluators."""
    pred_agg_box, pred_vic_box, action_score, pred_no_victim = \
        _unpack_HHI_grid(pred, target)
 
    # Slice to selected action indices
    if len(action_idx) > 0 and action_score.shape[0] > 0:
        action_score = action_score[action_idx, :]
 
    # Build GT arrays from target
    agg_idx = target['aggressor_index']   # list or 1-D tensor
    vic_idx = target['victim_index']

 
    if torch.is_tensor(agg_idx):
        agg_idx = agg_idx.cpu().numpy()
    if torch.is_tensor(vic_idx):
        vic_idx = vic_idx.cpu().numpy()
 
    human_boxes   = target['human_boxes'].cpu().numpy()   # (n_humans, 4)
    viol_actions  = target['violence_actions']             # may be (n_pairs,) or (n_pairs, n_act)
    has_visible   = target['has_target_visible']
 
    if torch.is_tensor(viol_actions):
        viol_actions = viol_actions.cpu().numpy()
    if torch.is_tensor(has_visible):
        has_visible = has_visible.cpu().numpy()
 
    n_humans = human_boxes.shape[0]
    if len(agg_idx) > 0:
        assert np.all(agg_idx >= 0) and np.all(agg_idx < n_humans), \
            f"aggressor_index out of range: {agg_idx} for {n_humans} human boxes " \
            f"(image: {target.get('imgae_name_for_track', '?')})"

        bad_vic = (vic_idx < -1) | (vic_idx >= n_humans)
        assert not bad_vic.any(), \
            f"victim_index has invalid values {vic_idx[bad_vic]} " \
            f"(must be -1 or in [0, {n_humans})) " \
            f"(image: {target.get('imgae_name_for_track', '?')})"
 
    if len(agg_idx) == 0:
        gt_agg_box = np.zeros((0, 4), dtype=np.float32)
        gt_vic_box = np.zeros((0, 4), dtype=np.float32)
        n_act_sel  = len(action_idx) if action_idx else action_score.shape[0]
        gt_act     = np.zeros((0, n_act_sel), dtype=np.float32)
        gt_visible = np.zeros((0,),           dtype=np.float32)

    else:
        gt_agg_box = human_boxes[agg_idx]   # (n_g, 4)
 
        # ── CRITICAL FIX ────────────────────────────────────────────────
        valid_vic  = (vic_idx >= 0)
        if valid_vic.any():
            gt_vic_box[valid_vic] = human_boxes[vic_idx[valid_vic]]

 
        # violence_actions can be (n_pairs,) 1-D when there is 1 action class
        if viol_actions.ndim == 1:
            viol_actions = viol_actions[:, None]
 
        gt_act     = viol_actions[:, action_idx] if len(action_idx) > 0 else viol_actions
        gt_visible = has_visible
 

        if gt_visible.ndim == 1:
            mismatch = (~valid_vic) & (gt_visible > 0.5)
        else:
            mismatch = (~valid_vic) & (gt_visible[:, 0] > 0.5)
        if mismatch.any():
            print(f"[evaluator_unified][_update_b2] WARNING: "
                  f"{int(mismatch.sum())} pair(s) have victim_index==-1 but "
                  f"has_target_visible>0.5 — forcing visible=0 for consistency.")
            gt_visible = gt_visible.copy()
            if gt_visible.ndim == 1:
                gt_visible[mismatch] = 0.0
            else:
                gt_visible[mismatch, 0] = 0.0



    role_eval1.add_data(pred_agg_box, pred_vic_box, action_score, pred_no_victim,
                        gt_agg_box, gt_vic_box, gt_act, gt_visible)
    role_eval2.add_data(pred_agg_box, pred_vic_box, action_score, pred_no_victim,
                        gt_agg_box, gt_vic_box, gt_act, gt_visible)
    

def _update_b2(role_eval1, role_eval2, pred, target, action_idx):
    """Feed one image into a pair of APHHIRole evaluators."""
    pred_agg_box, pred_vic_box, action_score, pred_no_victim = \
        _unpack_HHI_grid(pred, target)

    # Slice to selected action indices
    if len(action_idx) > 0 and action_score.shape[0] > 0:
        action_score = action_score[action_idx, :]

    # Build GT arrays from target
    agg_idx = target['aggressor_index']   # list or 1-D tensor
    vic_idx = target['victim_index']


    if torch.is_tensor(agg_idx):
        agg_idx = agg_idx.cpu().numpy()
    if torch.is_tensor(vic_idx):
        vic_idx = vic_idx.cpu().numpy()

    human_boxes   = target['human_boxes'].cpu().numpy()   # (n_humans, 4)
    viol_actions  = target['violence_actions']             # may be (n_pairs,) or (n_pairs, n_act)
    has_visible   = target['has_target_visible']

    if torch.is_tensor(viol_actions):
        viol_actions = viol_actions.cpu().numpy()
    if torch.is_tensor(has_visible):
        has_visible = has_visible.cpu().numpy()

    n_humans = human_boxes.shape[0]
    if len(agg_idx) > 0:
        assert np.all(agg_idx >= 0) and np.all(agg_idx < n_humans), \
            f"aggressor_index out of range: {agg_idx} for {n_humans} human boxes " \
            f"(image: {target.get('imgae_name_for_track', '?')})"

        bad_vic = (vic_idx < -1) | (vic_idx >= n_humans)
        assert not bad_vic.any(), \
            f"victim_index has invalid values {vic_idx[bad_vic]} " \
            f"(must be -1 or in [0, {n_humans})) " \
            f"(image: {target.get('imgae_name_for_track', '?')})"

    if len(agg_idx) == 0:
        gt_agg_box = np.zeros((0, 4), dtype=np.float32)
        gt_vic_box = np.zeros((0, 4), dtype=np.float32)
        n_act_sel  = len(action_idx) if action_idx else action_score.shape[0]
        gt_act     = np.zeros((0, n_act_sel), dtype=np.float32)
        gt_visible = np.zeros((0,),           dtype=np.float32)

    else:
        gt_agg_box = human_boxes[agg_idx]   # (n_g, 4)

        # ── CRITICAL FIX ────────────────────────────────────────────────

        gt_vic_box = np.zeros((len(vic_idx), 4), dtype=np.float32)
        valid_vic  = (vic_idx >= 0)
        if valid_vic.any():
            gt_vic_box[valid_vic] = human_boxes[vic_idx[valid_vic]]

        if viol_actions.ndim == 1:
            viol_actions = viol_actions[:, None]

        gt_act     = viol_actions[:, action_idx] if len(action_idx) > 0 else viol_actions
        gt_visible = has_visible


        if gt_visible.ndim == 1:
            mismatch = (~valid_vic) & (gt_visible > 0.5)
        else:
            mismatch = (~valid_vic) & (gt_visible[:, 0] > 0.5)
        if mismatch.any():
            print(f"[evaluator_unified][_update_b2] WARNING: "
                  f"{int(mismatch.sum())} pair(s) have victim_index==-1 but "
                  f"has_target_visible>0.5 — forcing visible=0 for consistency.")
            gt_visible = gt_visible.copy()
            if gt_visible.ndim == 1:
                gt_visible[mismatch] = 0.0
            else:
                gt_visible[mismatch, 0] = 0.0



    role_eval1.add_data(pred_agg_box, pred_vic_box, action_score, pred_no_victim,
                        gt_agg_box, gt_vic_box, gt_act, gt_visible)
    role_eval2.add_data(pred_agg_box, pred_vic_box, action_score, pred_no_victim,
                        gt_agg_box, gt_vic_box, gt_act, gt_visible)


# ---------------------------------------------------------------------------
# Process targets (rescale boxes to original image size)
# ---------------------------------------------------------------------------

def _process_targets(targets, orig_target_sizes):

    for idx, (tgt, sz) in enumerate(zip(targets, orig_target_sizes)):
        targets[idx]['boxes']       = rescale_bboxes(tgt['boxes'],       sz)
        if 'human_boxes' in tgt and tgt['human_boxes'].shape[0] > 0:
            targets[idx]['human_boxes'] = rescale_bboxes(tgt['human_boxes'], sz)
        # pair_boxes may be absent (HHI dataset) or empty (no HOI pairs this image)
        if 'pair_boxes' in tgt and tgt['pair_boxes'].shape[0] > 0:
            targets[idx]['pair_boxes'] = rescale_pairs(tgt['pair_boxes'], sz)
    return targets


# ---------------------------------------------------------------------------
# Main evaluation engine
# ---------------------------------------------------------------------------

@torch.no_grad()
def unified_evaluate(model, criterion, postprocessors, data_loader, device, args,
                     print_results=True, thr=0.3, wandb_log=False):
    """
    Single validation loop for DualBranchModel.

    Returns
    -------
    dict with keys:
        'vcoco_s1', 'vcoco_s2'       – V-COCO HOI mAP (scenario 1 & 2)
        'violence_s1', 'violence_s2' – Violence HOI mAP
        'HHI_s1', 'HHI_s2'           – HHI mAP
    """
    model.eval()
    criterion.eval()

    # ── Evaluator instances ────────────────────────────────────────────────
    vcoco_eval1,    vcoco_eval2    = _make_b1_evaluators(args.object_actions_vcoco)
    violence_eval1, violence_eval2 = _make_b1_evaluators(args.object_actions_violence)
    HHI_eval1,      HHI_eval2      = _make_b2_evaluators(args.human_actions_HHI)


    det_recall_hoi = UnaryDetectionRecallHOI(iou_threshold=0.5)
    det_recall_HHI = UnaryDetectionRecallHHI(iou_threshold=0.5)

    iou_quality_hoi = IoUQualityHOI()
    iou_quality_HHI = IoUQualityHHI()

    # Action index filters (subset of full action dim)
    vcoco_idx    = list(args.valid_ids_vcoco)
    violence_idx = list(args.valid_ids_violence)
    HHI_idx      = list(getattr(args, 'valid_ids_HHI', range(len(args.human_actions_HHI))))

    metric_logger    = loggers.MetricLogger(mode="test", delimiter="  ")
    header           = 'Unified Evaluation (HOI-vcoco | HOI-violence | HHI)'
    hoi_times        = []

    for samples, targets in metric_logger.log_every(data_loader, 1, header):
        samples = samples.to(device)
        targets = [{k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in t.items()} for t in targets]

        outputs   = model(samples)
        loss_dict = criterion(outputs, targets)
        utils.reduce_dict(loss_dict)   # keep DDP in sync

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessors(outputs, orig_target_sizes)



        targets = _process_targets(targets, orig_target_sizes)
        hoi_times.append(results[0].get('hoi_recognition_time', 0.0) * 1000)

        # ── Per-image accumulation ────────────────────────────────────────
        for pred, tgt in zip(results, targets):
            # numpy arrays shared by both B1 evaluators
            hbox = pred['h_box'].cpu().numpy()
            obox = pred['o_box'].cpu().numpy()

            det_boxes_full = pred['boxes'].cpu().numpy()
 
            _update_detection_recall_b1(det_recall_hoi, det_boxes_full, tgt)
            _update_detection_recall_b2(HHI_eval1, HHI_eval2, pred, tgt, HHI_idx,det_recall_HHI, det_boxes_full)

            _update_iou_quality_b1(iou_quality_hoi, hbox, obox, tgt)
            _update_iou_quality_b2(iou_quality_HHI, hbox, tgt)

            # Branch 1 — V-COCO
            # Guard: skip if this target has no HOI pair annotations (e.g. pure HHI image)
            has_hoi_pairs = ('pair_boxes' in tgt and tgt['pair_boxes'].shape[0] > 0)
            if 'pair_score_vcoco' in pred and has_hoi_pairs:
                score_vcoco = pred['pair_score_vcoco'].cpu().numpy()
                _update_b1(vcoco_eval1, vcoco_eval2,
                           hbox, obox, score_vcoco,
                           tgt, vcoco_idx, args.num_human_act_vcoco,
                           gt_action_key='pair_actions')

            # Branch 1 — Violence HOI
            if 'pair_score_violence' in pred and has_hoi_pairs:
                score_V = pred['pair_score_violence'].cpu().numpy()
                _update_b1(violence_eval1, violence_eval2,
                           hbox, obox, score_V,
                           tgt, violence_idx, args.num_human_act_violence,
                           gt_action_key='pair_violence')

            # Branch 2 — HHI
            if 'pair_score_HHI' in pred:
                _update_b2(HHI_eval1, HHI_eval2, pred, tgt, HHI_idx)

    avg_time = sum(hoi_times) / len(hoi_times) if hoi_times else 0.0
    print(f"[stats] HOI Recognition Time (avg) : {avg_time:.4f} ms")

    # ── Compute mAP ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BRANCH 1 — V-COCO HOI")
    vcoco_s1 = vcoco_eval1.evaluate(print_log=print_results)
    vcoco_s2 = vcoco_eval2.evaluate(print_log=print_results)

    print("\nBRANCH 1 — VIOLENCE HOI")
    violence_s1 = violence_eval1.evaluate(print_log=print_results)
    violence_s2 = violence_eval2.evaluate(print_log=print_results)

    print("\nBRANCH 2 — HHI")
    HHI_s1 = HHI_eval1.evaluate(print_log=print_results)
    HHI_s2 = HHI_eval2.evaluate(print_log=print_results)

    # ── Pure detection-layer diagnostic ────────────────────────────────────

    det_hoi_stats = det_recall_hoi.evaluate(print_log=ADD_VAL_PRINT)
    det_HHI_stats = det_recall_HHI.evaluate(print_log=ADD_VAL_PRINT)
 
    # Localization-quality diagnostic (continuous IoU, matched pairs only).
    iou_hoi_stats = iou_quality_hoi.evaluate(print_log=ADD_VAL_PRINT)
    iou_HHI_stats = iou_quality_HHI.evaluate(print_log=ADD_VAL_PRINT)
 
    if ADD_VAL_PRINT:
        print("\n" + "=" * 60)
        print("  UNIFIED EVALUATION SUMMARY")
        print(f"  V-COCO   HOI  |  S1: {vcoco_s1:6.2f}   S2: {vcoco_s2:6.2f}")
        print(f"  Violence HOI  |  S1: {violence_s1:6.2f}   S2: {violence_s2:6.2f}")
        print(f"  HHI           |  S1: {HHI_s1:6.2f}   S2: {HHI_s2:6.2f}")
        print(f"  Detection pair recall (HOI)            : {det_hoi_stats['pair_recall']:6.2f}%")
        print(f"  Detection pair recall (HHI, visible-only): {det_HHI_stats['pair_recall_visible_subset']:6.2f}%")
        print(f"  Mean IoU  human/object (HOI)  : {iou_hoi_stats['human']['mean']:.3f} / {iou_hoi_stats['object']['mean']:.3f}")
        print(f"  Mean IoU  aggressor/victim (HHI): {iou_HHI_stats['aggressor']['mean']:.3f} / {iou_HHI_stats['victim']['mean']:.3f}")
        print("=" * 60)
 
    if wandb_log and _WANDB_AVAILABLE and utils.is_main_process():
        import wandb as _wandb
        _wandb.log({
            'vcoco_s1':    vcoco_s1,    'vcoco_s2':    vcoco_s2,
            'violence_s1': violence_s1, 'violence_s2': violence_s2,
            'HHI_s1':      HHI_s1,      'HHI_s2':      HHI_s2,
            'det_recall_hoi_pair':           det_hoi_stats['pair_recall'],
            'det_recall_hoi_human':          det_hoi_stats['human_recall'],
            'det_recall_hoi_object':         det_hoi_stats['object_recall'],
            'det_recall_HHI_pair_visible':   det_HHI_stats['pair_recall_visible_subset'],
            'det_recall_HHI_aggressor':      det_HHI_stats['aggressor_recall'],
            'det_recall_HHI_victim_visible': det_HHI_stats['victim_recall_visible_subset'],
            'iou_hoi_human_mean':  iou_hoi_stats['human']['mean'],
            'iou_hoi_object_mean': iou_hoi_stats['object']['mean'],
            'iou_HHI_aggressor_mean': iou_HHI_stats['aggressor']['mean'],
            'iou_HHI_victim_mean':    iou_HHI_stats['victim']['mean'],
        })
    recall_hoi_pair=det_hoi_stats['pair_recall']
    recall_HHI_pair_visible=det_HHI_stats['pair_recall_visible_subset']

    iou_hoi_human_mean=iou_hoi_stats['human']['mean']
    iou_hoi_object_mean=iou_hoi_stats['object']['mean']
    iou_HHI_aggressor_mean= iou_HHI_stats['aggressor']['mean']
    iou_HHI_victim_mean= iou_HHI_stats['victim']['mean']

    return  (vcoco_s1,vcoco_s2, 
             violence_s1,violence_s2, 
             HHI_s1,HHI_s2,
             recall_hoi_pair,recall_HHI_pair_visible,
             iou_hoi_human_mean, iou_hoi_object_mean,
             iou_HHI_aggressor_mean, iou_HHI_victim_mean)


 