import numpy as np
from CBIF_HOTR.metrics.utils import _compute_ap, compute_overlap

PRINT_DEBUG = False


class HHI_APRole(object):
    """
    AP evaluator for the HHI (aggressor/victim) branch.

    Shapes (per add_data call)
    ---------------------------
    pred_agg_box   : (n_p, 4)
    pred_vic_box   : (n_p, 4)            zeros row == predicted "null victim"
    action_score   : (num_actions, n_p)
    pred_no_victim : (n_p,)              1 == model explicitly predicted null
    gt_agg_box     : (n_g, 4)
    gt_vic_box     : (n_g, 4)            zeros row == GT "victim not visible"
    gt_act         : (n_g, num_actions)
    gt_visible     : (n_g,) or (n_g, 1)
    """

    def __init__(self, human_actions, scenario_flag=True, iou_threshold=0.5):
        self.human_actions = human_actions
        self.iou_threshold = iou_threshold
        self.scenario_flag = scenario_flag  # True = strict (S1), False = relaxed (S2)

        self.fp = [np.zeros((0,))] * len(human_actions)
        self.tp = [np.zeros((0,))] * len(human_actions)
        self.score = [np.zeros((0,))] * len(human_actions)
        self.num_ann = [0] * len(human_actions)

    # ------------------------------------------------------------------
    def add_data(self, pred_agg_box, pred_vic_box, action_score, pred_no_victim,
                 gt_agg_box, gt_vic_box, gt_act, gt_visible):
        n_p = pred_agg_box.shape[0]
        if n_p == 0:
            return  # nothing predicted → skip

        n_g = gt_agg_box.shape[0]

        # ── Guard: catch action-vocabulary length/ordering mismatches loudly.
        n_act_pred = action_score.shape[0]
        assert n_act_pred == len(self.human_actions), (
            f"HHI_APRole: predicted action_score has {n_act_pred} action "
            f"columns but human_actions has {len(self.human_actions)} entries "
            f"— vocab length mismatch."
        )
        if n_g > 0:
            n_act_gt = gt_act.shape[1]
            assert n_act_gt == len(self.human_actions), (
                f"HHI_APRole: GT gt_act has {n_act_gt} action columns but "
                f"human_actions has {len(self.human_actions)} entries — "
                f"vocab length mismatch (check HHI_idx / valid_ids_HHI ordering "
                f"matches the dataset's HHI_actions column order)."
            )

        if n_g == 0:
            # no GT → every prediction is a false positive
            for label in range(len(self.human_actions)):
                self.score[label] = np.append(self.score[label], action_score[label])
                self.tp[label]    = np.append(self.tp[label],    np.zeros(n_p, dtype=np.uint8))
                self.fp[label]    = np.append(self.fp[label],    np.ones(n_p,  dtype=np.uint8))
            return

        # --- clean padding rows -----------------------------------------
        valid_gt_mask = (gt_act[:, 0] != -1) & (gt_agg_box[:, 0] != -1)
        gt_agg_box = gt_agg_box[valid_gt_mask]
        gt_vic_box = gt_vic_box[valid_gt_mask]
        gt_act     = gt_act[valid_gt_mask]
        gt_visible = gt_visible[valid_gt_mask]
        n_g = gt_agg_box.shape[0]

        if n_g == 0:
            for label in range(len(self.human_actions)):
                self.score[label] = np.append(self.score[label], action_score[label])
                self.tp[label]    = np.append(self.tp[label],    np.zeros(n_p, dtype=np.uint8))
                self.fp[label]    = np.append(self.fp[label],    np.ones(n_p,  dtype=np.uint8))
            return

        if PRINT_DEBUG:
            print("\n[ap_role_HHI][add_data] gt_agg_box:", gt_agg_box)
            print("[ap_role_HHI][add_data] gt_vic_box:", gt_vic_box)
            print("[ap_role_HHI][add_data] gt_act:", gt_act)
            print("[ap_role_HHI][add_data] gt_visible:", gt_visible)

        # --- annotation counts for recall denominator -------------------
        for label in range(len(self.human_actions)):
            self.num_ann[label] += int((gt_act[:, label] == 1).sum())

        gt_is_no_victim = np.all(gt_vic_box == 0, axis=1)            # (n_g,)
        if gt_visible.ndim == 1:
            gt_invisible = (gt_visible == 0)
        else:
            gt_invisible = (gt_visible[:, 0] == 0)
        gt_null = gt_is_no_victim | gt_invisible                      # (n_g,) — authoritative GT null flag

        pred_is_empty_box = np.all(pred_vic_box == 0, axis=1)         # (n_p,)
        pred_null = pred_is_empty_box | (pred_no_victim == 1)          # (n_p,) — authoritative pred null flag

        agg_overlaps = compute_overlap(pred_agg_box, gt_agg_box)       # (n_p, n_g)

        vic_iou_raw = compute_overlap(pred_vic_box, gt_vic_box)        # (n_p, n_g)
        vic_agree = vic_iou_raw.copy()
        both_null    = pred_null[:, None] & gt_null[None, :]
        either_null  = pred_null[:, None] ^ gt_null[None, :]           # exactly one side null
        vic_agree[both_null]   = 1.0
        vic_agree[either_null] = 0.0

        if PRINT_DEBUG:
            print("[ap_role_HHI][add_data] agg_overlaps:", agg_overlaps)
            print("[ap_role_HHI][add_data] vic_agree:", vic_agree)

        # --- per-action scoring -----------------------------------------
        for label in range(len(self.human_actions)):
            sort_inds = np.argsort(action_score[label])[::-1]
            self.score[label] = np.append(self.score[label], action_score[label][sort_inds])

            active_gt = (gt_act[:, label] == 1)   # (n_g,)
            if active_gt.sum() == 0:
                self.tp[label] = np.append(self.tp[label], np.zeros(n_p, dtype=np.uint8))
                self.fp[label] = np.append(self.fp[label], np.ones(n_p,  dtype=np.uint8))
                continue

            # ── Joint matching, restricted to GT rows active for THIS label ──

            agg_ok = (agg_overlaps > self.iou_threshold) & active_gt[None, :]   # (n_p, n_g)
            joint_cost = np.where(agg_ok, agg_overlaps + vic_agree, -1.0)  # (n_p, n_g)

            has_candidate = agg_ok.any(axis=1)             # (n_p,) — at least one eligible GT row
            assigned_gt_idx = np.argmax(joint_cost, axis=1)  # (n_p,) — best row per pred (may be invalid if no candidate)

            max_agg_overlap = agg_overlaps[np.arange(n_p), assigned_gt_idx]
            matched_victim_ok = vic_agree[np.arange(n_p), assigned_gt_idx] > self.iou_threshold


            agg_iou_passed = has_candidate & (max_agg_overlap > self.iou_threshold)

            if PRINT_DEBUG:
                print(f"\n[label={label}] active_gt:", active_gt)
                print(f"[label={label}] assigned_gt_idx:", assigned_gt_idx)
                print(f"[label={label}] agg_iou_passed:", agg_iou_passed)
                print(f"[label={label}] matched_victim_ok:", matched_victim_ok)

            assigned_gt_sorted = assigned_gt_idx[sort_inds]
            agg_passed_sorted  = agg_iou_passed[sort_inds]

            mapped_gt_null = gt_null[assigned_gt_sorted]
            pred_null_sorted = pred_null[sort_inds]

            if self.scenario_flag:  # Scenario 1 — strict: model is held
                vic_passed_sorted = matched_victim_ok[sort_inds]
                agree_null = mapped_gt_null & pred_null_sorted
                agree_real = (~mapped_gt_null) & (~pred_null_sorted) & vic_passed_sorted
                vic_iou_passed = agree_null | agree_real
            else: 
                vic_passed_sorted = matched_victim_ok[sort_inds]
                vic_iou_passed = mapped_gt_null | vic_passed_sorted

            iou_inds = agg_passed_sorted & vic_iou_passed

            p_nonzero = iou_inds.nonzero()[0]
            p_inds    = assigned_gt_sorted[p_nonzero]
            _, unique_idx = np.unique(p_inds, return_index=True)
            p_tp = p_nonzero[unique_idx]   

            t = np.zeros(n_p, dtype=np.uint8)
            t[p_tp] = 1

            self.tp[label] = np.append(self.tp[label], t)
            self.fp[label] = np.append(self.fp[label], 1 - t)

    # ------------------------------------------------------------------
    def evaluate(self, print_log=True):
        """Returns mAP (float, 0-100 scale)."""
        average_precisions = {}
        role_num = 1 if self.scenario_flag else 2

        for label in range(len(self.human_actions)):
            if self.num_ann[label] == 0:
                average_precisions[label] = 0.0
                continue

            global_sort = np.argsort(-self.score[label])
            cum_fp = np.cumsum(self.fp[label][global_sort])
            cum_tp = np.cumsum(self.tp[label][global_sort])

            recall    = cum_tp / self.num_ann[label]
            precision = cum_tp / np.maximum(cum_tp + cum_fp, np.finfo(np.float64).eps)

            average_precisions[label] = _compute_ap(recall, precision) * 100

        if print_log:
            print(f'\n============= AP (HHI Role scenario_{role_num}) ==============')

        s, n = 0, 0
        for label in range(len(self.human_actions)):
            label_name = "_".join(self.human_actions[label].split("_")[1:])
            if print_log:
                print('{: >23}: AP = {:0.2f} (#pos = {:d})'.format(
                    label_name, average_precisions[label], self.num_ann[label]))
            if self.num_ann[label] != 0:
                s += average_precisions[label]
                n += 1

        mAP = s / n if n > 0 else 0.0
        if print_log:
            print('| mAP(role scenario_{:d}): {:0.2f}'.format(role_num, mAP))
            print('----------------------------------------------------')

        return mAP
