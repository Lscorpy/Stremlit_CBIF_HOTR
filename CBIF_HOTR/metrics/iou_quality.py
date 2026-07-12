"""
iou_quality.py

Per-role Mean IoU — reports the CONTINUOUS localization quality that
mAP/AP throws away the moment it applies its IoU > threshold gate.

Branch-specific roles
----------------------
Branch 1 (HOI)  : human, object (object excluded for no-object actions,
                   matching the same convention as detection_recall.py)
Branch 2 (HHI)  : aggressor, victim (victim restricted to VISIBLE-victim
                   matched pairs only — there is no IoU to compute for a
                   correctly-predicted null victim, and including it
                   would either need a fake sentinel value or silently
                   bias the mean; both are worse than excluding it and
                   reporting the visible-only count plainly)

"""

import numpy as np
from CBIF_HOTR.metrics.utils import compute_overlap


class _RoleIoUTracker(object):
    """Accumulates IoU values for one role (e.g. 'aggressor', 'victim')."""

    def __init__(self):
        self.values = []

    def add(self, iou_value):
        self.values.append(float(iou_value))

    def add_many(self, iou_values):
        self.values.extend([float(v) for v in np.asarray(iou_values).ravel()])

    def summary(self):
        if len(self.values) == 0:
            return {"n": 0, "mean": 0.0, "median": 0.0, "std": 0.0,
                     "min": 0.0, "max": 0.0}
        arr = np.asarray(self.values, dtype=np.float64)
        return {
            "n": int(arr.shape[0]),
            "mean":   float(arr.mean()),
            "median": float(np.median(arr)),
            "std":    float(arr.std()),
            "min":    float(arr.min()),
            "max":    float(arr.max()),
        }


class IoUQualityHOI(object):
    """
    Per-role mean IoU for Branch 1 (HOI / tool triplets).

    add_data is called once per image, AFTER matching has already
    happened — pass only the IoU values (or the matched box pairs) for
    pairs that the corresponding APRole instance would have counted as
    a true positive at the given IoU threshold, so this reports
    "how tight are the boxes for predictions we already trust", not
    "how tight are boxes in general including wrong predictions."
    """

    def __init__(self):
        self.human = _RoleIoUTracker()
        self.object = _RoleIoUTracker()

    def add_data(self, pred_h_box, gt_h_box, pred_o_box=None, gt_o_box=None,
                 has_object=None):
        """
        Parameters
        ----------
        pred_h_box, gt_h_box : (n_matched, 4) matched human box pairs
        pred_o_box, gt_o_box : (n_matched_obj, 4) matched object box
            pairs — pass None / omit rows where has_object is False
        has_object : (n_matched,) bool, optional — if provided alongside
            pred_o_box/gt_o_box of the same length as pred_h_box, rows
            where has_object is False are skipped for the object IoU.
        """
        if pred_h_box.shape[0] > 0:
            ious = np.diag(compute_overlap(pred_h_box, gt_h_box))
            self.human.add_many(ious)

        if pred_o_box is not None and gt_o_box is not None and pred_o_box.shape[0] > 0:
            if has_object is not None:
                mask = np.asarray(has_object).astype(bool)
                pred_o_box = pred_o_box[mask]
                gt_o_box = gt_o_box[mask]
            if pred_o_box.shape[0] > 0:
                ious = np.diag(compute_overlap(pred_o_box, gt_o_box))
                self.object.add_many(ious)

    def evaluate(self, print_log=True):
        human_s = self.human.summary()
        object_s = self.object.summary()

        if print_log:
            print("\n========= Per-Role Mean IoU (HOI) =========")
            print(f"  Human  : n={human_s['n']:5d}  mean={human_s['mean']:.3f}  "
                  f"median={human_s['median']:.3f}  std={human_s['std']:.3f}  "
                  f"min={human_s['min']:.3f}  max={human_s['max']:.3f}")
            print(f"  Object : n={object_s['n']:5d}  mean={object_s['mean']:.3f}  "
                  f"median={object_s['median']:.3f}  std={object_s['std']:.3f}  "
                  f"min={object_s['min']:.3f}  max={object_s['max']:.3f}")
            print("  (computed over MATCHED true-positive pairs only — this is")
            print("   localization tightness, not detection/classification correctness)")
            print("=============================================")

        return {"human": human_s, "object": object_s}


class IoUQualityHHI(object):
    """
    Per-role mean IoU for Branch 2 (HHI / violence triplets).

    Victim IoU is restricted to VISIBLE-victim matched pairs only.
    Invisible-victim correct predictions (null matches null) have no
    spatial IoU to report — that correctness is already captured by
    HHI_APRole's AP, not by this localization-quality metric.
    """

    def __init__(self):
        self.aggressor = _RoleIoUTracker()
        self.victim = _RoleIoUTracker()

    def add_data(self, pred_agg_box, gt_agg_box,
                 pred_vic_box=None, gt_vic_box=None, vic_visible=None):
        """
        Parameters
        ----------
        pred_agg_box, gt_agg_box : (n_matched, 4) matched aggressor pairs
        pred_vic_box, gt_vic_box : (n_matched, 4) matched victim pairs,
            same length/order as pred_agg_box
        vic_visible : (n_matched,) bool — True where the victim is
            actually visible for that matched pair; rows where this is
            False are excluded from the victim IoU computation entirely
            (not given a sentinel value of 0 or 1 — simply not counted).
        """
        if pred_agg_box.shape[0] > 0:
            ious = np.diag(compute_overlap(pred_agg_box, gt_agg_box))
            self.aggressor.add_many(ious)

        if pred_vic_box is not None and gt_vic_box is not None and pred_vic_box.shape[0] > 0:
            if vic_visible is not None:
                mask = np.asarray(vic_visible).astype(bool)
                pred_vic_box = pred_vic_box[mask]
                gt_vic_box = gt_vic_box[mask]
            if pred_vic_box.shape[0] > 0:
                ious = np.diag(compute_overlap(pred_vic_box, gt_vic_box))
                self.victim.add_many(ious)

    def evaluate(self, print_log=True):
        agg_s = self.aggressor.summary()
        vic_s = self.victim.summary()

        if print_log:
            print("\n========= Per-Role Mean IoU (HHI) =========")
            print(f"  Aggressor : n={agg_s['n']:5d}  mean={agg_s['mean']:.3f}  "
                  f"median={agg_s['median']:.3f}  std={agg_s['std']:.3f}  "
                  f"min={agg_s['min']:.3f}  max={agg_s['max']:.3f}")
            print(f"  Victim    : n={vic_s['n']:5d}  mean={vic_s['mean']:.3f}  "
                  f"median={vic_s['median']:.3f}  std={vic_s['std']:.3f}  "
                  f"min={vic_s['min']:.3f}  max={vic_s['max']:.3f}")
            print("  (victim IoU restricted to VISIBLE-victim matched pairs only;")
            print("   computed over true-positive pairs, not all predictions)")
            print("=============================================")

        return {"aggressor": agg_s, "victim": vic_s}
