"""
detection_recall.py

Unary / Pairwise Detection Recall — a pure detection-layer diagnostic,
deliberately blind to everything downstream (action labels, role
assignment, pointer/query identity, confidence ranking).

Purpose
-------
Answers exactly one question per GT pair:
    "Did the shared DETR detection head produce *some* box, anywhere in
     its candidate set, that spatially matches each required GT box?"

Branch-specific definitions
----------------------------
Branch 1 (HOI / tool triplets) — ⟨Human, Tool⟩:
    A GT pair is "detection-recalled" iff BOTH the human box and the
    object/tool box have some DETR candidate box at IoU > threshold.
    (Standard HOI pairwise instance recall — no null case.)

Branch 2 (HHI / violence triplets) — ⟨Aggressor, Victim⟩:
    A GT pair is "detection-recalled" iff the aggressor box has some
    DETR candidate at IoU > threshold, AND:
        - if the victim is visible (vic_idx != -1): the victim box also
          has some DETR candidate at IoU > threshold, OR
        - if the victim is invisible (vic_idx == -1): this component is
          trivially satisfied (there is no box to miss).
    This is the one place HHI's definition must diverge from the
    textbook HOI version — a missing victim box is not a detection
    failure when the dataset says there was never a box to find.

"""

import numpy as np
from CBIF_HOTR.metrics.utils import compute_overlap


def _any_match(gt_box, cand_boxes, iou_threshold):
    """
    True if at least one candidate box overlaps gt_box above threshold.
    gt_box: (4,) ; cand_boxes: (K, 4). Returns a python bool.
    """
    if cand_boxes.shape[0] == 0:
        return False
    overlaps = compute_overlap(gt_box[None, :], cand_boxes)  # (1, K)
    return bool(overlaps.max() > iou_threshold)


class UnaryDetectionRecallHOI(object):
    """
    Pairwise Instance Recall for Branch 1 (HOI / tool triplets).

    Tracks, across the whole eval set:
      - human_recall   : fraction of GT pairs whose human box was detectable
      - object_recall  : fraction of GT pairs whose object box was detectable
      - pair_recall    : fraction of GT pairs where BOTH were detectable

    This is purely about spatial recoverability — it does NOT check
    whether the action/verb was predicted correctly, nor whether the
    interaction/pointer head selected the right query for that box.
    """

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold
        self.n_total = 0
        self.n_human_hit = 0
        self.n_object_hit = 0
        self.n_pair_hit = 0

        self.per_image_records = []

    def add_data(self, det_boxes, gt_human_box, gt_object_box, has_object,
                 image_name=None):
        """
        Parameters
        ----------
        det_boxes    : (K, 4) ALL detection candidate boxes for this image,
                       pixel-space xyxy, UNFILTERED by confidence.
        gt_human_box : (n_g, 4) GT human boxes, one row per GT pair.
        gt_object_box: (n_g, 4) GT object boxes, one row per GT pair.
                       For VCOCO-style no-object actions, this should be
                       a real box if one exists, otherwise pass a row and
                       set has_object[i] = False for that row (the row's
                       content is then ignored).
        has_object   : (n_g,) bool — whether this GT pair actually has an
                       object (False for no-object actions e.g. "running").
        image_name   : optional str, for per-image diagnostics.
        """
        n_g = gt_human_box.shape[0]
        for i in range(n_g):
            human_hit = _any_match(gt_human_box[i], det_boxes, self.iou_threshold)

            if has_object[i]:
                object_hit = _any_match(gt_object_box[i], det_boxes, self.iou_threshold)
            else:
                object_hit = True  # nothing to miss

            pair_hit = human_hit and object_hit

            self.n_total += 1
            self.n_human_hit  += int(human_hit)
            self.n_object_hit += int(object_hit)
            self.n_pair_hit   += int(pair_hit)

            self.per_image_records.append({
                "image_name": image_name,
                "pair_idx": i,
                "human_hit": human_hit,
                "object_hit": object_hit,
                "pair_hit": pair_hit,
                "has_object": bool(has_object[i]),
            })

    def evaluate(self, print_log=True):
        if self.n_total == 0:
            if print_log:
                print("[UnaryDetectionRecallHOI] No GT pairs accumulated.")
            return {"human_recall": 0.0, "object_recall": 0.0, "pair_recall": 0.0, "n_total": 0}

        human_recall  = 100.0 * self.n_human_hit  / self.n_total
        object_recall = 100.0 * self.n_object_hit / self.n_total
        pair_recall   = 100.0 * self.n_pair_hit   / self.n_total

        if print_log:
            print("\n========= Unary / Pairwise Detection Recall (HOI) =========")
            print(f"  GT pairs total        : {self.n_total}")
            print(f"  Human box recall       : {human_recall:6.2f}%")
            print(f"  Object box recall      : {object_recall:6.2f}%")
            print(f"  PAIR recall (both)     : {pair_recall:6.2f}%")
            print("  ---------------------------------------------------")
            if pair_recall > 90.0:
                print("  Detector is NOT the bottleneck — mAP gap is downstream")
                print("  (action classification / role / pointer assignment).")
            else:
                print("  Detector IS a meaningful bottleneck — some GT boxes are")
                print("  spatially unrecoverable regardless of downstream fixes.")
            print("=====================================================")

        return {
            "human_recall": human_recall,
            "object_recall": object_recall,
            "pair_recall": pair_recall,
            "n_total": self.n_total,
        }


class UnaryDetectionRecallHHI(object):
    """
    Pairwise Instance Recall for Branch 2 (HHI / violence triplets).

    Same idea as the HOI version, but the "object" role (victim) has a
    legitimate null case (vic_idx == -1) that must NOT count against
    detection recall — there is no box to miss when the dataset itself
    says the victim isn't visible.

    Tracks:
      - aggressor_recall      : fraction of GT pairs whose aggressor box
                                 was detectable
      - victim_recall         : fraction of GT pairs whose victim box was
                                 detectable, RESTRICTED to pairs where the
                                 victim is actually visible (denominator
                                 excludes invisible-victim pairs entirely,
                                 so this number is not artificially
                                 inflated by null cases)
      - pair_recall           : fraction of ALL GT pairs where the
                                 aggressor was detectable AND (victim was
                                 detectable OR victim is legitimately
                                 invisible)
      - visible_subset_pair_recall : same as pair_recall but restricted to
                                 visible-victim pairs only, for a clean
                                 apples-to-apples comparison against the
                                 HOI branch's pair_recall (which has no
                                 null case at all)
    """

    def __init__(self, iou_threshold=0.5):
        self.iou_threshold = iou_threshold

        self.n_total = 0
        self.n_visible = 0          # GT pairs with a real, visible victim
        self.n_invisible = 0        # GT pairs with vic_idx == -1

        self.n_agg_hit = 0          # aggressor detectable, over ALL pairs
        self.n_vic_hit_visible = 0  # victim detectable, over VISIBLE pairs only
        self.n_pair_hit = 0         # full pair_recall numerator (all pairs)
        self.n_pair_hit_visible = 0 # pair_recall numerator, visible-only subset

        self.per_image_records = []

    def add_data(self, det_boxes, gt_agg_box, gt_vic_box, vic_visible,
                 image_name=None):
        """
        Parameters
        ----------
        det_boxes   : (K, 4) ALL detection candidate boxes for this image,
                      pixel-space xyxy, UNFILTERED by confidence.
        gt_agg_box  : (n_g, 4) GT aggressor boxes, one row per GT pair.
        gt_vic_box  : (n_g, 4) GT victim boxes. Content of a row is
                      ignored when vic_visible[i] is False.
        vic_visible : (n_g,) bool — True if victim is visible
                      (i.e. dataset's vic_idx != -1), False if invisible.
        image_name  : optional str, for per-image diagnostics.
        """
        n_g = gt_agg_box.shape[0]
        for i in range(n_g):
            agg_hit = _any_match(gt_agg_box[i], det_boxes, self.iou_threshold)
            is_visible = bool(vic_visible[i])

            if is_visible:
                vic_hit = _any_match(gt_vic_box[i], det_boxes, self.iou_threshold)
            else:
                vic_hit = True  # legitimately nothing to detect


            self.n_total += 1
            self.n_agg_hit += int(agg_hit)


            if is_visible:
                self.n_visible += 1
                self.n_vic_hit_visible += int(vic_hit)

            else:
                self.n_invisible += 1

            self.per_image_records.append({
                "image_name": image_name,
                "pair_idx": i,
                "aggressor_hit": agg_hit,
                "victim_hit": vic_hit,

                "vic_visible": is_visible,
            })

    def evaluate(self, print_log=True):
        if self.n_total == 0:
            if print_log:
                print("[UnaryDetectionRecallHHI] No GT pairs accumulated.")
            return {
                "aggressor_recall": 0.0,
                "victim_recall_visible_subset": 0.0,
                "pair_recall": 0.0,
                "pair_recall_visible_subset": 0.0,
                "n_total": 0, "n_visible": 0, "n_invisible": 0,
            }

        aggressor_recall = 100.0 * self.n_agg_hit / self.n_total

        if self.n_visible > 0:
            victim_recall_visible = 100.0 * self.n_vic_hit_visible / self.n_visible
            pair_recall_visible   = 100.0 * self.n_pair_hit_visible / self.n_visible
        else:
            victim_recall_visible = 0.0
            pair_recall_visible   = 0.0

        if print_log:
            print("\n========= Unary / Pairwise Detection Recall (HHI) =========")
            print(f"  GT pairs total              : {self.n_total}")
            print(f"    - visible-victim pairs     : {self.n_visible}")
            print(f"    - invisible-victim pairs   : {self.n_invisible}")
            print(f"  Aggressor box recall (all)   : {aggressor_recall:6.2f}%")
            print(f"  Victim box recall (visible-only subset): {victim_recall_visible:6.2f}%")
            print(f"  PAIR recall (visible-only subset)       : {pair_recall_visible:6.2f}%")
            print("  ---------------------------------------------------")
            if pair_recall_visible > 90.0 and aggressor_recall > 90.0:
                print("  Detector is NOT the bottleneck for HHI — mAP gap is")
                print("  downstream (action / pointer / visibility-head capacity).")
            else:
                print("  Detector IS a meaningful bottleneck for HHI — some GT")
                print("  boxes are spatially unrecoverable regardless of fixes")
                print("  to the action/pointer/visibility heads.")
            print("=====================================================")

        return {
            "aggressor_recall": aggressor_recall,
            "victim_recall_visible_subset": victim_recall_visible,

            "pair_recall_visible_subset": pair_recall_visible,
            "n_total": self.n_total,
            "n_visible": self.n_visible,
            "n_invisible": self.n_invisible,
        }
