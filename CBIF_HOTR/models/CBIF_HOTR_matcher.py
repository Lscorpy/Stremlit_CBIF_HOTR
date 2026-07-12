# ------------------------------------------------------------------------
# Modified from HOTR (https://github.com/kakaobrain/hotr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Modules to compute the matching cost and solve the corresponding LSAP.
""" 

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn

from CBIF_HOTR.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou
import CBIF_HOTR.util.misc as utils
 
import wandb

PRINT_COST_OUTPUT = False


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes 
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox  * cost_bbox + \
            self.cost_class * cost_class + \
            self.cost_giou  * cost_giou
        
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # print(targets[]["imgae_name_for_track"])
        if PRINT_COST_OUTPUT:
            print(f"\n  Matched {len(indices[0])} DETR_ORI predictions to {len(indices[1])} GT pairs")
            for idx in range(2):
                print(targets[idx]["imgae_name_for_track"])
                a,b=indices[idx]
                print("predict outcome")
                # print(outputs["pred_human_box"])
                print(a)
                for i in a:
                    print(f"query {i} :",outputs["pred_boxes"][idx][i])
                print("Actual box")
                # print(b)
                # print(targets["human_boxes"])
                for i in b:
                    print(f"query {i} :",targets[idx]["boxes"][i])
                print("%"*50)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class DETR_HHI_HungarianMatcher(nn.Module):
    """
    Computes an optimal 1-to-1 matching between the base DETR object queries 
    and the human-only ground-truth targets for HHI training.
    """
 
    def __init__(self,cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()

        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert  cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):

        bs, num_queries = outputs["pred_human_box"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_bbox = outputs["pred_human_box"].flatten(0, 1)  # [batch_size * num_queries, 4]

        tgt_bbox = torch.cat([v["human_boxes"] for v in targets])

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + \
            self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["human_boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        if PRINT_COST_OUTPUT:
        
            print(f"\n  Matched {len(indices[0])} DETR_HHI predictions to {len(indices[1])} GT pairs")

            for idx in range(2):
                print(targets[idx]["imgae_name_for_track"])
                a,b=indices[idx]
                # print("predict outcome")
                # print(outputs["pred_human_box"])
                print(a)
                for i in a:
                    print(f"query {i} :",outputs["pred_boxes"][idx][i])
                print("Actual box")
                # print(b)
                # print(targets["human_boxes"])
                for i in b:
                    print(f"query {i} :",targets[idx]["boxes"][i])
                print("%"*50)

        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class HungarianPairMatcher(nn.Module):
    """
    Hungarian matcher for the HOTR pair-interaction head.

    Runs one independent assignment per mode (vcoco / violence) and returns
    separate index lists and h/o label lists for each head so they never
    overwrite each other.
    """

    def __init__(self, args):
        super().__init__()
        self.cost_action = args.set_cost_act
        self.cost_hbox   = args.set_cost_idx
        self.cost_obox   = args.set_cost_idx

        self.invalid_ids_vcoco = args.invalid_ids_vcoco
        self.valid_ids_vcoco   = args.valid_ids_vcoco
        self.valid_ids_violence = args.valid_ids_violence

        self.log_printer = args.wandb


        assert self.cost_action != 0 or self.cost_hbox != 0 or self.cost_obox != 0, \
            "All matching costs are 0 — check your args."

    # ──────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def forward(self, outputs, targets, detr_indices):
        """
        Run both heads and return their independent assignments.

        Args:
            outputs       : model output dict (pred_actions, pred_violence_actions, …)
            targets       : list of per-image target dicts (len == batch_size)
            detr_indices  : DETR Hungarian output — list of (src_idx, tgt_idx) tuples,
                            one per image.  Used to map GT boxes → DETR query indices.

        Returns:
            vcoco_indices    : list of (src, tgt) int64 tensors — VCOCO assignment
            violence_indices : list of (src, tgt) int64 tensors — Violence assignment
            h_labels_vcoco   : list[Tensor]  — GT human query idx per image (VCOCO)
            o_labels_vcoco   : list[Tensor]  — GT object query idx per image (VCOCO)
            h_labels_violence: list[Tensor]  — GT human query idx per image (violence)
            o_labels_violence: list[Tensor]  — GT object query idx per image (violence)
            has_vcoco_labels : list[bool]    — True if image has VCOCO-valid annotations
        """
        assert "pred_actions" in outputs,           "Missing pred_actions"
        assert "pred_violence_actions" in outputs,  "Missing pred_violence_actions"


        bs = outputs["pred_actions"].shape[0] # same for action and violence action
        bs, num_hoi_queries = outputs["pred_actions"].shape[:2]
        num_det_queries = outputs["pred_boxes"].shape[1]
        if PRINT_COST_OUTPUT:
            print("\n[DEBUG INFO HOTR MATCHER]--------")
            print(f"Batch size: {bs}, HOTR queries: {num_hoi_queries}, DETR queries: {num_det_queries}")#------------------------------------------------------------------

        vcoco_indices = []
        violence_indices = []
        

        for b_idx in range(bs):
            # ── 1. Build the shared bbox look-up table ─────────────────────
            tgt_bbox = targets[b_idx]["boxes"]    # (N, 4)
            tgt_cls  = targets[b_idx]["labels"]   # (N,)
            device   = tgt_bbox.device

            if PRINT_COST_OUTPUT:
                print("\n",targets[b_idx]["imgae_name_for_track"])

            targets[b_idx]["pair_actions"][:, self.invalid_ids_vcoco] = 0

            # 2. Row-by-row check 
            keep_idx = (targets[b_idx]["pair_actions"].sum(dim=-1) != 0)

            # 3. Convert dataset labels to a matching boolean tensor 
            has_vcoco_bool = (targets[b_idx]["has_vcoco_labels"] == 1)

            if keep_idx.shape!=has_vcoco_bool.shape:
                print("targets[b_idx][\"pair_actions\"]",targets[b_idx]["pair_actions"])
                print("keep_idx",keep_idx)
                print("has_vcoco_bool",has_vcoco_bool)
                print(targets[b_idx]["imgae_name_for_track"])


            # 4. Safe element-by-element comparison
            if (keep_idx != has_vcoco_bool).any():
                print(targets[b_idx]["imgae_name_for_track"])
                print(f"error: action check mismatch at batch {b_idx}!")
                print(f"Calculated: {keep_idx.tolist()} | Dataset: {has_vcoco_bool.tolist()}")
                print(targets[b_idx]["has_vcoco_labels"])
                print(keep_idx)
                print("error the action check not match with dataset vcoco_labels (invalid vcoco action)")


            num_pairs = max(targets[b_idx]["pair_actions"].sum(),0)
            num_pairs_vio = max(targets[b_idx]["pair_violence"].sum(),0)

            if PRINT_COST_OUTPUT:
                print(f"  Batch {b_idx}: {num_pairs} GT pairs, {len(detr_indices[b_idx][0])} DETR matches(get from detr)")
                print(f"  Batch {b_idx}: {num_pairs_vio} GT violence pairs, {len(detr_indices[b_idx][0])} DETR matches(get from detr)")

            bbox_with_cls = torch.cat([tgt_bbox, tgt_cls.unsqueeze(-1).float()], dim=1)
            bbox_with_cls, k_idx, bbox_idx = self._reduce_redundant_gt_box(
                bbox_with_cls, detr_indices[b_idx]
            )
            # Append a sentinel row for "no object" (occluded / missing)
            sentinel = torch.as_tensor([-1.]*5, device=device).unsqueeze(0).to(device)
            bbox_with_cls = torch.cat([bbox_with_cls, sentinel], dim=0)

            k_idx    = k_idx.to(device)
            bbox_idx = bbox_idx.to(device)

            # # ── VCOCO head assignment ────────────────────────────────────
            vi, h_v, o_v = self._match_vcoco(
                b_idx, outputs, 
                targets[b_idx], 
                bbox_with_cls, k_idx, bbox_idx, 
                device
            )
            vcoco_indices.append(vi)
            targets[b_idx]["h_labels"] = h_v.to(device)
            targets[b_idx]["o_labels"] = o_v.to(device)

            # ── Violence head assignment ─────────────────────────────────
            vi2, h_vio, o_vio = self._match_violence(
                b_idx, outputs, 
                targets[b_idx], 
                bbox_with_cls, k_idx, bbox_idx, 
                device
            )
            violence_indices.append(vi2)
            targets[b_idx]["h_labels_v"] = h_vio.to(device)
            targets[b_idx]["o_labels_v"] = o_vio.to(device)


        def _to_int64(pairs):
            return [(torch.as_tensor(i, dtype=torch.int64),
                     torch.as_tensor(j, dtype=torch.int64)) for i, j in pairs]
        
        
        return (
            _to_int64(vcoco_indices),
            _to_int64(violence_indices),
            targets
        )
    
    # ──────────────────────────────────────────────────────────────────────────
    # Per-head matching helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _match_vcoco(self, b_idx, outputs, tgt, bbox_with_cls, k_idx, bbox_idx, device):
        """Run Hungarian matching for the VCOCO action head."""

        num_queries = outputs["pred_actions"].shape[1]

        # Zero out invalid action slots and filter pairs with nothing left
        pair_actions = tgt["pair_actions"].clone()

        tgt_pbox = tgt["pair_boxes"]        # (P, 8) paired interactions: [human_box(4), object_box(4)]
        tgt_act  = pair_actions              # (P, 29) ground-truth action labels
        tgt_tgt  = tgt["pair_targets"]      # (P,)  ground-truth object class in the pair


        keep = pair_actions.sum(dim=-1) != 0
        has_vcoco = bool(keep.any().item())

        if not has_vcoco: 
            indices=(
                torch.as_tensor([], dtype=torch.int64),
                torch.as_tensor([], dtype=torch.int64)
            )

            tgt_hids = torch.as_tensor([], dtype=torch.int64, device=device)
            

            tgt_oids = torch.as_tensor([], dtype=torch.int64, device=device)

            if PRINT_COST_OUTPUT:
                print()
                pred_pair_idx,gt_pair_idx=indices
                print(f"  Matched {len(pred_pair_idx)} VCOCO_HOTR predictions to {len(gt_pair_idx)} GT pairs")
                for p, g in zip(pred_pair_idx[:3], gt_pair_idx[:3]):
                    print(f"    VCOCO_HOTR query {p} -> Pair {g} (cost: {C[p, g]:.3f})")


            return indices, tgt_hids, tgt_oids

        tgt_hbox = tgt_pbox[:, :4] # (num_pair_boxes, 4)  ground-truth human and object boxes for each pair
        tgt_obox = tgt_pbox[:, 4:]

        # Build cost matrix
        C, tgt_hids, tgt_oids = self._build_cost_matrix(
            b_idx, outputs, "pred_actions", self.valid_ids_vcoco,
            tgt_hbox, tgt_obox, tgt_act, tgt_tgt,
            bbox_with_cls, k_idx, bbox_idx, device, num_queries,
            person_cls_append=1.0,  # V-COCO uses class 1 for person
        )

        indices = linear_sum_assignment(C)

        if PRINT_COST_OUTPUT:
            print()
            pred_pair_idx,gt_pair_idx=indices
            print(f"  Matched {len(pred_pair_idx)} VCOCO_HOTR predictions to {len(gt_pair_idx)} GT pairs")
            print(f"  Cost range: [{C.min():.3f}, {C.max():.3f}]")
            for p, g in zip(pred_pair_idx[:3], gt_pair_idx[:3]):
                print(f"    HOTR query {p} -> Pair {g} (cost: {C[p, g]:.3f})")


        return indices, tgt_hids, tgt_oids

    def _match_violence(self, b_idx, outputs, tgt, bbox_with_cls, k_idx, bbox_idx, device):
        """Run Hungarian matching for the Violence action head."""

        num_queries = outputs["pred_violence_actions"].shape[1]

        tgt_pbox = tgt["pair_boxes"]          # (P, 8)
        tgt_act  = tgt["pair_violence"]        # (P, 6)
        tgt_tgt  = tgt["pair_targets"]         # (P,)

        tgt_hbox = tgt_pbox[:, :4]
        tgt_obox = tgt_pbox[:, 4:]


        C, tgt_hids, tgt_oids = self._build_cost_matrix(
            b_idx, outputs, "pred_violence_actions", self.valid_ids_violence,
            tgt_hbox, tgt_obox, tgt_act, tgt_tgt,
            bbox_with_cls, k_idx, bbox_idx, device, num_queries,
            person_cls_append=1.0,
        )

        indices = linear_sum_assignment(C)
        if PRINT_COST_OUTPUT:
            pred_pair_idx,gt_pair_idx=indices
            print()
            print(f"  Matched {len(pred_pair_idx)} Violence_HOTR predictions to {len(gt_pair_idx)} GT pairs")
            print(f"  Cost range: [{C.min():.3f}, {C.max():.3f}]")
            for p, g in zip(pred_pair_idx[:3], gt_pair_idx[:3]):
                print(f"    HOTR query {p} -> Pair {g} (cost: {C[p, g]:.3f})")

        return indices, tgt_hids, tgt_oids

    # ──────────────────────────────────────────────────────────────────────────
    # Core cost-matrix builder  (shared by both heads)
    # ──────────────────────────────────────────────────────────────────────────
    def _build_cost_matrix(
        self, b_idx, outputs, pred_key, valid_ids,
        tgt_hbox, tgt_obox, tgt_act, tgt_tgt,
        bbox_with_cls, k_idx, bbox_idx, device, num_queries,
        person_cls_append=1.0,
    ):
        """
        Compute the [num_queries × num_gt_pairs] cost matrix for one head.

        Returns:
            C         : numpy cost matrix ready for linear_sum_assignment
            tgt_hids  : Tensor — which DETR query is the GT human for each pair
            tgt_oids  : Tensor — which DETR query is the GT object for each pair
        """
        # ── Pair → GT box index lookup ─────────────────────────────────────
        hbox_with_cls = torch.cat([tgt_hbox, torch.ones((tgt_hbox.shape[0], 1)).to(device)], dim=1)

        obox_with_cls = torch.cat([tgt_obox, tgt_tgt.unsqueeze(-1).float()], dim=1)
        # Occluded / missing objects get class sentinel −1
        obox_with_cls[obox_with_cls[:, :4].sum(dim=1) == -4, -1] = -1

        cost_hbox = torch.cdist(hbox_with_cls, bbox_with_cls, p=1)
        cost_obox = torch.cdist(obox_with_cls, bbox_with_cls, p=1)


        h_match = torch.nonzero(cost_hbox == 0, as_tuple=False)
        o_match = torch.nonzero(cost_obox == 0, as_tuple=False)

        k_idx = k_idx.to(device)
        bbox_idx = bbox_idx.to(device)

        # n_pairs = min(len(h_match), len(o_match))
        tgt_hids, tgt_oids = [], []

        n_pairs = min(len(h_match), len(o_match))  # should always be equal
        for idx in range(n_pairs):
            _, H_bbox_idx = h_match[idx] # which unique box is the human?
            _, O_bbox_idx = o_match[idx] # which unique box is the object?

            # If object was occluded/missing, fall back to the human's bbox slot
            if O_bbox_idx == (len(bbox_with_cls) - 1):
                O_bbox_idx = H_bbox_idx

            q_h = k_idx[(bbox_idx == H_bbox_idx).nonzero(as_tuple=False).squeeze(-1)] # → DETR query ID
            q_o = k_idx[(bbox_idx == O_bbox_idx).nonzero(as_tuple=False).squeeze(-1)] # → DETR query ID

            tgt_hids.append(q_h)
            tgt_oids.append(q_o)
        

        no_gt_match = (len(tgt_hids) == 0)

        # Guard: if nothing matched, use sentinel −1 (ignored by cross-entropy)
        if len(tgt_hids) == 0:
            tgt_hids.append(torch.as_tensor([-1]))
        if len(tgt_oids) == 0:
            tgt_oids.append(torch.as_tensor([-1]))

        tgt_hids = torch.cat(tgt_hids).to(device)
        tgt_oids = torch.cat(tgt_oids).to(device)

        # ── Action cost ────────────────────────────────────────────────────
        # Append a "no-interaction" column to tgt_act
        if tgt_act.shape[0] == 0:
            # No GT pairs → dummy row so linear_sum_assignment doesn't crash
            tgt_act = torch.zeros((1, tgt_act.shape[1]), device=device)
            tgt_sum = torch.ones(1, device=device).unsqueeze(0)
        else:
            tgt_sum = tgt_act.sum(dim=-1).unsqueeze(0).float()   # (1, P)

        tgt_act_ext = torch.cat(
            [tgt_act, torch.zeros(tgt_act.shape[0], 1, device=device)], dim=-1
        )  # (P, C+1)

        out_act = outputs[pred_key][b_idx].clone().float()  # (Q, C+1)


        cost_pos = (-torch.matmul(out_act, tgt_act_ext.t())) / tgt_sum.clamp(min=1)
        neg_mask = (~tgt_act_ext.bool()).float()
        neg_denom = neg_mask.sum(dim=-1).unsqueeze(0).clamp(min=1)
        cost_neg  = torch.matmul(out_act, neg_mask.t()) / neg_denom
        cost_action = cost_pos + cost_neg       # (100_queries, num_pairs)                    # (Q, P)

        # ── Pointer costs ──────────────────────────────────────────────────
        out_hprob = outputs["pred_hidx"][b_idx].softmax(-1)  
        out_oprob = outputs["pred_oidx"][b_idx].softmax(-1) 


        # How well do pred_hidx/oidx predictions match GT human/object assignments?
        cost_h = -out_hprob[:, tgt_hids]   # (Q, P) # (16_pairs, num_gt_pairs)
        cost_o = -out_oprob[:, tgt_oids]    # (Q, P)  # (16_pairs, num_gt_pairs)

        assert cost_h.shape == cost_o.shape == cost_action.shape, \
            f"Shape mismatch: h={cost_h.shape}, o={cost_o.shape}, act={cost_action.shape}"

        C = (self.cost_hbox * cost_h
             + self.cost_obox * cost_o
             + self.cost_action * cost_action)

        return C.view(num_queries, -1).cpu().numpy(), tgt_hids, tgt_oids

    # ──────────────────────────────────────────────────────────────────────────
    # Utility
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _reduce_redundant_gt_box(bbox_with_cls, detr_index):
        """
        Remove duplicate GT boxes that arise from random-crop augmentation.

        bbox_with_cls : (N, 5)  [cx, cy, w, h, cls]
        detr_index    : (src_idx, tgt_idx) from DETR Hungarian output

        Returns:
            bbox_unique : (M, 5)  deduplicated boxes
            k_idx       : 1-D tensor — DETR query indices corresponding to each unique box
            bbox_idx    : 1-D tensor — unique-box indices corresponding to each query
        """
        bbox_unique, map_idx, _ = torch.unique(
            bbox_with_cls, dim=0, return_inverse=True, return_counts=True
        )


        k_idx, raw_bbox_idx = detr_index  # both are 1-D tensors of the same length

        if len(bbox_with_cls) != len(bbox_unique):
            # Deduplicate: keep first occurrence of each unique box
            map_dict        = {int(orig): int(uniq) for orig, uniq in enumerate(map_idx)}
            seen, bbox_lst, k_lst = set(), [], []
            for b_id, k_id in zip(raw_bbox_idx.tolist(), k_idx.tolist()):
                uniq_id = map_dict[b_id]
                if uniq_id not in seen:
                    seen.add(uniq_id)
                    bbox_lst.append(uniq_id)
                    k_lst.append(k_id)
            bbox_idx = torch.tensor(bbox_lst, device=bbox_with_cls.device)
            k_idx    = torch.tensor(k_lst,    device=bbox_with_cls.device)
        else:
            bbox_idx = raw_bbox_idx.to(bbox_with_cls.device)
            k_idx    = k_idx.to(bbox_with_cls.device)
            bbox_unique = bbox_with_cls
        return bbox_unique, k_idx, bbox_idx


def to_idx_tensor(x, dtype, device):
    if torch.is_tensor(x):
        return x.clone().detach().to(dtype=dtype, device=device)
    return torch.as_tensor(x, dtype=dtype, device=device)

class HungarianHHIMatcher(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Cost weights for different matching components
        self.cost_action = args.set_cost_act      # Weight for action classification cost
        self.cost_agg = args.set_cost_idx         # Weight for aggressor pointer cost
        self.cost_vic = args.set_cost_idx         # Weight for victim pointer cost
        self.cost_visible = args.set_cost_idx     # Weight for visibility cost

        self.log_printer = args.wandb

        self.cost_exclusive =args.set_cost_idx

        assert self.cost_action != 0 or self.cost_agg != 0 or self.cost_vic != 0, \
            "All matching costs are 0 — check your args."

    @torch.no_grad()
    def forward(self, outputs, targets, detr_indices):
        ''' Performs bipartite matching for violence pairs using DETR matcher results.
        
        Follows HOTR approach: uses DETR matches to find which queries correspond to
        aggressor/victim boxes, then computes pointer and action costs.

        Params:
            outputs: Dict with HHI predictions
            targets: List of dicts with GT violence pairs
            detr_indices: List of (query_idx, gt_box_idx) from DETR matcher

        Returns:
            List of (pred_HHI_idx, gt_pair_idx) tuples for each batch
        '''
        bs, num_HHI_queries = outputs["pred_action_logits"].shape[:2] #2,8
        num_det_queries_victim = outputs["pred_victim_idx"].shape[2]-1 #101
        

        return_list = []
        if self.log_printer and self.log:
            log_dict = {'a_cost': [], 'v_cost': [], 'act_cost': [],'cis_cost':[]}
        
        for batch_idx in range(bs):
            # ===== Extract DETR Matcher Results =====
            detr_query_idx, detr_gt_idx = detr_indices[batch_idx]  # Maps queries to GT boxes
            
            device=targets[batch_idx]["human_boxes"].device

            keep_idx = (targets[batch_idx]["violence_actions"].sum(dim=-1) != 0)
            
            # Prune out the completely idle background pairs from this image batch
            targets[batch_idx]["aggressor_index"] = targets[batch_idx]["aggressor_index"][keep_idx]
            targets[batch_idx]["victim_index"] = targets[batch_idx]["victim_index"][keep_idx]
            targets[batch_idx]["violence_actions"] = targets[batch_idx]["violence_actions"][keep_idx]
            targets[batch_idx]["has_target_visible"] = targets[batch_idx]["has_target_visible"][keep_idx]

            target_agg_idx = targets[batch_idx]["aggressor_index"]      # [num_pairs(valid)] - indices in human_boxes
            target_vic_idx = targets[batch_idx]["victim_index"]         # [num_pairs(valid)] - indices in human_boxes
            target_actions = targets[batch_idx]["violence_actions"]    # [num_pairs(valid), num_HHI_action]
            target_visible = targets[batch_idx]["has_target_visible"]  # [num_pairs(valid)]



            num_pairs = len(target_actions)

            
            # Handle empty targets
            if num_pairs == 0:
                # 1. Keep the indices completely empty so all loss list comprehensions skip smoothly
                return_list.append((
                    torch.as_tensor([], dtype=torch.int64),
                    torch.as_tensor([], dtype=torch.int64)
                ))
                num_actions = 4  
                targets[batch_idx]['violence_actions'] = torch.zeros((0, num_actions), dtype=torch.int64,device=device)
                targets[batch_idx]["has_target_visible"] = torch.as_tensor([], dtype=torch.float32, device=device)
                targets[batch_idx]["a_labels"] = torch.as_tensor([], dtype=torch.int64, device=device)
                targets[batch_idx]["v_labels"] = torch.as_tensor([], dtype=torch.int64, device=device)
                continue
            

            # Create mapping: gt_box_idx -> query_idx
            gt_box_to_query = {}  # maps GT box index to query index
            for q_idx, gt_idx in zip(detr_query_idx, detr_gt_idx):
                gt_box_to_query[int(gt_idx)] = int(q_idx)


            null_query_token = num_det_queries_victim  # Index 100 becomes the null pointer slot
            # For each GT pair, get the query indices of the aggressor and victim
            gt_agg_query_idx = []
            gt_vic_query_idx = []

            
            for pair_idx in range(num_pairs):

                agg_box_idx = int(target_agg_idx[pair_idx])
                vic_box_idx = int(target_vic_idx[pair_idx])
                agg_query = gt_box_to_query[agg_box_idx]
                
                # --- SAFE VICTIM LOOKUP ---
                if vic_box_idx == -1:
                    vic_query = null_query_token
                else:
                    # Otherwise, retrieve the mapped query index safely
                    vic_query = gt_box_to_query[vic_box_idx]
                
                
                gt_agg_query_idx.append(agg_query)
                gt_vic_query_idx.append(vic_query)

            if len(gt_agg_query_idx) == 0: gt_agg_query_idx.append(torch.as_tensor([-1])) # we later ignore the label -1
            if len(gt_vic_query_idx) == 0: gt_vic_query_idx.append(torch.as_tensor([-1])) # we later ignore the label -1

            gt_agg_query_idx = torch.tensor(gt_agg_query_idx, dtype=torch.int64, device=device)
            gt_vic_query_idx = torch.tensor(gt_vic_query_idx, dtype=torch.int64, device=device)

            # ===== Extract Predictions =====
            pred_action = outputs["pred_action_logits"][batch_idx].clone()   # [num_HHI_queries, num_actions+1]
            pred_agg_ptr = outputs["pred_aggressor_idx"][batch_idx]  # [num_HHI_queries, num_det_queries]
            pred_vic_ptr = outputs["pred_victim_idx"][batch_idx]     # [num_HHI_queries, num_det_queries+1]
            pred_vis = outputs["pred_victim_visible"][batch_idx]     # [num_HHI_queries, 2]
            

            # ===== Compute Cost Matrix [num_HHI_queries, num_pairs] =====
            target_sum = (target_actions.sum(dim=-1)).unsqueeze(0)
            target_actions = torch.cat([target_actions, 
                                        torch.zeros(target_actions.shape[0]).unsqueeze(-1).to(device)
                                        ], dim=-1)
    
            
            # Action cost: CE loss on correct action
            cost_pos_action = (-torch.matmul(pred_action, target_actions.t().float())) / target_sum
            cost_neg_action = (torch.matmul(pred_action, (~target_actions.bool()).type(torch.int64).t().float())) / (~target_actions.bool()).type(torch.int64).sum(dim=-1).unsqueeze(0)
            action_cost = cost_pos_action + cost_neg_action

            # Aggressor pointer cost: vectorized like HOTR
            agg_probs = F.softmax(pred_agg_ptr, dim=-1)  # [num_HHI_queries, num_det_queries]
            agg_cost = -agg_probs[:, gt_agg_query_idx]  # [num_HHI_queries, num_pairs]
            # Mask invalid pairs (where DETR didn't match aggressor box)
            agg_cost[:, gt_agg_query_idx < 0] = 1e10
            
            # Victim pointer cost: vectorized with visibility handling
            vic_probs = F.softmax(pred_vic_ptr, dim=-1)  # [num_HHI_queries, num_det_queries+1]

            # Grab the victim costs ASSUMING they are visible
            safe_indices = torch.clamp(gt_vic_query_idx, min=0)
            vic_cost = -vic_probs[:, safe_indices]

            # If invisible (target_visible <= 0.5), ignore by setting cost to 0
            is_invisible = (target_visible <= 0.5)
            vic_cost[:, is_invisible] = 0.0  # Zero penalty! Matcher will ignore this column's victim cost.

            # Mask invalid pairs (Safety guard: if it SHOULD be visible but DETR missed it)
            invalid_visible = (target_visible > 0.5) & (gt_vic_query_idx < 0)
            vic_cost[:, invalid_visible] = 1e10
            
            # Visibility cost: vectorized like HOTR
            vis_probs = F.softmax(pred_vis, dim=-1)  # [num_HHI_queries, 2]
            vis_labels = target_visible.long()  # [num_pairs]
            vis_cost = -vis_probs[:, vis_labels]  # [num_HHI_queries, num_pairs]

            # ===== Exclusivity cost (discourage agg==vic predictions) =====
            # Compute probability that aggressor and victim pointers coincide per HHI query
            agg_probs = F.softmax(pred_agg_ptr, dim=-1)
            vic_probs = F.softmax(pred_vic_ptr, dim=-1)
            # exclude the victim-null slot when computing same-box probability
            vic_probs_no_null = vic_probs[:, :pred_agg_ptr.shape[1]]
            p_same = (agg_probs * vic_probs_no_null).sum(dim=-1, keepdim=True)  # [Q, 1]

            # Mask: only penalize when GT aggressor and victim are different and victim is visible
            different_pair_mask = (gt_agg_query_idx != gt_vic_query_idx)  # [num_pairs]
            visible_mask = (target_visible > 0.5)
            apply_mask = (different_pair_mask & visible_mask).to(p_same.device)  # [num_pairs]

            # Expand to [Q, num_pairs] and zero-out columns where not applicable
            excl_cost = p_same.repeat(1, num_pairs)
            if excl_cost.size(1) == apply_mask.size(0):
                excl_cost[:, ~apply_mask] = 0.0

            # ===== Weighted Sum of Costs =====
            total_cost = (
                self.cost_action * action_cost +
                self.cost_agg * agg_cost +
                self.cost_vic * vic_cost +
                self.cost_visible * vis_cost
            )

            if self.cost_exclusive != 0.0:
                total_cost = total_cost + self.cost_exclusive * excl_cost

            # Move the total cost matrix to CPU for SciPy
            total_cost = total_cost.cpu().detach()

            pred_HHI_idx, gt_pair_idx = linear_sum_assignment(total_cost)


            if PRINT_COST_OUTPUT:
                print(f"  Matched {len(pred_HHI_idx)} HHI predictions to {len(gt_pair_idx)} GT pairs")
                print(f"  Cost range: [{total_cost.min():.3f}, {total_cost.max():.3f}]")
                for p, g in zip(pred_HHI_idx[:3], gt_pair_idx[:3]):
                    print(f"    HHI query {p} -> Pair {g} (cost: {total_cost[p, g]:.3f})")
            

            return_list.append((
                torch.as_tensor(pred_HHI_idx, dtype=torch.int64),
                torch.as_tensor(gt_pair_idx, dtype=torch.int64)
            ))
            if len(gt_agg_query_idx) == 0:
                targets[batch_idx]["a_labels"] = torch.empty(0, dtype=torch.int64, device=device)
                targets[batch_idx]["v_labels"] = torch.empty(0, dtype=torch.int64, device=device)
            else:
                targets[batch_idx]["a_labels"] = to_idx_tensor(gt_agg_query_idx, torch.int64, device)
                targets[batch_idx]["v_labels"] = to_idx_tensor(gt_vic_query_idx, torch.int64, device)

            
            if self.log_printer and self.log:
                log_dict['a_cost'].append(agg_cost.min(dim=0)[0].mean())
                log_dict['v_cost'].append(vic_cost.min(dim=0)[0].mean())
                log_dict['act_cost'].append(action_cost.min(dim=0)[0].mean())
                log_dict['vis_cost'].append(vis_cost.min(dim=0)[0].mean())
                
        if self.log_printer and self.log:
            log_dict['a_cost'] = torch.stack(log_dict['a_cost']).mean()
            log_dict['v_cost'] = torch.stack(log_dict['v_cost']).mean()
            log_dict['act_cost'] = torch.stack(log_dict['act_cost']).mean()
            log_dict['vis_cost'].append(log_dict['vis_cost'].mean())
            if utils.get_rank() == 0: wandb.log(log_dict)

        return return_list, targets


def build_HHI_matcher(args):
    return HungarianHHIMatcher(args)

def build_HOI_matcher(args):
    return HungarianPairMatcher(args)

def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)

def build_DETR_HHI_matcher(args):
    return DETR_HHI_HungarianMatcher(cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)
