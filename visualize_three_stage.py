"""
visualize_three_stage.py
=========================
Three-stage diagnostic visualization for the CBIF_HOTR (HOI + HHI).

"""

# =============================================================================
# Imports
# =============================================================================
import argparse
import json
import os
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

DATA_PATH  = r"test\gun_violence (61)_frame_00002.jpg"
WEIGHTS    = r"best.pth"
OUTPUT_DIR = r"three_stage_out2"

SHOW_PLOTS = False


# =============================================================================
# ── Colour / style constants ──────────────────────────────────────────────
# =============================================================================
# Stage 1 — raw detections
COL_DET_HUMAN   = "#00FF99"   # mint green   — any human-class detection
COL_DET_OBJECT  = "#5DA9E9"   # sky blue     — any non-human confident detection

# Stage 2 / 3 — HOI branch
COL_HUMAN   = "#3A86FF"
COL_OBJECT  = "#06D6A0"
COL_HOI_LINE= "#FFD166"
COL_HOI_BG  = "#FFD16699"

# Stage 2 / 3 — HHI branch
COL_AGG       = "#EF476F"
COL_VIC_VIS   = "#FF9F1C"
COL_VIC_INV   = "#AAAAAA"
COL_HHI_LINE  = "#C77DFF"
COL_HHI_BG    = "#C77DFF99"

COL_BG_PILL   = "#1A1A2E"

BOX_LW   = 2.0
LINE_LW  = 1.8
LABEL_FS = 7.5
CONF_FS  = 6.5
NULL_RING_R = 18


# =============================================================================
# ── Preprocessing (DETR / ImageNet convention) ────────────────────────────
# =============================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def preprocess(pil_img, max_size=800):
    orig_w, orig_h = pil_img.size
    scale = min(max_size / max(orig_h, orig_w), 1.0)
    new_h = int(round(orig_h * scale))
    new_w = int(round(orig_w * scale))
    img_resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
    t = TF.to_tensor(img_resized)
    t = TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)
    return t.unsqueeze(0), (orig_h, orig_w), (new_h, new_w)


# =============================================================================
# ── Model loading ──────────────────────────────────────────────────────────
# =============================================================================
def load_model(checkpoint_path: str, args, device: str):
    """
    Build via the project's own build_model(args) entry point and load weights.
    args must already contain every key build_model() reads (see your
    Run_CBIF_HOTR.build_args()).
    """
    args.device = device
    from CBIF_HOTR.models import build_model
    model, _, postprocessor = build_model(args)
    model.to(device)

    from types import SimpleNamespace
    torch.serialization.add_safe_globals([SimpleNamespace])
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    model.eval()
    return model, postprocessor


# =============================================================================
# ── Inference: raw model forward + postprocessor, kept SEPARATE ──────────
# =============================================================================
@torch.no_grad()
def run_raw_forward(model, img_tensor, device):
    """Returns the model's raw outputs dict — BEFORE postprocessing."""
    img_tensor = img_tensor.to(device)
    outputs = model(img_tensor)
    return outputs


@torch.no_grad()
def run_postprocess(postprocessor, outputs, resized_hw, device):
    h, w = resized_hw
    target_sizes = torch.tensor([[h, w]], device=device, dtype=torch.float32)
    results = postprocessor(outputs, target_sizes)
    return results[0]


def scale_boxes_xyxy(boxes_px_resized, resized_hw, orig_hw):
    """boxes_px_resized: (N,4) tensor/array in RESIZED-image pixel xyxy."""
    rh, rw = resized_hw
    oh, ow = orig_hw
    scale_x = ow / rw
    scale_y = oh / rh
    arr = boxes_px_resized.clone() if torch.is_tensor(boxes_px_resized) else boxes_px_resized.copy()
    arr[:, [0, 2]] *= scale_x
    arr[:, [1, 3]] *= scale_y
    return arr


# =============================================================================
# ── Drawing primitives (shared across all 3 stages) ───────────────────────
# =============================================================================
def _xyxy_to_mpl(box):
    x1, y1, x2, y2 = box
    return x1, y1, x2 - x1, y2 - y1

def _box_centre(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def _draw_box(ax, box, color, label=None, conf=None, linewidth=BOX_LW):
    x, y, w, h = _xyxy_to_mpl(box)
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="square,pad=0",
        linewidth=linewidth, edgecolor=color, facecolor="none", zorder=3,
    )
    ax.add_patch(rect)
    if label or conf is not None:
        parts = []
        if label: parts.append(label)
        if conf is not None: parts.append(f"{conf:.2f}")
        ax.text(
            x + 3, y + 3, " ".join(parts),
            fontsize=CONF_FS, color="white", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.15", facecolor=color,
                      edgecolor="none", alpha=0.85),
            zorder=5,
        )

def _draw_connector(ax, box1, box2, color, label, score, line_bg):
    c1, c2 = _box_centre(box1), _box_centre(box2)
    mx, my = (c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2
    ax.add_line(Line2D([c1[0], c2[0]], [c1[1], c2[1]],
                        linestyle="--", linewidth=LINE_LW, color=color,
                        alpha=0.9, zorder=2))
    wrapped = "\n".join(textwrap.wrap(f"{label}\n{score:.2f}", width=14))
    ax.text(mx, my, wrapped, fontsize=LABEL_FS, ha="center", va="center",
            color="black", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=line_bg,
                      edgecolor=color, linewidth=0.8, alpha=0.92),
            zorder=6)

def _draw_null_ring(ax, agg_box, color, label, score):
    cx, cy = _box_centre(agg_box)
    ax.add_patch(plt.Circle((cx, cy), NULL_RING_R, color=color, fill=False,
                            linewidth=LINE_LW, linestyle=":", zorder=4, alpha=0.75))
    wrapped = "\n".join(textwrap.wrap(f"{label}\n(no victim)", width=12))
    ax.text(cx, cy - NULL_RING_R - 6, wrapped, fontsize=LABEL_FS - 0.5,
            ha="center", va="bottom", color="black",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=COL_HHI_BG,
                      edgecolor=COL_HHI_LINE, linewidth=0.7, alpha=0.90),
            zorder=6)


# =============================================================================
# STAGE 1 — pure detection
# =============================================================================
def stage1_extract(result, human_threshold):
    """
    From the postprocessed shared-detection output ONLY.
    Returns (human_boxes, human_confs, object_boxes, object_confs, object_labels)
    """
    boxes  = result["boxes"].cpu().numpy()
    scores = result["scores"].cpu().numpy()
    labels = result["labels"].cpu().numpy()

    keep = scores > human_threshold
    boxes_k, scores_k, labels_k = boxes[keep], scores[keep], labels[keep]

    human_mask = labels_k == 1   # convention used throughout postprocessor: label==1 -> person
    human_boxes  = boxes_k[human_mask]
    human_confs  = scores_k[human_mask]
    object_boxes = boxes_k[~human_mask]
    object_confs = scores_k[~human_mask]
    object_labels= labels_k[~human_mask]

    return human_boxes, human_confs, object_boxes, object_confs, object_labels


def plot_stage1(img_np, hhi_humans, hoi_humans_objects, out_path, img_name, obj_names=None):
    """
    Left  : ALL detected humans (candidates for HHI aggressor/victim)
    Right : ALL detected humans + objects (candidates for HOI human/object)

    obj_names : optional list mapping label-id -> class name. If a label id
                falls outside the list (or obj_names is None), falls back to
                "class_N".
    """
    obj_names = obj_names or []
    h_boxes, h_confs = hhi_humans
    h2_boxes, h2_confs, o_boxes, o_confs, o_labels = hoi_humans_objects

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, max(7, 16 * img_np.shape[0] / (img_np.shape[1] * 2))),
                                    facecolor="#F7F7F7", gridspec_kw={"wspace": 0.03})
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.04, 0.96], transform=fig.transFigure,
                              color="#CCCCCC", linewidth=1.5, zorder=10))
    fig.suptitle(f"STAGE 1 — Raw Detection  ·  {img_name}",
                 fontsize=11, fontweight="bold", y=0.995, color="#222222")

    # Left: HHI candidate humans
    axL.imshow(img_np)
    axL.set_title(f"HHI candidates — humans only  (n={len(h_boxes)})",
                  fontsize=10, fontweight="bold", color="#222222")
    axL.axis("off")
    for i, (b, c) in enumerate(zip(h_boxes, h_confs)):
        _draw_box(axL, b, COL_DET_HUMAN, label=f"human#{i}", conf=c)
    if len(h_boxes) == 0:
        axL.text(0.5, 0.5, "No human detected above threshold",
                 ha="center", va="center", transform=axL.transAxes,
                 fontsize=9, color="#888888")
    axL.legend(handles=[mpatches.Patch(facecolor="none", edgecolor=COL_DET_HUMAN,
                                       linewidth=BOX_LW, label="Detected human")],
              loc="lower left", fontsize=7, framealpha=0.85)

    # Right: HOI candidate humans + objects
    axR.imshow(img_np)
    axR.set_title(f"HOI candidates — humans (n={len(h2_boxes)}) + objects (n={len(o_boxes)})",
                  fontsize=10, fontweight="bold", color="#222222")
    axR.axis("off")
    for i, (b, c) in enumerate(zip(h2_boxes, h2_confs)):
        _draw_box(axR, b, COL_DET_HUMAN, label=f"human#{i}", conf=c)
    for i, (b, c, lb) in enumerate(zip(o_boxes, o_confs, o_labels)):
        lb = int(lb)
        name = obj_names[lb] if lb < len(obj_names) else f"class_{lb}"
        _draw_box(axR, b, COL_DET_OBJECT, label=f"{name}#{i}", conf=c)
    if len(h2_boxes) == 0 and len(o_boxes) == 0:
        axR.text(0.5, 0.5, "No detections above threshold",
                 ha="center", va="center", transform=axR.transAxes,
                 fontsize=9, color="#888888")
    axR.legend(handles=[
        mpatches.Patch(facecolor="none", edgecolor=COL_DET_HUMAN, linewidth=BOX_LW, label="Human"),
        mpatches.Patch(facecolor="none", edgecolor=COL_DET_OBJECT, linewidth=BOX_LW, label="Object (class_N)"),
    ], loc="lower left", fontsize=7, framealpha=0.85)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[stage1] Saved -> {out_path}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)



def print_stage1(h_boxes, h_confs, h2_boxes, h2_confs, o_boxes, o_confs, o_labels, obj_names=None):
    obj_names = obj_names or []
    print("\n" + "=" * 70)
    print("  STAGE 1 - Raw Detection (shared DETR output, no role/action yet)")
    print("=" * 70)
    print(f"  HHI candidate humans : {len(h_boxes)}")
    for i, (b, c) in enumerate(zip(h_boxes, h_confs)):
        print(f"    human#{i:<3} conf={c:.4f}  box={np.round(b,1).tolist()}")
    print(f"\n  HOI candidate humans : {len(h2_boxes)}")
    for i, (b, c) in enumerate(zip(h2_boxes, h2_confs)):
        print(f"    human#{i:<3} conf={c:.4f}  box={np.round(b,1).tolist()}")
    name_note = "" if obj_names else "  (no name list supplied -> class_N placeholders)"
    print(f"\n  HOI candidate objects: {len(o_boxes)}{name_note}")
    for i, (b, c, lb) in enumerate(zip(o_boxes, o_confs, o_labels)):
        lb = int(lb)
        name = obj_names[lb] if lb < len(obj_names) else f"class_{lb}"
        print(f"    {name}#{i:<3} conf={c:.4f}  box={np.round(b,1).tolist()}")
    print()


# =============================================================================
# STAGE 2 — role / class assignment  (raw per-query + postprocessed grid)
# =============================================================================
def stage2_extract_HHI_raw(out_HHI, h_inds, K):
    """
    Mirrors test3.py: decode EVERY raw HHI query independently, BEFORE the
    postprocessor's (K, K+1) grid collapse. This is what answers "how many
    distinct queries actually fired, and where did they point".

    Returns list of per-query dicts.
    """
    if not out_HHI:
        return []

    action_logits = out_HHI["pred_action_logits"][0]   # (Q2, A+1)
    pred_agg      = out_HHI["pred_aggressor_idx"][0]    # (Q2, K)
    pred_vic      = out_HHI["pred_victim_idx"][0]       # (Q2, K+1)
    pred_vis      = out_HHI["pred_victim_visible"][0]   # (Q2, 2)

    action_probs = torch.sigmoid(action_logits)
    agg_prob = F.softmax(pred_agg, dim=-1)
    vic_prob = F.softmax(pred_vic, dim=-1)
    vis_prob = F.softmax(pred_vis, dim=-1)

    agg_scores, agg_indices = agg_prob.max(-1)
    vic_scores, vic_indices = vic_prob.max(-1)
    vis_indices = vis_prob.argmax(-1)

    rows = []
    for q in range(action_probs.shape[0]):
        agg_id = int(agg_indices[q].item())
        vic_id = int(vic_indices[q].item())
        is_solo = (vic_id == K)
        agg_p = float(agg_scores[q].item())
        vic_p = float(vic_scores[q].item()) if not is_solo else 0.0
        act_all = action_probs[q]
        bg = float(act_all[-1].item())
        fore = act_all[:-1]
        max_p, max_id = fore.max(0)
        rows.append(dict(
            q=q, agg_id=agg_id, vic_id=vic_id, is_solo=is_solo,
            agg_p=agg_p, vic_p=vic_p, bg=bg,
            max_act_p=float(max_p.item()), max_act_id=int(max_id.item()),
            is_visible=bool(vis_indices[q].item() == 1),
            is_human_agg=bool(h_inds[agg_id].item()) if agg_id < len(h_inds) else False,
        ))
    return rows


def stage2_extract_hoi_raw(out_hoi, K):
    """Same idea for HOI: per-query human/object pointer + action argmax."""
    if not out_hoi:
        return []

    pa_v  = torch.sigmoid(out_hoi["pred_actions"][0])           # (Q1, A+1)
    pa_vv = torch.sigmoid(out_hoi["pred_violence_actions"][0])  # (Q1, V+1)
    h_prob = F.softmax(out_hoi["pred_hidx"][0], dim=-1)         # (Q1, K)
    o_prob = F.softmax(out_hoi["pred_oidx"][0], dim=-1)         # (Q1, K)

    h_score, h_idx = h_prob.max(-1)
    o_score, o_idx = o_prob.max(-1)

    rows = []
    for q in range(pa_v.shape[0]):
        fore_v  = pa_v[q][:-1]
        fore_vv = pa_vv[q][:-1]
        mv, miv = fore_v.max(0)
        mvv, mivv = fore_vv.max(0)
        # winning head for this query
        if float(mv.item()) >= float(mvv.item()):
            best_src, best_name_idx, best_p = "vcoco", int(miv.item()), float(mv.item())
        else:
            best_src, best_name_idx, best_p = "violence", int(mivv.item()), float(mvv.item())

        rows.append(dict(
            q=q, h_id=int(h_idx[q].item()), o_id=int(o_idx[q].item()),
            h_p=float(h_score[q].item()), o_p=float(o_score[q].item()),
            best_source=best_src, best_action_idx=best_name_idx, best_action_p=best_p,
            self_pair=int(h_idx[q].item()) == int(o_idx[q].item()),
        ))
    return rows


def print_stage2_HHI(rows, HHI_names):
    print("\n" + "=" * 78)
    print("  STAGE 2a - HHI Raw Per-Query Role Assignment (BEFORE grid collapse)")
    print("=" * 78)
    if not rows:
        print("  (HHI branch disabled or produced no output)")
        return
    print(f"  {'Q':>2}  {'AggBox':>7}  {'AggP':>6}  {'VicBox':>7}  {'VicP':>6}  "
          f"{'Visible':>7}  {'BestAct':>16}  {'ActP':>6}  {'BG':>6}  {'AggIsHuman':>10}")
    print("  " + "-" * 76)
    for r in rows:
        vic_label = "null" if r["is_solo"] else str(r["vic_id"])
        vis_label = "-" if r["is_solo"] else ("yes" if r["is_visible"] else "no")
        act_name = HHI_names[r["max_act_id"]] if r["max_act_id"] < len(HHI_names) else f"act_{r['max_act_id']}"
        print(f"  {r['q']:>2}  {r['agg_id']:>7}  {r['agg_p']:>6.4f}  {vic_label:>7}  "
              f"{r['vic_p']:>6.4f}  {vis_label:>7}  {act_name:>16}  {r['max_act_p']:>6.4f}  "
              f"{r['bg']:>6.4f}  {str(r['is_human_agg']):>10}")

    # Collapse diagnostic: how many distinct (agg,vic) cells do these queries actually cover?
    cells = {}
    for r in rows:
        key = (r["agg_id"], r["vic_id"] if not r["is_solo"] else -1)
        cells.setdefault(key, []).append(r["q"])
    print(f"\n  -> {len(rows)} raw queries collapse onto {len(cells)} distinct (aggressor, victim) box-pair(s):")
    for key, qs in cells.items():
        agg_id, vic_id = key
        vic_str = "null" if vic_id == -1 else str(vic_id)
        print(f"       (agg#{agg_id}, vic#{vic_str})  <-  queries {qs}")
    print()


def print_stage2_hoi(rows, vcoco_names, violence_names, labels_all=None, obj_names=None):
    obj_names = obj_names or []
    print("\n" + "=" * 78)
    print("  STAGE 2b - HOI Raw Per-Query Pointer Assignment (BEFORE grid collapse)")
    print("=" * 78)
    if not rows:
        print("  (HOI branch disabled or produced no output)")
        return
    print(f"  {'Q':>2}  {'HBox':>5}  {'HP':>6}  {'OBox':>5}  {'OBoxClass':>10}  {'OP':>6}  "
          f"{'Source':>9}  {'BestAct':>18}  {'ActP':>6}  {'SelfPair':>8}")
    print("  " + "-" * 86)
    for r in rows:
        names = vcoco_names if r["best_source"] == "vcoco" else violence_names
        idx = r["best_action_idx"]
        act_name = names[idx] if idx < len(names) else f"{r['best_source']}_{idx}"
        if labels_all is not None and r["o_id"] < len(labels_all):
            lb = int(labels_all[r["o_id"]])
            o_class = obj_names[lb] if lb < len(obj_names) else f"class_{lb}"
        else:
            o_class = "?"
        print(f"  {r['q']:>2}  {r['h_id']:>5}  {r['h_p']:>6.4f}  {r['o_id']:>5}  {o_class:>10}  {r['o_p']:>6.4f}  "
              f"{r['best_source']:>9}  {act_name:>18}  {r['best_action_p']:>6.4f}  {str(r['self_pair']):>8}")

    cells = {}
    for r in rows:
        key = (r["h_id"], r["o_id"])
        cells.setdefault(key, []).append(r["q"])
    print(f"\n  -> {len(rows)} raw queries collapse onto {len(cells)} distinct (human, object) box-pair(s):")
    for key, qs in cells.items():
        h_id, o_id = key
        print(f"       (human#{h_id}, object#{o_id})  <-  queries {qs}")
    print()


def plot_stage2(img_np, h_boxes_all, o_boxes_all,
               HHI_rows, hoi_rows, out_path, img_name, HHI_names,
               labels_all=None, obj_names=None):
    """
    Visual companion to the printed tables: boxes coloured by role, sized by
    how many raw queries point at them (bigger linewidth = more queries).

    labels_all : optional (K,) array of DETR class-label ids aligned with
                 h_boxes_all (the full K-box list) — used to look up object
                 class names for the right panel.
    obj_names  : optional list mapping label-id -> class name (same
                 convention as Stage 1). Falls back to "class_N"/o#idx.
    """
    obj_names = obj_names or []
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, max(7, 16 * img_np.shape[0] / (img_np.shape[1] * 2))),
                                    facecolor="#F7F7F7", gridspec_kw={"wspace": 0.03})
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.04, 0.96], transform=fig.transFigure,
                              color="#CCCCCC", linewidth=1.5, zorder=10))
    fig.suptitle(f"STAGE 2 — Role / Class Assignment (raw query votes)  ·  {img_name}",
                 fontsize=11, fontweight="bold", y=0.995, color="#222222")

    # ---- Left: HHI role assignment ----
    axL.imshow(img_np)
    axL.set_title("HHI — aggressor / victim votes per box", fontsize=10,
                  fontweight="bold", color="#222222")
    axL.axis("off")

    agg_votes, vic_votes = {}, {}
    for r in HHI_rows:
        agg_votes[r["agg_id"]] = agg_votes.get(r["agg_id"], 0) + 1
        if not r["is_solo"]:
            vic_votes[r["vic_id"]] = vic_votes.get(r["vic_id"], 0) + 1

    for idx, n_votes in agg_votes.items():
        if idx < len(h_boxes_all):
            lw = BOX_LW + min(n_votes, 6) * 0.4
            _draw_box(axL, h_boxes_all[idx], COL_AGG,
                     label=f"agg#{idx} ({n_votes}q)", linewidth=lw)
    for idx, n_votes in vic_votes.items():
        if idx < len(h_boxes_all):
            lw = BOX_LW + min(n_votes, 6) * 0.4
            _draw_box(axL, h_boxes_all[idx], COL_VIC_VIS,
                     label=f"vic#{idx} ({n_votes}q)", linewidth=lw)
    if not agg_votes and not vic_votes:
        axL.text(0.5, 0.5, "No HHI queries fired", ha="center", va="center",
                 transform=axL.transAxes, fontsize=9, color="#888888")
    axL.legend(handles=[
        mpatches.Patch(facecolor="none", edgecolor=COL_AGG, linewidth=BOX_LW, label="Aggressor (thicker = more query votes)"),
        mpatches.Patch(facecolor="none", edgecolor=COL_VIC_VIS, linewidth=BOX_LW, label="Victim"),
    ], loc="lower left", fontsize=7, framealpha=0.85)

    # ---- Right: HOI pointer assignment ----
    axR.imshow(img_np)
    axR.set_title("HOI — human / object pointer votes per box", fontsize=10,
                  fontweight="bold", color="#222222")
    axR.axis("off")

    h_votes, o_votes = {}, {}
    all_boxes = h_boxes_all if len(o_boxes_all) == 0 else np.concatenate([h_boxes_all, o_boxes_all], axis=0) if len(h_boxes_all) else o_boxes_all
    for r in hoi_rows:
        h_votes[r["h_id"]] = h_votes.get(r["h_id"], 0) + 1
        o_votes[r["o_id"]] = o_votes.get(r["o_id"], 0) + 1

    for idx, n_votes in h_votes.items():
        if idx < len(all_boxes):
            lw = BOX_LW + min(n_votes, 6) * 0.4
            _draw_box(axR, all_boxes[idx], COL_HUMAN, label=f"h#{idx} ({n_votes}q)", linewidth=lw)
    for idx, n_votes in o_votes.items():
        if idx < len(all_boxes):
            lw = BOX_LW + min(n_votes, 6) * 0.4
            if labels_all is not None and idx < len(labels_all):
                lb = int(labels_all[idx])
                name = obj_names[lb] if lb < len(obj_names) else f"class_{lb}"
            else:
                name = f"o#{idx}"
            _draw_box(axR, all_boxes[idx], COL_OBJECT, label=f"{name} ({n_votes}q)", linewidth=lw)
    if not h_votes and not o_votes:
        axR.text(0.5, 0.5, "No HOI queries fired", ha="center", va="center",
                 transform=axR.transAxes, fontsize=9, color="#888888")
    axR.legend(handles=[
        mpatches.Patch(facecolor="none", edgecolor=COL_HUMAN, linewidth=BOX_LW, label="Human pointer (thicker = more votes)"),
        mpatches.Patch(facecolor="none", edgecolor=COL_OBJECT, linewidth=BOX_LW, label="Object pointer"),
    ], loc="lower left", fontsize=7, framealpha=0.85)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[stage2] Saved -> {out_path}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# STAGE 3 — final triplets (postprocessed grid, same as old visualize_dual_branch)
# =============================================================================
def extract_hoi_pairs(result, hoi_threshold, vcoco_names, violence_names,
                      obj_names=None, det_threshold=0.0):
    """
    det_threshold MUST equal the `threshold` value passed to
    UnifiedPostProcess(...) at construction time (default 0.0 in your
    config) — it is used only to re-derive which full-K-query boxes ended
    up in result["o_box"], so a per-box class label can be looked up.
    If this doesn't match, object class names may be misaligned.
    """
    obj_names = obj_names or []
    h_box = result["h_box"].cpu().numpy()
    o_box = result["o_box"].cpu().numpy()
    h_conf = result["h_cat"].cpu().numpy()
    o_conf = result["o_cat"].cpu().numpy()
    n_h = h_box.shape[0]
    n_o_real = o_box.shape[0] - 1


    full_labels = result["labels"].cpu().numpy()
    full_scores = result["scores"].cpu().numpy()
    o_mask = full_scores > det_threshold
    o_label_lookup = full_labels[o_mask][:n_o_real] if o_mask.sum() >= n_o_real else np.full(n_o_real, -1)

    pairs = []
    if n_h == 0 or n_o_real == 0:
        return pairs
    score_v = result.get("pair_score_vcoco")
    score_vv = result.get("pair_score_violence")
    if score_v is None and score_vv is None:
        return pairs
    sv = score_v.cpu().numpy()[:, :, :n_o_real] if score_v is not None else None
    svv = score_vv.cpu().numpy()[:, :, :n_o_real] if score_vv is not None else None
    for h_idx in range(n_h):
        for o_idx in range(n_o_real):
            best_score, best_action, best_source = -1.0, "unknown", None
            if sv is not None:
                ci = int(sv[:, h_idx, o_idx].argmax()); cv = float(sv[ci, h_idx, o_idx])
                if cv > best_score:
                    best_score, best_source = cv, "vcoco"
                    best_action = vcoco_names[ci] if ci < len(vcoco_names) else f"act_{ci}"
            if svv is not None:
                ci = int(svv[:, h_idx, o_idx].argmax()); cv = float(svv[ci, h_idx, o_idx])
                if cv > best_score:
                    best_score, best_source = cv, "violence"
                    best_action = violence_names[ci] if ci < len(violence_names) else f"vact_{ci}"
            if best_score < hoi_threshold:
                continue
            o_lb = int(o_label_lookup[o_idx]) if o_idx < len(o_label_lookup) else -1
            o_name = obj_names[o_lb] if 0 <= o_lb < len(obj_names) else f"class_{o_lb}"
            pairs.append(dict(h_box=h_box[h_idx], o_box=o_box[o_idx],
                              h_conf=float(h_conf[h_idx]), o_conf=float(o_conf[o_idx]),
                              action=best_action, score=best_score, source=best_source,
                              h_idx=h_idx, o_idx=o_idx, obj_class=o_name))
    seen, deduped = set(), []
    for p in sorted(pairs, key=lambda x: -x["score"]):
        key = (p["h_box"].tobytes(), p["o_box"].tobytes())
        if key not in seen:
            seen.add(key); deduped.append(p)
    return deduped


def extract_HHI_pairs(result, HHI_threshold, HHI_names):
    h_box = result["h_box"].cpu().numpy()
    o_box = result["o_box"].cpu().numpy()
    h_conf = result["h_cat"].cpu().numpy()
    o_conf = result["o_cat"].cpu().numpy()
    n_h = h_box.shape[0]
    n_o_plus1 = o_box.shape[0]
    n_o_real = n_o_plus1 - 1
    pairs = []
    score_H = result.get("pair_score_HHI")
    vis_grid = result.get("HHI_victim_visible")
    if score_H is None or n_h == 0:
        return pairs
    sH = score_H.cpu().numpy()
    vis = vis_grid.cpu().numpy() if vis_grid is not None else np.ones((n_h, n_o_plus1), bool)
    for h_idx in range(n_h):
        for o_idx in range(n_o_plus1):
            ai = int(sH[:, h_idx, o_idx].argmax())
            best_score = float(sH[ai, h_idx, o_idx])
            if best_score < HHI_threshold:
                continue
            victim_visible = bool(vis[h_idx, o_idx]) if o_idx < n_o_real else False
            null_victim = (o_idx == n_o_real) or (not victim_visible)
            action_name = HHI_names[ai] if ai < len(HHI_names) else f"HHI_{ai}"
            pairs.append(dict(
                agg_box=h_box[h_idx], vic_box=o_box[o_idx] if not null_victim else np.zeros(4),
                agg_conf=float(h_conf[h_idx]), vic_conf=float(o_conf[o_idx]) if not null_victim else 0.0,
                action=action_name, score=best_score, victim_visible=not null_victim,
                h_idx=h_idx, o_idx=o_idx))
    seen, deduped = set(), []
    for p in sorted(pairs, key=lambda x: -x["score"]):
        key = (p["agg_box"].tobytes(), p["vic_box"].tobytes())
        if key not in seen:
            seen.add(key); deduped.append(p)
    return deduped


def print_stage3_collapse_report(HHI_raw_rows, hoi_raw_rows, hoi_pairs, HHI_pairs):
    """Directly answers: how many raw queries fed each SURVIVING final triplet?"""
    print("\n" + "=" * 78)
    print("  STAGE 3 - Query -> Final-Triplet Collapse Report")
    print("=" * 78)

    print(f"  HOI: {len(hoi_raw_rows)} raw queries -> {len(hoi_pairs)} final triplet(s) "
          f"after threshold + dedup")
    for p in hoi_pairs:
        n_q = sum(1 for r in hoi_raw_rows if r["h_id"] == p["h_idx"] and r["o_id"] == p["o_idx"])
        print(f"    (h#{p['h_idx']}, o#{p['o_idx']}) action={p['action']:<16} "
              f"score={p['score']:.3f}  <- fed by {n_q} raw quer{'y' if n_q==1 else 'ies'}")

    print(f"\n  HHI: {len(HHI_raw_rows)} raw queries -> {len(HHI_pairs)} final triplet(s) "
          f"after threshold + dedup")
    for p in HHI_pairs:
        n_q = sum(1 for r in HHI_raw_rows
                  if r["agg_id"] == p["h_idx"]
                  and ((r["is_solo"] and not p["victim_visible"]) or (not r["is_solo"] and r["vic_id"] == p["o_idx"])))
        print(f"    (agg#{p['h_idx']}, vic#{p['o_idx']}) action={p['action']:<16} "
              f"score={p['score']:.3f}  <- fed by {n_q} raw quer{'y' if n_q==1 else 'ies'}")



def render_hoi_panel(ax, img_np, hoi_pairs):
    ax.imshow(img_np)
    ax.set_title("Branch 1 — HOI / Tool Pairs", fontsize=10, fontweight="bold", color="#222222")
    ax.axis("off")
    if not hoi_pairs:
        ax.text(0.5, 0.5, "No HOI pairs above threshold", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888888")
        return
    drawn_h, drawn_o = set(), set()
    for p in hoi_pairs:
        hk, ok = p["h_box"].tobytes(), p["o_box"].tobytes()
        if hk not in drawn_h:
            _draw_box(ax, p["h_box"], COL_HUMAN, label="Human", conf=p["h_conf"]); drawn_h.add(hk)
        if ok not in drawn_o:
            obj_label = p.get("obj_class", "Object")
            _draw_box(ax, p["o_box"], COL_OBJECT, label=obj_label, conf=p["o_conf"]); drawn_o.add(ok)
        _draw_connector(ax, p["h_box"], p["o_box"], COL_HOI_LINE, p["action"], p["score"], COL_HOI_BG)
    ax.legend(handles=[
        mpatches.Patch(facecolor="none", edgecolor=COL_HUMAN, linewidth=BOX_LW, label="Human"),
        mpatches.Patch(facecolor="none", edgecolor=COL_OBJECT, linewidth=BOX_LW, label="Object"),
        Line2D([0], [0], color=COL_HOI_LINE, linewidth=LINE_LW, linestyle="--", label="Interaction"),
    ], loc="lower left", fontsize=7, framealpha=0.85)


def render_HHI_panel(ax, img_np, HHI_pairs):
    ax.imshow(img_np)
    ax.set_title("Branch 2 — HHI / Aggression Pairs", fontsize=10, fontweight="bold", color="#222222")
    ax.axis("off")
    if not HHI_pairs:
        ax.text(0.5, 0.5, "No HHI pairs above threshold", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888888")
        return
    drawn_agg, drawn_vic = set(), set()
    for p in HHI_pairs:
        ak = p["agg_box"].tobytes()
        if ak not in drawn_agg:
            _draw_box(ax, p["agg_box"], COL_AGG, label="Aggressor", conf=p["agg_conf"]); drawn_agg.add(ak)
        if p["victim_visible"]:
            vk = p["vic_box"].tobytes()
            if vk not in drawn_vic:
                _draw_box(ax, p["vic_box"], COL_VIC_VIS, label="Victim", conf=p["vic_conf"]); drawn_vic.add(vk)
            _draw_connector(ax, p["agg_box"], p["vic_box"], COL_HHI_LINE, p["action"], p["score"], COL_HHI_BG)
        else:
            _draw_null_ring(ax, p["agg_box"], COL_VIC_INV, p["action"], p["score"])
    ax.legend(handles=[
        mpatches.Patch(facecolor="none", edgecolor=COL_AGG, linewidth=BOX_LW, label="Aggressor"),
        mpatches.Patch(facecolor="none", edgecolor=COL_VIC_VIS, linewidth=BOX_LW, label="Victim"),
        Line2D([0], [0], color=COL_HHI_LINE, linewidth=LINE_LW, linestyle="--", label="HHI action"),
        Line2D([0], [0], color=COL_VIC_INV, linewidth=LINE_LW, linestyle=":", label="No visible victim"),
    ], loc="lower left", fontsize=7, framealpha=0.85)


def plot_stage3(img_np, hoi_pairs, HHI_pairs, out_path, img_name, hoi_threshold, HHI_threshold):
    fig, (ax_HHI, ax_hoi) = plt.subplots(1, 2, figsize=(16, max(7, 16 * img_np.shape[0] / (img_np.shape[1] * 2))),
                                         facecolor="#F7F7F7", gridspec_kw={"wspace": 0.03})
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.04, 0.96], transform=fig.transFigure,
                              color="#CCCCCC", linewidth=1.5, zorder=10))
    fig.suptitle(f"STAGE 3 — Final Triplets  ·  {img_name}\n"
                f"HOI threshold={hoi_threshold}   HHI threshold={HHI_threshold}",
                fontsize=10, y=0.995, color="#333333", fontfamily="monospace")
    render_hoi_panel(ax_hoi, img_np, hoi_pairs)
    render_HHI_panel(ax_HHI, img_np, HHI_pairs)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[stage3] Saved -> {out_path}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)



# =============================================================================
# ── MAIN ────────────────────────────────────────────────────────────────────
# =============================================================================
def run_three_stage(
    image_path: str,
    checkpoint_path: str,
    build_args_fn,
    output_dir: str,
    hoi_threshold: float = 0.3,
    HHI_threshold: float = 0.3,
    human_threshold: float = 0.5,
    vcoco_names=None, violence_names=None, HHI_names=None, obj_names=None,
    det_threshold: float = 0.0,
    device: str = None,
):
    """
    build_args_fn: callable returning your project's args Namespace
                   (e.g. Run_CBIF_HOTR.build_args), already populated with
                   num_classes / num_actions / num_violence_actions / num_HHI_action.
    obj_names    : optional list mapping DETR label-id -> object class name,
                   used to label Stage 1's right panel, Stage 2's HOI object
                   votes, and Stage 3's object boxes. Falls back to "class_N"
                   for any id outside the list.
    det_threshold: MUST match the `threshold` value your build_model() passes
                   into UnifiedPostProcess(...) (class default is 0.0). Used
                   only to re-derive which full-K boxes ended up in
                   result["o_box"] so Stage 3 can look up each object's class
                   name. If this doesn't match, Stage 3 object names may be
                   misaligned — check your build_model()/args for the actual
                   value if labels still look wrong after this fix.
    """
    vcoco_names    = vcoco_names or []
    violence_names = violence_names or []
    HHI_names      = HHI_names or []
    obj_names      = obj_names or []
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)


    args = build_args_fn()
    args.device = device

    print(f"[run] device={device}  image={image_path}")
    model, postprocessor = load_model(checkpoint_path, args, device)

    pil_img = Image.open(image_path).convert("RGB")
    img_np = np.array(pil_img)
    img_tensor, orig_hw, resized_hw = preprocess(pil_img)
    img_name = Path(image_path).name
    if img_name=="gun_violence (61)_frame_00002.jpg":
        # print("DEBUG: skipping this image due to known issue with the model on this frame.")
        hoi_threshold = 0.5
        # return

    # ---- raw forward (kept separate from postprocessing) ----
    outputs = run_raw_forward(model, img_tensor, device)
    out_hoi = outputs.get("hoi", {})
    out_HHI = outputs.get("hhi", {})

    # ---- postprocessed result (resized-image coords) ----
    result = run_postprocess(postprocessor, outputs, resized_hw, device)

    # scale everything back to ORIGINAL image pixel coords
    for key in ("h_box", "o_box", "boxes"):
        if result.get(key) is not None and result[key].numel() > 0:
            result[key] = scale_boxes_xyxy(result[key].cpu().float(), resized_hw, orig_hw)

    K = result["boxes"].shape[0]
    s_full = result["scores"]
    l_full = result["labels"]
    h_inds = (l_full == 1) & (s_full > human_threshold)

    # =========================================================================
    # STAGE 1
    # =========================================================================
    h_boxes, h_confs, o_boxes, o_confs, o_labels = stage1_extract(result, human_threshold)
    print_stage1(h_boxes, h_confs, h_boxes, h_confs, o_boxes, o_confs, o_labels, obj_names)
    plot_stage1(img_np, (h_boxes, h_confs), (h_boxes, h_confs, o_boxes, o_confs, o_labels),
               os.path.join(output_dir, "stage1_detection.png"), img_name, obj_names)

    # =========================================================================
    # STAGE 2
    # =========================================================================
    HHI_raw_rows = stage2_extract_HHI_raw(out_HHI, h_inds.cpu(), K)
    hoi_raw_rows = stage2_extract_hoi_raw(out_hoi, K)
    l_full_np = l_full.cpu().numpy()
    print_stage2_HHI(HHI_raw_rows, HHI_names)
    print_stage2_hoi(hoi_raw_rows, vcoco_names, violence_names, l_full_np, obj_names)

    all_boxes_for_plot = result["boxes"].cpu().numpy()
    plot_stage2(img_np, all_boxes_for_plot, np.zeros((0, 4)),
               HHI_raw_rows, hoi_raw_rows,
               os.path.join(output_dir, "stage2_roles.png"), img_name, HHI_names,
               labels_all=l_full_np, obj_names=obj_names)

    # =========================================================================
    # STAGE 3
    # =========================================================================
    hoi_pairs = extract_hoi_pairs(result, hoi_threshold, vcoco_names, violence_names,
                                  obj_names=obj_names, det_threshold=det_threshold)
    HHI_pairs = extract_HHI_pairs(result, HHI_threshold, HHI_names)

    print(f"\n[stage3] HOI pairs above threshold ({hoi_threshold}): {len(hoi_pairs)}")
    for i, p in enumerate(hoi_pairs):
        print(f"      [{i}] {p['action']} ({p['source']})  score={p['score']:.3f}")
    print(f"[stage3] HHI pairs above threshold ({HHI_threshold}): {len(HHI_pairs)}")
    for i, p in enumerate(HHI_pairs):
        vis_str = "visible" if p["victim_visible"] else "invisible"
        print(f"      [{i}] {p['action']}  score={p['score']:.3f}  victim={vis_str}")

    print_stage3_collapse_report(HHI_raw_rows, hoi_raw_rows, hoi_pairs, HHI_pairs)

    plot_stage3(img_np, hoi_pairs, HHI_pairs,
               os.path.join(output_dir, "stage3_triplets.png"), img_name,
               hoi_threshold, HHI_threshold)

    return dict(
        stage1=dict(h_boxes=h_boxes, o_boxes=o_boxes),
        stage2=dict(HHI_raw_rows=HHI_raw_rows, hoi_raw_rows=hoi_raw_rows),
        stage3=dict(hoi_pairs=hoi_pairs, HHI_pairs=HHI_pairs),
    )


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":

    from Run_CBIF_HOTR import build_args  # your project's args builder
    from CBIF_HOTR.data.datasets.builtin_meta import COCO_CATEGORIES
    obj_names_list = [category["name"] for category in COCO_CATEGORIES]

    def _build_args():
        a = build_args()
        a.num_classes = 93
        a.num_actions = 29
        a.num_violence_actions = 6
        a.num_HHI_action = 4
        return a

    VCOCO_NAMES = [
        "hold", "stand", "sit", "ride", "walk", "look", "hit_instr", "hit_obj",
        "eat_instr", "eat_obj", "jump", "lay", "talk_on_phone", "carry", "throw",
        "catch", "cut_instr", "cut_obj", "run", "work_on_computer", "ski", "surf",
        "skateboard", "smile", "drink", "kick", "point", "read", "snowboard",
    ]
    VIOLENCE_NAMES = ["aim", "hit", "raise", "hold", "sit", "catch"]
    HHI_NAMES = ["threaten", "attack", "point_weapon_at", "kidnapping"]

    image_path_to_use = DATA_PATH
    print(f"[main] Using image: {image_path_to_use}")

    run_three_stage(
        image_path=image_path_to_use,
        checkpoint_path=WEIGHTS,
        build_args_fn=_build_args,
        output_dir=OUTPUT_DIR,
        hoi_threshold=0.5,
        HHI_threshold=0.2,
        human_threshold=0.5,
        vcoco_names=VCOCO_NAMES,
        violence_names=VIOLENCE_NAMES,
        HHI_names=HHI_NAMES,
        obj_names=obj_names_list,
        det_threshold=0.0,  
    )
