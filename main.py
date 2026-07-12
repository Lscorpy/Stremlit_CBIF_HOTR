import string
import numpy as np

def make_person_labels(n):
    labels = []
    letters = string.ascii_uppercase
    for i in range(n):
        if i < 26:
            labels.append(letters[i])
        else:
            first = (i // 26) - 1
            second = i % 26
            labels.append(letters[first] + letters[second])
    return labels


def compute_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter_w, inter_h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter_area = inter_w * inter_h
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0


def match_person(query_box, person_registry, iou_threshold=0.3):
    if query_box is None or np.all(np.asarray(query_box) == 0):
        return None  # e.g. HHI null-victim placeholder box
    best_iou, best_label = 0.0, None
    for p in person_registry:
        iou = compute_iou(query_box, p["box"])
        if iou > best_iou:
            best_iou, best_label = iou, p["id"]
    return best_label if best_iou >= iou_threshold else None


import os
import io
import base64
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from huggingface_hub import hf_hub_download
from torch.quantization import quantize_dynamic

from CBIF_HOTR.models import build_model
from Run_CBIF_HOTR import build_args
from CBIF_HOTR.data.datasets.builtin_meta import COCO_CATEGORIES

import visualize_three_stage as vts
vts.SHOW_PLOTS = False   # critical for headless server

app = FastAPI()

DEVICE = "cpu"
MODEL = None
POSTPROCESSOR = None
OBJ_NAMES = [c["name"] for c in COCO_CATEGORIES]

VCOCO_NAMES = ["hold","stand","sit","ride","walk","look","hit_instr","hit_obj",
    "eat_instr","eat_obj","jump","lay","talk_on_phone","carry","throw",
    "catch","cut_instr","cut_obj","run","work_on_computer","ski","surf",
    "skateboard","smile","drink","kick","point","read","snowboard"]
VIOLENCE_NAMES = ["aim", "hit", "raise", "hold", "sit", "catch"]
HHI_NAMES = ["threaten", "attack", "point_weapon_at", "kidnapping"]

HOI_THRESHOLD = 0.5
HHI_THRESHOLD = 0.2
HUMAN_THRESHOLD = 0.5
OUTPUT_DIR = "/tmp/inference_out"


def build_deploy_args():
    args = build_args()
    args.num_classes = 93
    args.num_actions = 29
    args.num_violence_actions = 6
    args.num_HHI_action = 4
    args.device = DEVICE
    return args


@app.on_event("startup")
def load_model_on_startup():
    global MODEL, POSTPROCESSOR
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    weights_path = hf_hub_download(
        repo_id="Lscropy/CBIF-HOTR",
        filename="CBIF_HOTR_quantized.pth",
        token=os.environ.get("HF_TOKEN"),
    )

    args = build_deploy_args()
    model, postprocessor = build_model(args)
    model.to(DEVICE)
    model = quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    state = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    MODEL = model
    POSTPROCESSOR = postprocessor
    print("[startup] Model loaded and ready.")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(pil_img)

    img_tensor, orig_hw, resized_hw = vts.preprocess(pil_img)

    outputs = vts.run_raw_forward(MODEL, img_tensor, DEVICE)
    out_hoi = outputs.get("hoi", {})
    out_HHI = outputs.get("hhi", {})

    result = vts.run_postprocess(POSTPROCESSOR, outputs, resized_hw, DEVICE)
    for key in ("h_box", "o_box", "boxes"):
        if result.get(key) is not None and result[key].numel() > 0:
            result[key] = vts.scale_boxes_xyxy(result[key].cpu().float(), resized_hw, orig_hw)

    K = result["boxes"].shape[0]
    s_full, l_full = result["scores"], result["labels"]
    h_inds = (l_full == 1) & (s_full > HUMAN_THRESHOLD)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Stage 1: detection + person registry ----
    h_boxes, h_confs, o_boxes, o_confs, o_labels = vts.stage1_extract(result, HUMAN_THRESHOLD)

    labels = make_person_labels(len(h_boxes))
    person_registry = [
        {"id": lab, "box": box.tolist(), "conf": float(conf)}
        for lab, box, conf in zip(labels, h_boxes, h_confs)
    ]

    stage1_path = os.path.join(OUTPUT_DIR, f"stage1_{file.filename}.png")
    vts.plot_stage1(img_np, (h_boxes, h_confs), (h_boxes, h_confs, o_boxes, o_confs, o_labels),
                     stage1_path, file.filename, OBJ_NAMES)

    # ---- Stage 2: raw role decoding ----
    HHI_raw_rows = vts.stage2_extract_HHI_raw(out_HHI, h_inds.cpu(), K)
    hoi_raw_rows = vts.stage2_extract_hoi_raw(out_hoi, K)
    l_full_np = l_full.cpu().numpy()
    all_boxes_for_plot = result["boxes"].cpu().numpy()

    stage2_path = os.path.join(OUTPUT_DIR, f"stage2_{file.filename}.png")
    vts.plot_stage2(img_np, all_boxes_for_plot, np.zeros((0, 4)),
                     HHI_raw_rows, hoi_raw_rows, stage2_path, file.filename,
                     HHI_NAMES, labels_all=l_full_np, obj_names=OBJ_NAMES)

    # ---- Stage 3: final triplets ----
    hoi_pairs = vts.extract_hoi_pairs(result, HOI_THRESHOLD, VCOCO_NAMES, VIOLENCE_NAMES,
                                       obj_names=OBJ_NAMES, det_threshold=0.0)
    HHI_pairs = vts.extract_HHI_pairs(result, HHI_THRESHOLD, HHI_NAMES)

    stage3_path = os.path.join(OUTPUT_DIR, f"stage3_{file.filename}.png")
    vts.plot_stage3(img_np, hoi_pairs, HHI_pairs, stage3_path, file.filename,
                     HOI_THRESHOLD, HHI_THRESHOLD)

    # ---- Person matching -> final JSON record ----
    hoi_records = [{
        "person": match_person(p["h_box"], person_registry),
        "object": p.get("obj_class", "object"),
        "action": p["action"],
        "score": float(p["score"]),
        "source": p["source"],
    } for p in hoi_pairs]

    HHI_records = [{
        "aggressor": match_person(p["agg_box"], person_registry),
        "victim": match_person(p["vic_box"], person_registry) if p["victim_visible"] else None,
        "victim_visible": p["victim_visible"],
        "action": p["action"],
        "score": float(p["score"]),
    } for p in HHI_pairs]

    record = {
        # "persons": person_registry,
        "hoi_interactions": hoi_records,
        "hhi_interactions": HHI_records,
    }

    def to_b64(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    return JSONResponse({
        "record": record,
        "stage1_image_base64": to_b64(stage1_path),
        "stage2_image_base64": to_b64(stage2_path),
        "stage3_image_base64": to_b64(stage3_path),
    })
