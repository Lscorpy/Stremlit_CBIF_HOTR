"""
Unified target schema (per image)
----------------------------------
Shared / DETR backbone
  boxes              (n_ann, 4)  float32   all instance boxes (xyxy, unnorm)
  labels             (n_ann,)    int64     COCO category ids
  image_id           (1,)        int64
  imgae_name_for_track  str
  orig_size          (2,)        int64     [H, W] before transforms
  size               (2,)        int64     [H, W] after transforms

Branch 1 — HOI (tool triplets)
  inst_actions       (n_ann, N_HOI_ACT)  int64   per-instance vcoco+violence flags
  pair_boxes         (n_pairs, 8)        float32 [h_box | o_box]
  pair_actions       (n_pairs, 29)       int64   vcoco one-hot
  pair_violence      (n_pairs, N_V_ACT)  int64   violence one-hot
  pair_targets       (n_pairs,)          int64   object COCO class
  has_vcoco_labels   (n_pairs,)          float32

Branch 2 — HHI (aggressor/victim pairs)
  human_boxes        (n_humans, 4)  float32
  aggressor_index    (n_pairs,)     int64   index into human_boxes
  victim_index       (n_pairs,)     int64   index into human_boxes, -1=invisible
  violence_actions   (n_pairs, N_HHI_ACT)  int64  one-hot
  has_target_visible (n_pairs,)     float32
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import json

from pathlib import Path

import CBIF_HOTR.data.transforms.transforms as T
from CBIF_HOTR.data.datasets import builtin_meta


class UnifiedViolenceDetection(Dataset):
    def __init__(self, img_folder, ann_file, transforms=None):
        self.img_folder = img_folder
        self.ann_file = ann_file
        self._transforms = transforms
        
        with open(ann_file, 'r') as f:
            self.data = json.load(f)

        # Unified Verb Spaces
        self.verb_list = ["aim", "hit", "raise", "hold", "sit", "catch"] # From Source 1
        self.HHI_action_list = ["threaten", "attack", "point_weapon_at", "kidnapping"] # From Source 2

        self.num_coco_action = 29
        self.VCOCO_ACTIONS = [
            "hold", "stand", "sit", "ride", "walk", "look", "hit_instr", "hit_obj", "eat_instr", "eat_obj",
            "jump", "lay", "talk_on_phone", "carry", "throw", "catch", "cut_instr", "cut_obj", "run",
            "work_on_computer", "ski", "surf", "skateboard", "smile", "drink", "kick", "point", "read", "snowboard"
        ]
        self.COCO_CLASSES = builtin_meta._get_coco_instances_meta()['coco_classes']
        
        self.file_meta = {
            "coco_classes": self.COCO_CLASSES,
            "violence_action_classes": self.verb_list,
            "HHI_action_classes": self.HHI_action_list,
        }

        


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_info = self.data[idx]
        img_path = os.path.join(self.img_folder, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")

        # Merge results from both annotation processors
        target = self.get_ann_info(idx)

        w, h = image.size
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        if self._transforms is not None:
            image, target = self._transforms(image, target)
        
        return image, target
    
    def _process_hoi(self, img, inst_bbox, inst_label,num_ann):
        hois = img.get("hoi_annotation", [])

        inst_actions = np.zeros((num_ann, self.num_vio_action()), dtype=np.int64)

        # =========================
        # PAIR LEVEL
        # =========================
        pair_bbox = []
        pair_violence_actions = []
        pair_targets = []
        pair_actions = [] 
        has_vcoco_labels=[]

        VCOCO_MAPPING = {
            "hit": 6,     # example index in V-COCO
            "sit": 2,
            "hold": 0,
            "catch":15
        }

        for hoi in hois:

            sub = hoi["subject_id"]
            obj = hoi["object_id"]
            verb = hoi["category_id"]
            verb_name=self.verb_list[verb]

            check_exist=list(VCOCO_MAPPING.keys())
            if verb_name in check_exist:
                has_vcoco_labels.append(True)
            else:
                has_vcoco_labels.append(False)
            
            # safety check
            if sub >= num_ann or obj >= num_ann:
                print("trigger: sub >= num_ann or obj >= num_ann")
                print(img["file_name"])
                continue

            # subject must be person
            if inst_label[sub] != 1:
                print("trigger: ",inst_label[sub]," !=1")
                continue

            # ===== instance action =====
            inst_actions[sub, verb] = 1

            # ===== pair bbox =====
            h_box = inst_bbox[sub]
            o_box = inst_bbox[obj]


            pair_bbox.append(np.concatenate([h_box, o_box]))

            # ===== pair violence action (6-dim violence classification) =====
            action_vec = np.zeros((self.num_vio_action(),), dtype=np.int64)
            action_vec[verb] = 1
            pair_violence_actions.append(action_vec)

            # ===== pair target (object class) =====
            pair_targets.append(inst_label[obj])

            # ===== pair action (29-dim VCOCO classification) =====
            action_vcoco = np.zeros((self.num_coco_action,), dtype=np.int64)
            if verb_name in VCOCO_MAPPING:
                action_vcoco[VCOCO_MAPPING[verb_name]] = 1
            else:
                # Violence verb not in VCOCO → mark as "no interaction"
                action_vcoco = np.zeros((29,), dtype=np.int64)

            pair_actions.append(action_vcoco)

        
        if len(pair_bbox) == 0:
            pair_bbox = np.zeros((0, 8), dtype=np.float32)
            pair_violence_actions = np.zeros((0, self.num_vio_action()), dtype=np.int64)
            pair_actions = np.zeros((0, self.num_coco_action), dtype=np.int64)
            pair_targets = np.zeros((0,), dtype=np.int64)
            has_vcoco_labels = np.zeros((0,), dtype=np.float32)

        else:
            pair_bbox = np.array(pair_bbox, dtype=np.float32)
            pair_violence_actions = np.array(pair_violence_actions, dtype=np.int64)
            pair_actions = np.array(pair_actions, dtype=np.int64)
            pair_targets = np.array(pair_targets, dtype=np.int64)
            has_vcoco_labels = np.array(has_vcoco_labels, dtype=np.float32)

        
        # ===== VALIDATION: Ensure pair consistency =====

        assert len(pair_violence_actions) == len(pair_actions), \
            f"Mismatch: {len(pair_violence_actions)} violence pairs vs {len(pair_actions)} VCOCO pairs. This breaks the matcher!"
        
        assert len(pair_bbox) == len(pair_violence_actions), \
            f"Mismatch: {len(pair_bbox)} bounding boxes vs {len(pair_violence_actions)} violence pairs. This breaks the matcher!\n{img}"
 
        return {
            "inst_actions":inst_actions,
            "pair_bbox": pair_bbox, "pair_actions": pair_actions, 
            "pair_violence": pair_violence_actions, "pair_targets": pair_targets, 
            "has_vcoco": has_vcoco_labels
        }

    def _process_HHI(self, img, human_bbox_list, human_id_list):
        HHIs = img.get("violence_annotation", [])
        if human_bbox_list:  
            human_bbox=np.array(human_bbox_list, dtype=np.float32)
        else:
            human_bbox = np.zeros((0, 4), dtype=np.float32)

        human_num = len(human_bbox_list)

        # =========================
        # PAIR LEVEL
        # =========================
        aggre_labels=[]
        victim_labels=[]
        violence_action=[]
        has_target_visible=[]

        for HHI in HHIs:

            aggressor = HHI["subject_id"]
            victim = HHI["victim_id"]
            verb = HHI["category_id"]
            target_visible = HHI["target_visible"]


            if aggressor >= human_num or victim >= human_num:
                if aggressor in human_id_list and victim in human_id_list:
                    aggressor = human_id_list.index(aggressor)
                    victim = human_id_list.index(victim)
                elif victim==-1 and aggressor in human_id_list:
                    aggressor = human_id_list.index(aggressor)
                    pass
                else:
                    print(img["file_name"])
                    print("not found id")



            aggre_labels.append(aggressor)
            victim_labels.append(victim)

            # violence_action for the training format
            action_vec = np.zeros((self.HHI_num_action(),), dtype=np.int64)
            action_vec[verb] = 1
            violence_action.append(action_vec)

            has_target_visible.append(target_visible)
        
        if len(human_bbox) == 0:

            violence_action = np.zeros((0,), dtype=np.int64)
            aggre_labels = np.zeros((0,), dtype=np.int64)
            victim_labels = np.zeros((0,), dtype=np.int64)
            has_target_visible = np.zeros((0,), dtype=np.float32)
        else:
            violence_action= np.array(violence_action, dtype=np.int64)
            aggre_labels= np.array(aggre_labels, dtype=np.int64)
            victim_labels= np.array(victim_labels, dtype=np.int64)
            has_target_visible = np.array(has_target_visible, dtype=np.float32)

        return {
            "human_bbox": human_bbox, "agg_idx": aggre_labels, 
            "vic_idx": victim_labels, "vio_act": violence_action, 
            "vis": has_target_visible
        }
    

    def get_ann_info(self, idx):
        img = self.data[idx]
        annotations = img["annotations"]
        # Standard Instance Level Processing (Shared)
        num_ann = len(annotations)
        inst_bbox = np.zeros((num_ann, 4), dtype=np.float32)
        inst_label = np.zeros((num_ann,), dtype=np.int64)
        human_id_list=[]
        human_bbox_list=[]

        
        for i, ann in enumerate(annotations):
            x, y, w, h = ann["bbox"]
            inst_bbox[i] = [x, y, x + w, y + h]
            inst_label[i] = ann["category_id"]
            if ann["category_id"] == 1: # Assuming 1 is Person
                human_bbox_list.append([x, y, x + w, y + h])
                human_id_list.append(ann["id"])

        # Process HOI Annotations (from Source 1 logic)
        hoi_results = self._process_hoi(img,inst_bbox, inst_label,num_ann)

        # Process HHI Annotations (from Source 2 logic)
        HHI_results = self._process_HHI(img, human_bbox_list, human_id_list)

        # Merge into a single sample dictionary
        sample = {
            "image_id": torch.tensor([idx]),
            "imgae_name_for_track": img["file_name"],
            "boxes": torch.as_tensor(inst_bbox, dtype=torch.float32),
            "labels": torch.as_tensor(inst_label, dtype=torch.int64),
            # HOI Specific fields
            "inst_actions": torch.tensor(hoi_results["inst_actions"], dtype=torch.int64),
            "pair_boxes": torch.as_tensor(hoi_results["pair_bbox"], dtype=torch.float32),
            "pair_actions": torch.tensor(hoi_results["pair_actions"], dtype=torch.int64),
            "pair_violence": torch.tensor(hoi_results["pair_violence"], dtype=torch.int64),
            "pair_targets": torch.tensor(hoi_results["pair_targets"], dtype=torch.int64),
            "has_vcoco_labels": torch.tensor(hoi_results["has_vcoco"], dtype=torch.float32),
            # HHI Specific fields
            "human_boxes": torch.as_tensor(HHI_results["human_bbox"], dtype=torch.float32),
            "aggressor_index": torch.as_tensor(HHI_results["agg_idx"], dtype=torch.int64),
            "victim_index": torch.as_tensor(HHI_results["vic_idx"], dtype=torch.int64),
            "violence_actions": torch.tensor(HHI_results["vio_act"], dtype=torch.int64),
            "has_target_visible": torch.tensor(HHI_results["vis"], dtype=torch.float32),
        }
        return sample

## HHI
    def HHI_num_action(self):
        return len(self.HHI_action_list)

    def get_HHI_actions(self):
        return self.HHI_action_list

    def get_HHI_human_action(self):
        self.HHI_human_action = ['human_threaten','human_attack','human_point_weapon_at','human_kidnapping']
        self.num_HHI_subject_act = len(self.HHI_human_action)
        return self.HHI_human_action

    def num_HHI_human_act(self):
        """Number of intransitive (human-only) actions."""
        return self.num_HHI_subject_act
    

## COCO object
    def num_COCO_category(self):
        return len(self.COCO_CLASSES)

    def get_COCO_categories(self):
        return self.COCO_CLASSES
    

## Violence HOI
    def get_vio_actions(self):
        """
        Violence manual define
        """
        return self.verb_list
    
    def num_vio_action(self):
        return len(self.verb_list)
    
    def get_vio_human_action(self):
        self.human_action= ['human_aim','human_hit','human_raise','human_hold','human_sit','human_catch']
        self.num_subject_act = len(self.human_action)
        return self.human_action
    
    def get_vio_object_action(self):
        self.object_action= [
                'object_aim_instr', 'object_hit_instr', 'object_raise_instr', 'object_hold_instr', 'object_sit_obj', 'object_catch_instr'
                ]
        self.num_object_act = len(self.object_action)
        return self.object_action
    
    def num_vio_human_act(self):
        """Number of intransitive (human-only) actions."""
        return self.num_subject_act
    
    
    def get_valid_object_label_idx(self): 
        """
        Returns a list of length num_category() where entry i is 1 if
        COCO class i can appear as an interaction object, else 0.
        For violence we allow all non-background COCO classes.
        """
        self.obj_label_to_action=[7,8,9,10,11,12] # only these 6 COCO classes can be objects in violence actions]
        return self.obj_label_to_action
    
## VCOCO HOI
    def get_vcoco_actions(self):
        """
        VCOCO manual define
        """
        return self.VCOCO_ACTIONS
    
    def num_vcoco_action(self):
        return len(self.VCOCO_ACTIONS)
    
    def get_human_action_vcoco(self):
        self.human_action_vcoco= ['human_hold',
                'human_stand','human_sit','human_ride','human_walk','human_look',
                'human_hit','human_eat','human_jump','human_lay','human_talk_on_phone', 
                'human_carry', 'human_throw', 'human_catch', 'human_cut', 'human_run',
                'human_work_on_computer', 'human_ski', 'human_surf', 'human_skateboard', 'human_smile',
                'human_drink', 'human_kick', 'human_point', 'human_read', 'human_snowboard'
                ]
        self.num_subject_act_vcoco = len(self.human_action_vcoco)
        return self.human_action_vcoco

    def get_object_action_vcoco(self):
        self.object_action_vcoco= [
                'object_hold_obj', 'object_sit_instr', 'object_ride_instr', 'object_look_obj', 'object_hit_instr',
                'object_hit_obj', 'object_eat_instr', 'object_eat_obj', 'object_jump_instr', 'object_lay_instr',
                'object_talk_on_phone_instr', 'object_carry_obj', 'object_throw_obj', 'object_catch_obj', 'object_cut_instr',
                'object_cut_obj', 'object_work_on_computer_instr', 'object_ski_instr', 'object_surf_instr', 'object_skateboard_instr',
                'object_drink_instr', 'object_kick_obj', 'object_point_instr', 'object_read_obj', 'object_snowboard_instr'
                ]
        self.num_object_act_vcoco = len(self.object_action_vcoco)
        return self.object_action_vcoco

    def num_human_act_vcoco(self):
        """Number of intransitive (human-only) actions."""
        return self.num_subject_act_vcoco
    
    def get_object_label_idx_vcoco(self): 
        normal_obj_label_to_action = [
        0,  # hold
        0,  # stand
        1,  # sit
        2,  # ride
        0,  # walk
        3,  # look
        4, 5,  # hit
        6, 7,  # eat
        8,  # jump
        9,  # lay
        10, # talk_on_phone
        11, # carry
        12, # throw
        13, # catch
        14, 15, # cut
        0,  # run
        16, # work_on_computer
        17, # ski
        18, # surf
        19, # skateboard
        0,  # smile
        20, # drink
        21, # kick
        22, # point
        23, # read
        24,  # snowboard
        ]
        # In the vcoco.py call save_action_name() first that collect number of human action,
        # then call this get_object_label_idx_vcoco() to get the object label idx for each action.
        # The obj_idx start from 26
        vcoco_obj_label_to_action = [
        26,  # hold
        0,  # stand
        27,  # sit
        28,  # ride
        0,  # walk
        29,  # look
        30, 31,  # hit
        32, 33,  # eat
        34,  # jump
        35,  # lay
        36, # talk_on_phone
        37, # carry
        38, # throw
        39, # catch
        40, 41, # cut
        0,  # run
        42, # work_on_computer
        43, # ski
        44, # surf
        45, # skateboard
        0,  # smile
        46, # drink
        47, # kick
        48, # point
        49, # read
        50,  # snowboard
        ]
        return vcoco_obj_label_to_action



def make_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    # T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    if image_set == 'test':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided data path {root} does not exist'
    PATHS = {
        "train": (root / "images/train_images/", root / "violence_HOI_HHI/final_violence_train.json"),
        "val": (root / "images/val_images/", root / "violence_HOI_HHI/final_violence_val.json"),
        "test": (root / "images/val_images/", root / "violence_HOI_HHI/final_violence_val.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = UnifiedViolenceDetection(
        img_folder=img_folder,
        ann_file=ann_file,
        transforms=make_transforms(image_set)
    )

    dataset.file_meta['image_set'] = image_set

    return dataset













