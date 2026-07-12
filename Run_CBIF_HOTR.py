import types

DATA_PATH = r"dataset"
WEIGHTS   = r"vcoco_q16.pth"
OUTPUT    = r"DUAL_cross_branch_model"

DEVICE = "cuda"


def build_args():
    args = types.SimpleNamespace(

        # ===== Backbone =====
        backbone='resnet50',
        dilation=False,
        position_embedding='sine',
        masks             = False,

        data_path=DATA_PATH,

        device=DEVICE,

        # ── Training ─────────────────────────────────────────────────
        batch_size    = 2,
        lr            = 1e-4,
        lr_backbone   = 1e-6,
        weight_decay  = 1e-5,
        epochs        = 60,
        lr_drop       = 30,
        num_workers   = 2,
        clip_max_norm = 0.1,


        # Evaluation
        validate=True,   # run val loop each epoch
        print_validate=True,
        eval=True,
 
        wandb = False,

        # Needed by save_ckpt / hoi_evaluator
        output_dir=OUTPUT,
        distributed=False,
        seed=42,
        start_epoch=0,

        # transformer
        hidden_dim=256,
        enc_layers=6,
        dec_layers=6,
        nheads=8,
        dim_feedforward=2048,
        dropout=0.1,
        pre_norm=False,

        # HHI transformer
        HHI_enc_layers=6,
        HHI_dec_layers=6,
        HHI_nheads=8,
        HHI_dim_feedforward=2048,

        # HOI transformer
        HOI_enc_layers=6,
        HOI_dec_layers=6,
        HOI_nheads=8,
        HOI_dim_feedforward=2048,

        # DETR
        num_classes=93,
        num_queries=100,
        # ── DETR loss coefficients
        aux_loss=True,
        bbox_loss_coef=5,
        giou_loss_coef=2,
        eos_coef=0.1,

        # HOI
        HOIDet=True,
        num_vcoco_actions=29, #29 VCOCO and 1 background
        num_hoi_queries=16,
        hoi_aux_loss=True,
        hoi_idx_loss_coef=1,
        hoi_act_loss_coef=5.0,
        hoi_violence_loss_coef = 10.0, # first use:0.8


        # ── Violence head ────────────────────────────────────────────
        num_violence_actions = 6,

        vcoco_action_names=None,
        human_actions_vcoco=None,
        object_actions_vcoco=None,
        num_human_act_vcoco=None,

        violence_action_names=None,
        human_actions_violence=None,
        object_actions_violence=None,
        num_human_act_violence=None,

        # VCOCO
        valid_ids_vcoco=list(range(29)),
        invalid_ids_vcoco=[],
        valid_ids_violence=list(range(6)),

        # HHI
        HHIDet=True,
        num_HHI_queries=8,
        num_HHI_action=4,
        HHI_idx_loss_coef=1,
        HHI_action_idx_loss_coef=10,
        HHI_aux_loss=True,

        HHI_action_names=None,
        human_actions_HHI=None,
        num_human_act_HHI=None,



        #HHI, HOTR
        share_enc=True,
        pretrained_dec=True,
        temperature=0.05,
        frozen_weights=None,
        freeze_detr=False,


        # matcher
        set_cost_class=1,
        set_cost_bbox=5,
        set_cost_giou=2,
        set_cost_idx=10,
        set_cost_act   = 5,

        # CBAF  (NEW)
        cbaf_nhead              = 8,
        cbaf_dropout            = 0.1,
        cbaf_use_gate           = True,
        cbaf_ffn                = True,
        cbaf_use_conf_gate  = True,
        anchor_loss_weight  =  0.5,
    )

    return args