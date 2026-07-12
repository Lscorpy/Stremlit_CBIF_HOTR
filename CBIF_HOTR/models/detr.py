# ------------------------------------------------------------------------
# Modified from HOTR (https://github.com/kakaobrain/hotr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
DETR & HOTR model and criterion classes.
"""


import torch
import torch.nn.functional as F 
from torch import nn

from CBIF_HOTR.util.misc import (NestedTensor, nested_tensor_from_tensor_list)

from .backbone import build_backbone

from .transformer import build_transformer, build_hoi_transformer, build_har_transformer

from .unified_postprocess import UnifiedPostProcess

from .feed_forward import MLP


from .CBIF_HOTR import DualBranchModel


 
class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary d ecoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1) #ori_num_classes + 1
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim) #100
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss


    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        out = {
            'pred_logits': outputs_class[-1], # (num_queries × num_classes)(100x94)
            'pred_boxes': outputs_coord[-1] # (num_queries × 4)(100x4)
            }
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

  
def build(args):
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    base_detr = DETR(
        backbone,
        transformer,
        num_classes=args.num_classes, # current 93
        num_queries=args.num_queries, #current 100
        aux_loss=args.aux_loss,
    )

    # ── Branch decoders ───────────────────────────────────────────────────────
    # Branch 1 – HOTR-style interaction transformer
    interaction_transformer = build_hoi_transformer(args)

    # Branch 2 – HHI-style HHI transformer
    har_transformer = build_har_transformer(args)


    model = DualBranchModel(
        detr                    = base_detr,
        # Branch 1
        num_hoi_queries         = args.num_hoi_queries,
        num_actions             = args.num_actions,
        num_violence_actions    = args.num_violence_actions,
        interaction_transformer = interaction_transformer,
        # Branch 2
        num_HHI_queries         = args.num_HHI_queries,
        num_HHI_actions         = args.num_HHI_action,
        har_transformer         = har_transformer,
        # Shared
        freeze_detr             = (args.frozen_weights is not None),
        share_enc               = args.share_enc,
        pretrained_dec          = args.pretrained_dec,
        temperature             = args.temperature,
        aux_loss                = args.aux_loss,
        hoi_aux_loss            = args.hoi_aux_loss,
        HHI_aux_loss            = args.HHI_aux_loss,
         # CBAF  (NEW)
        cbaf_nhead              = args.cbaf_nhead,
        cbaf_dropout            = args.cbaf_dropout,
        cbaf_use_gate           = args.cbaf_use_gate,
        cbaf_ffn                = args.cbaf_ffn,
        cbaf_use_conf_gate      = args.cbaf_use_conf_gate,
        anchor_loss_weight      = args.anchor_loss_weight,
    
    )


    postprocessors = UnifiedPostProcess(
        args.HOIDet,
        args.HHIDet,
    )
   
    return model, postprocessors
