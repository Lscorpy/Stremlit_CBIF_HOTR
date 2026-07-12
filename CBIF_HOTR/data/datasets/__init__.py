# ------------------------------------------------------------------------
# Modified from HOTR (https://github.com/kakaobrain/hotr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


from CBIF_HOTR.data.datasets.Unified_dataset import build as build_merge_data

def build_dataset(image_set, args):
        return build_merge_data(image_set, args)