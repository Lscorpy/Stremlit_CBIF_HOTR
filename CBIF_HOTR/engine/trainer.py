# ------------------------------------------------------------------------
# Modified from HOTR (https://github.com/kakaobrain/hotr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import math
import torch
import sys
import CBIF_HOTR.util.misc as utils
import CBIF_HOTR.util.logger as loggers
from typing import Iterable
import wandb


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_epoch: int, max_norm: float = 0, log: bool = False):
    model.train()
    criterion.train()
    metric_logger = loggers.MetricLogger(mode="train", delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    space_fmt = str(len(str(max_epoch)))
    header = 'Epoch [{start_epoch: >{fill}}/{end_epoch}]'.format(start_epoch=epoch+1, end_epoch=max_epoch, fill=space_fmt)
    print_freq = int(len(data_loader)/5)


    print(f"\n>>> Epoch #{(epoch+1)}")
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device)

        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]


        outputs = model(samples)

        loss_dict = criterion(outputs, targets, log=False)

        weight_dict = criterion.combined_weight_dict


        loss_total = sum(
            loss_dict[k] * weight_dict[k]
            for k in loss_dict
            if k in weight_dict
        )

        loss_dict["loss_total"] = loss_total

        loss_dict_reduced = utils.reduce_dict(loss_dict)

        # Scaled versions of individual losses (for logging only)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print(f"[trainer.py] Loss is {loss_value}, stopping training.")
            print(loss_dict_reduced)
            sys.exit(1)

        # ── Optimiser step ────────────────────────────────────────────────────
        optimizer.zero_grad()
        loss_total.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        #after variables are defined)
        if utils.get_rank() == 0 and log:
            wandb.log({
                **loss_dict_reduced_scaled,
                "loss_total": loss_total
            })

        if epoch == 0:
            print("DEBUG in trainer.py")
            print("loss_total:", loss_total)
            print( )



        metric_logger.update(
            loss_total=loss_total,
            **loss_dict_reduced_scaled
        )

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
