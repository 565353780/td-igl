import os
import sys
import math
import time
import json
import torch
import datetime
import numpy as np
import torch.backends.cudnn as cudnn
from timm.utils import ModelEma
from typing import Iterable, Optional

from a_sdf.Loss.chamfer_distance import chamferDistance

from td_ilg.Data.smoothed_value import SmoothedValue
from td_ilg.Dataset.points import PointsDataset
from td_ilg.Model.asdf_autoencoder import ASDFAutoEncoder
from td_ilg.Method.io import save_model, auto_load_model
from td_ilg.Method.distributed import (
    init_distributed_mode,
    get_rank,
    get_world_size,
    is_main_process,
)
from td_ilg.Method.time import getCurrentTime
from td_ilg.Optimizer.opt import create_optimizer
from td_ilg.Optimizer.layer_decay_value_assigner import LayerDecayValueAssigner
from td_ilg.Optimizer.native_scaler import NativeScalerWithGradNormCount as NativeScaler
from td_ilg.Optimizer.scheduler import cosine_scheduler
from td_ilg.Module.Logger.metric import MetricLogger
from td_ilg.Module.Logger.tensorboard import TensorboardLogger


class ASDFAutoEncoderTrainer(object):
    def __init__(self) -> None:
        self.batch_size = 1
        self.epochs = 40000
        self.update_freq = 16
        self.save_ckpt_freq = 1
        self.drop = 0.0
        self.attn_drop_rate = 0.0
        self.drop_path = 0.1
        self.disable_eval = True
        self.model_ema = False

        self.opt = "adamw"
        self.lr = 1e-4
        self.warmup_lr = 1e-6
        self.min_lr = 1e-6
        self.weight_decay = 0.05
        self.weight_decay_end = None
        self.opt_eps = 1e-8
        self.opt_betas = None
        self.clip_grad = 0
        self.momentum = 0.9
        self.layer_decay = 1.0
        self.warmup_epochs = 1
        self.warmup_steps = -1

        self.points_dataset_folder_path = (
            "/home/chli/chLi/Dataset/ShapeNet/points/4000/"
        )
        self.device = "cuda"

        self.seed = 0
        self.resume = []
        self.auto_resume = True
        # self.no_auto_resume = True

        self.save_ckpt = True
        self.no_save_ckpt = False
        self.start_epoch = 0
        self.eval = False
        self.dist_eval = False
        self.num_workers = 16
        self.pin_mem = True
        self.no_pin_mem = False

        self.world_size = 1
        self.local_rank = -1
        self.dist_on_itp = False
        self.dist_url = "env://"

        current_time = getCurrentTime()
        # current_time = "v3"
        self.output_dir = "./output/" + current_time + "/"
        self.log_dir = "./logs/" + current_time + "/"
        return

    def train_batch(self, model, points: torch.Tensor):
        asdf_points = model(points)

        fit_dists2, coverage_dists2 = chamferDistance(
            asdf_points, points, self.device == "cpu"
        )[:2]

        fit_dists = torch.mean(torch.sqrt(fit_dists2) + 1e-6)
        coverage_dists = torch.mean(torch.sqrt(coverage_dists2) + 1e-6)

        loss_fit = torch.mean(fit_dists)
        loss_coverage = torch.mean(coverage_dists)
        loss = loss_fit + loss_coverage

        return (
            loss,
            loss_fit.item(),
            loss_coverage.item(),
        )

    def train_one_epoch(
        self,
        model: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        loss_scaler,
        max_norm: float = 0,
        model_ema: Optional[ModelEma] = None,
        log_writer=None,
        start_steps=None,
        lr_schedule_values=None,
        wd_schedule_values=None,
        num_training_steps_per_epoch=None,
        update_freq=None,
    ):
        model.train(True)
        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter(
            "min_lr", SmoothedValue(window_size=1, fmt="{value:.6f}")
        )
        header = "Epoch: [{}]".format(epoch)
        print_freq = 1

        if loss_scaler is None:
            model.zero_grad()
            model.micro_steps = 0
        else:
            optimizer.zero_grad()

        for data_iter_step, points in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
        ):
            step = data_iter_step // update_freq
            if step >= num_training_steps_per_epoch:
                continue
            it = start_steps + step  # global training iteration
            # Update LR & WD for the first acc
            if (
                lr_schedule_values is not None
                or wd_schedule_values is not None
                and data_iter_step % update_freq == 0
            ):
                for param_group in optimizer.param_groups:
                    if lr_schedule_values is not None:
                        param_group["lr"] = (
                            lr_schedule_values[it] * param_group["lr_scale"]
                        )
                    if (
                        wd_schedule_values is not None
                        and param_group["weight_decay"] > 0
                    ):
                        param_group["weight_decay"] = wd_schedule_values[it]

            points = points.to(self.device, non_blocking=True)

            if loss_scaler is None:
                raise NotImplementedError
            else:
                with torch.cuda.amp.autocast():
                    (
                        loss,
                        loss_fit,
                        loss_coverage,
                    ) = self.train_batch(model, points)

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            if loss_scaler is None:
                raise NotImplementedError
            else:
                # this attribute is added by timm on one optimizer (adahessian)
                is_second_order = (
                    hasattr(optimizer, "is_second_order") and optimizer.is_second_order
                )
                loss = loss / update_freq
                grad_norm = loss_scaler(
                    loss,
                    optimizer,
                    clip_grad=max_norm,
                    parameters=model.parameters(),
                    create_graph=is_second_order,
                    update_grad=(data_iter_step + 1) % update_freq == 0,
                )
                if (data_iter_step + 1) % update_freq == 0:
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(model)

                # FIXME: here loss_scale_value may not exist!
                loss_scale_value = None
                if "scale" in loss_scaler.state_dict().keys():
                    loss_scale_value = loss_scaler.state_dict()["scale"]

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)
            if loss_scale_value:
                metric_logger.update(loss_scale=loss_scale_value)
            metric_logger.update(loss_fit=loss_fit)
            metric_logger.update(loss_coverage=loss_coverage)

            min_lr = 10.0
            max_lr = 0.0
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            metric_logger.update(grad_norm=grad_norm)

            if log_writer is not None:
                log_writer.update(loss_fit=loss_fit, head="loss")
                log_writer.update(loss_coverage=loss_coverage, head="loss")
                log_writer.update(loss=loss_value, head="loss")
                if loss_scale_value:
                    log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")

                log_writer.set_step()

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def evaluate(self, data_loader, model):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Test:"

        # switch to evaluation mode
        model.eval()

        for points in metric_logger.log_every(data_loader, 1000, header):
            points = points.to(self.device, non_blocking=True)

            # compute output
            with torch.cuda.amp.autocast():
                asdf_points = model(points)

                fit_dists2, coverage_dists2 = chamferDistance(
                    asdf_points, points, self.device == "cpu"
                )[:2]

                fit_dists = torch.mean(torch.sqrt(fit_dists2) + 1e-6)
                coverage_dists = torch.mean(torch.sqrt(coverage_dists2) + 1e-6)

                loss_fit = torch.mean(fit_dists)
                loss_coverage = torch.mean(coverage_dists)
                loss = loss_fit + loss_coverage

            metric_logger.update(loss=loss.item())
            metric_logger.update(loss_fit=loss_fit.item())
            metric_logger.update(loss_coverage=loss_coverage.item())
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("* loss {losses.global_avg:.3f} ".format(losses=metric_logger.loss))
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    def train(self) -> bool:
        os.makedirs(self.output_dir, exist_ok=True)

        init_distributed_mode(self)

        # fix the seed for reproducibility
        seed = self.seed + get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)

        cudnn.benchmark = True

        dataset_train = PointsDataset(self.points_dataset_folder_path)

        if len(dataset_train) < self.batch_size:
            self.batch_size = len(dataset_train)

        if self.disable_eval:
            dataset_val = None
        else:
            dataset_val = PointsDataset(self.points_dataset_folder_path)

        if True:  # self.distributed:
            num_tasks = get_world_size()
            global_rank = get_rank()
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
            print("Sampler_train = %s" % str(sampler_train))
            if self.dist_eval:
                if len(dataset_val) % num_tasks != 0:
                    print(
                        "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                        "This will slightly alter validation results as extra duplicate entries are added to achieve "
                        "equal num of samples per-process."
                    )
                sampler_val = torch.utils.data.DistributedSampler(
                    dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
                )
            else:
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        if global_rank == 0 and self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)
            log_writer = TensorboardLogger(log_dir=self.log_dir)
        else:
            log_writer = None

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
            drop_last=True,
            prefetch_factor=1,
        )

        if dataset_val is not None:
            data_loader_val = torch.utils.data.DataLoader(
                dataset_val,
                sampler=sampler_val,
                batch_size=1,  # int(1.5 * self.batch_size),
                num_workers=self.num_workers,
                pin_memory=self.pin_mem,
                drop_last=False,
                # prefetch_factor=4,
            )
        else:
            data_loader_val = None

        model = ASDFAutoEncoder(
            asdf_channel=40, sh_2d_degree=3, sh_3d_degree=6, hidden_dim=128, dtype=torch.float32, device=self.device, sample_direction_num=200, direction_upscale=4
        )

        model.to(self.device)

        model_ema = None
        if self.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            model_ema = ModelEma(
                model,
                decay=self.model_ema_decay,
                device="cpu" if self.model_ema_force_cpu else "",
                resume="",
            )
            print("Using EMA with decay = %.8f" % self.model_ema_decay)

        model_without_ddp = model
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("Model = %s" % str(model_without_ddp))
        print("number of params:", n_parameters)

        total_batch_size = self.batch_size * self.update_freq * get_world_size()
        num_training_steps_per_epoch = len(dataset_train) // total_batch_size
        self.lr = self.lr * total_batch_size / 256
        print("LR = %.8f" % self.lr)
        print("Batch size = %d" % total_batch_size)
        print("Update frequent = %d" % self.update_freq)
        print("Number of training examples = %d" % len(dataset_train))
        print(
            "Number of training training per epoch = %d" % num_training_steps_per_epoch
        )

        # num_layers = model_without_ddp.get_num_layers()
        num_layers = 12
        if self.layer_decay < 1.0:
            assigner = LayerDecayValueAssigner(
                list(
                    self.layer_decay ** (num_layers + 1 - i)
                    for i in range(num_layers + 2)
                )
            )
        else:
            assigner = None

        if assigner is not None:
            print("Assigned values = %s" % str(assigner.values))

        skip_weight_decay_list = model.no_weight_decay()
        print("Skip weight decay list: ", skip_weight_decay_list)

        if self.distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[self.gpu], find_unused_parameters=False
            )
            model_without_ddp = model.module

        optimizer = create_optimizer(
            self,
            model_without_ddp,
            skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None,
            get_layer_scale=assigner.get_scale if assigner is not None else None,
        )
        loss_scaler = NativeScaler()

        print("Use step level LR scheduler!")
        lr_schedule_values = cosine_scheduler(
            self.lr,
            self.min_lr,
            self.epochs,
            num_training_steps_per_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_steps=self.warmup_steps,
        )
        if self.weight_decay_end is None:
            self.weight_decay_end = self.weight_decay
        wd_schedule_values = cosine_scheduler(
            self.weight_decay,
            self.weight_decay_end,
            self.epochs,
            num_training_steps_per_epoch,
        )
        print(
            "Max WD = %.7f, Min WD = %.7f"
            % (max(wd_schedule_values), min(wd_schedule_values))
        )

        auto_load_model(
            args=self,
            model=model,
            model_without_ddp=model_without_ddp,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            model_ema=model_ema,
        )

        print(f"Start training for {self.epochs} epochs")
        start_time = time.time()
        max_accuracy = 0.0
        for epoch in range(self.start_epoch, self.epochs):
            if self.distributed:
                data_loader_train.sampler.set_epoch(epoch)

            train_stats = self.train_one_epoch(
                model,
                data_loader_train,
                optimizer,
                epoch,
                loss_scaler,
                self.clip_grad,
                model_ema,
                log_writer=log_writer,
                start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values,
                wd_schedule_values=wd_schedule_values,
                num_training_steps_per_epoch=num_training_steps_per_epoch,
                update_freq=self.update_freq,
            )

            if self.output_dir and self.save_ckpt:
                if (epoch + 1) % self.save_ckpt_freq == 0 or epoch + 1 == self.epochs:
                    save_model(
                        args=self,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch=epoch,
                        model_ema=model_ema,
                    )
            if data_loader_val is not None and (
                epoch % 10 == 0 or epoch + 1 == self.epochs
            ):
                test_stats = self.evaluate(data_loader_val, model)

                if self.output_dir and self.save_ckpt:
                    save_model(
                        args=self,
                        model=model,
                        model_without_ddp=model_without_ddp,
                        optimizer=optimizer,
                        loss_scaler=loss_scaler,
                        epoch="best",
                        model_ema=model_ema,
                    )

                print(f"Max accuracy: {max_accuracy:.2f}%")
                if log_writer is not None:
                    log_writer.update(
                        test_loss=test_stats["loss"], head="perf", step=epoch
                    )

                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    **{f"test_{k}": v for k, v in test_stats.items()},
                    "epoch": epoch,
                    "n_parameters": n_parameters,
                }
            else:
                log_stats = {
                    **{f"train_{k}": v for k, v in train_stats.items()},
                    # **{f'test_{k}': v for k, v in test_stats.items()},
                    "epoch": epoch,
                    "n_parameters": n_parameters,
                }

            if self.output_dir and is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(
                    os.path.join(self.output_dir, "log.txt"), mode="a", encoding="utf-8"
                ) as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

        return True
