import os
import argparse

import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import TorchSyncBatchNorm

from config.parser import parse_args
from memfof.model_lit import MEMFOFLit, DataModule


def detect_cluster(args: argparse.Namespace) -> argparse.Namespace:
    if all(env in os.environ for env in ("SLURM_NTASKS_PER_NODE", "SLURM_JOB_NUM_NODES")):
        args.devices = int(os.environ["SLURM_NTASKS_PER_NODE"])
        args.num_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
    elif all(env in os.environ for env in ("WORLD_SIZE", "LOCAL_WORLD_SIZE")):
        args.devices = int(os.environ["LOCAL_WORLD_SIZE"])
        args.num_nodes = int(os.environ["WORLD_SIZE"]) // args.devices
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, help="experiment config file name", required=True)
    args = parse_args(parser)
    args = detect_cluster(args)

    if args.effective_batch_size % (args.num_nodes * args.devices * args.accumulate_grad_batches) != 0:
        raise ValueError(
            f"Requested effective_batch_size={args.effective_batch_size} can not be split into {args.num_nodes} nodes with {args.devices} devices each with accumulate_grad_batches={args.accumulate_grad_batches}."
        )

    args.batch_size = int(args.effective_batch_size / (args.num_nodes * args.devices * args.accumulate_grad_batches))

    monitor = LearningRateMonitor()
    checkpoint = ModelCheckpoint(
        dirpath="ckpts",
        filename=args.name,
        monitor=args.monitor,
        every_n_train_steps=args.num_steps if args.monitor is None else None,
    )

    wandb_logger = WandbLogger(
        project="MEMFOF",
        config=vars(args),
        log_model=True,
        checkpoint_name=args.name,
    )

    plugins = [
        TorchSyncBatchNorm(),
    ]

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.devices,
        strategy="ddp",
        num_nodes=args.num_nodes,
        logger=wandb_logger,
        gradient_clip_val=args.clip,
        precision="bf16-mixed",
        max_steps=args.num_steps,
        check_val_every_n_epoch=None,
        val_check_interval=args.val_steps * args.accumulate_grad_batches,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[monitor, checkpoint],
        plugins=plugins,
    )

    model = MEMFOFLit(args)
    datamodule = DataModule(args)
    trainer.fit(model, datamodule)
    wandb.finish()
