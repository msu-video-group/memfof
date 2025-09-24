import argparse

import torch
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils import data
import torch.nn.functional as F

from memfof import datasets
from memfof.model import MEMFOF
from memfof.datasets import fetch_dataloader
from memfof.utils.utils import load_ckpt
from memfof.loss import sequence_loss


class MEMFOFLit(pl.LightningModule):
    """PyTorch Lightning module for MEMFOF optical flow estimation.

    This class implements the training and validation logic for the MEMFOF model
    using PyTorch Lightning framework.

    Parameters
    ----------
    args : argparse.Namespace
        Configuration parameters for the model and training process.
    """

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.model = MEMFOF(
            backbone=self.args.pretrain,
            dim=self.args.dim,
            corr_radius=self.args.radius,
            num_blocks=self.args.num_blocks,
            use_var=self.args.use_var,
            var_min=self.args.var_min,
            var_max=self.args.var_max,
        )
        if self.args.restore_ckpt is not None:
            load_ckpt(self, self.args.restore_ckpt)
            print(f"restore ckpt from {self.args.restore_ckpt}")
        self.log_kwargs = {"sync_dist": True, "add_dataloader_idx": False}

    def training_step(
        self, data_blob: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Perform a single training step.

        Parameters
        ----------
        data_blob : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing (images, flow_gts, valids) tensors.
            - images: Input image sequence of shape (B, 3, 3, H, W)
            - flow_gts: Ground truth flow fields of shape (B, 2, 2, H, W)
            - valids: Validity masks of shape (B, 2, H, W)

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        images, flow_gts, valids = data_blob
        outputs = self.model(images, flow_gts=flow_gts, iters=self.args.iters)
        loss = sequence_loss(outputs, flow_gts, valids, self.args.gamma)
        self.log("train-sequence-loss", loss, **self.log_kwargs)
        return loss

    def backward_flow(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate backward optical flow.

        Parameters
        ----------
        images : torch.Tensor
            Input image sequence of shape (B, 3, 3, H, W)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing (flow, info) tensors.
            - flow: Backward flow field of shape (B, 2, H, W)
            - info: Additional information tensor of shape (B, 4, H, W)
        """
        output = self.model(images, iters=self.args.iters)
        flow_final = output["flow"][-1][:, 0]
        info_final = output["info"][-1][:, 0]
        return flow_final, info_final

    def forward_flow(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate forward optical flow.

        Parameters
        ----------
        images : torch.Tensor
            Input image sequence of shape (B, 3, 3, H, W)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing (flow, info) tensors.
            - flow: Forward flow field of shape (B, 2, H, W)
            - info: Additional information tensor of shape (B, 4, H, W)
        """
        output = self.model(images, iters=self.args.iters)
        flow_final = output["flow"][-1][:, 1]
        info_final = output["info"][-1][:, 1]
        return flow_final, info_final

    def get_val_scale(self, data_name: str) -> int:
        """Get validation scale factor for different datasets.

        Parameters
        ----------
        data_name : str
            Name of the validation dataset.

        Returns
        -------
        int
            Scale factor for validation. 0 means no scaling, 1 means 2x scaling, etc.
        """
        return {"spring": 0}.get(data_name, 1)

    def scale_and_forward_flow(
        self, images: torch.Tensor, scale: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate optical flow with specified scale.

        Parameters
        ----------
        images : torch.Tensor
            Input image sequence of shape (B, 3, 3, H, W)
        scale : float
            Images will be scaled 2^scale times

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple containing (flow, info) tensors.
            - flow: Optical flow field of shape (B, 2, H, W)
            - info: Additional information tensor of shape (B, 4, H, W)
        """
        imgs = []
        for i in range(3):
            imgs.append(
                F.interpolate(
                    images[:, i],
                    scale_factor=2**scale,
                    mode="bilinear",
                    align_corners=False,
                )
            )
        imgs = torch.stack(imgs, dim=1)

        flow, info = self.forward_flow(imgs)
        flow_down = F.interpolate(
            flow, scale_factor=0.5**scale, mode="bilinear", align_corners=False
        ) * (0.5**scale)
        info_down = F.interpolate(info, scale_factor=0.5**scale, mode="area")
        return flow_down, info_down

    def validation_step_1(
        self, data_blob: tuple[torch.Tensor, torch.Tensor, torch.Tensor], data_name: str
    ) -> None:
        """Validation step for chairs, sintel, and spring datasets.

        Parameters
        ----------
        data_blob : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing (images, flow_gts, valids) tensors.
        data_name : str
            Name of the validation dataset.
        """
        images, flow_gt, _ = data_blob
        flow_gt = flow_gt.squeeze(dim=1)
        flow, _ = self.scale_and_forward_flow(images, self.get_val_scale(data_name))
        epe = torch.sum((flow - flow_gt) ** 2, dim=1).sqrt()
        px1 = (epe < 1.0).float().mean(dim=[1, 2])
        px3 = (epe < 3.0).float().mean(dim=[1, 2])
        px5 = (epe < 5.0).float().mean(dim=[1, 2])
        epe = epe.mean(dim=[1, 2])
        self.log(
            f"val-{data_name}-px1",
            100 * (1 - px1).mean(),
            **self.log_kwargs,
        )
        self.log(
            f"val-{data_name}-px3",
            100 * (1 - px3).mean(),
            **self.log_kwargs,
        )
        self.log(
            f"val-{data_name}-px5",
            100 * (1 - px5).mean(),
            **self.log_kwargs,
        )
        self.log(f"val-{data_name}-epe", epe.mean(), **self.log_kwargs)

    def validation_step_2(
        self, data_blob: tuple[torch.Tensor, torch.Tensor, torch.Tensor], data_name: str
    ) -> None:
        """Validation step for KITTI dataset.

        Parameters
        ----------
        data_blob : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing (images, flows_gt, valids_gt) tensors.
        data_name : str
            Name of the validation dataset.
        """
        images, flow_gt, valid_gt = data_blob
        flow_gt = flow_gt.squeeze(dim=1)
        valid_gt = valid_gt.squeeze(dim=1)
        flow, _ = self.scale_and_forward_flow(images, self.get_val_scale(data_name))
        epe = torch.sum((flow - flow_gt) ** 2, dim=1).sqrt()
        mag = torch.sum(flow_gt**2, dim=1).sqrt()
        val = valid_gt >= 0.5
        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()
        epe_list = []
        out_valid_pixels = 0
        num_valid_pixels = 0
        for b in range(out.shape[0]):
            epe_list.append(epe[b][val[b]].mean())
            out_valid_pixels += out[b][val[b]].sum()
            num_valid_pixels += val[b].sum()
        epe = torch.mean(torch.tensor(epe_list, device=self.device))
        f1 = 100 * out_valid_pixels / num_valid_pixels
        self.log(f"val-{data_name}-epe", epe, **self.log_kwargs)
        self.log(f"val-{data_name}-f1", f1, **self.log_kwargs)

    def validation_step(
        self,
        data_blob: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Main validation step that routes to specific validation methods.

        Parameters
        ----------
        data_blob : Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Tuple containing validation data tensors.
        batch_idx : int
            Index of the current batch.
        dataloader_idx : int, optional
            Index of the current dataloader, by default 0
        """
        if not self.args.val_datasets:
            return
        data_name = self.args.val_datasets[dataloader_idx]
        if data_name in (
            "chairs",
            "sintel",
            "sintel-clean",
            "sintel-final",
            "spring",
            "spring-1080",
        ):
            self.validation_step_1(data_blob, data_name)
        elif data_name in ("kitti",):
            self.validation_step_2(data_blob, data_name)

    def configure_optimizers(self) -> dict:
        """Configure optimizers and learning rate schedulers.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing optimizer and scheduler configurations.
        """
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.wdecay,
            eps=self.args.epsilon,
        )
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.args.lr,
            self.args.num_steps + 100,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy="linear",
        )
        lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}


class DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for MEMFOF training and validation.

    Parameters
    ----------
    args : argparse.Namespace
        Configuration parameters for data loading.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.args = args

    def train_dataloader(self) -> data.DataLoader:
        """Get training dataloader.

        Returns
        -------
        data.DataLoader
            Training dataloader instance.
        """
        return fetch_dataloader(self.args)

    def val_dataloader(self) -> list[data.DataLoader]:
        """Get validation dataloaders for different datasets.

        Returns
        -------
        List[data.DataLoader]
            List of validation dataloaders.
        """
        kwargs = {
            "pin_memory": False,
            "shuffle": False,
            "num_workers": self.args.num_workers,
            "drop_last": False,
        }
        val_dataloaders = []
        for val_dataset in self.args.val_datasets:
            if val_dataset == "sintel":
                clean = datasets.three_frame_wrapper_val(
                    datasets.MpiSintel, {"split": "val", "dstype": "clean"}
                )
                final = datasets.three_frame_wrapper_val(
                    datasets.MpiSintel, {"split": "val", "dstype": "final"}
                )
                loader = data.DataLoader(clean + final, batch_size=8, **kwargs)
            elif val_dataset == "sintel-clean":
                clean = datasets.three_frame_wrapper_val(
                    datasets.MpiSintel, {"split": "val", "dstype": "clean"}
                )
                loader = data.DataLoader(clean, batch_size=8, **kwargs)
            elif val_dataset == "sintel-final":
                final = datasets.three_frame_wrapper_val(
                    datasets.MpiSintel, {"split": "val", "dstype": "final"}
                )
                loader = data.DataLoader(final, batch_size=8, **kwargs)
            elif val_dataset == "kitti":
                kitti = datasets.three_frame_wrapper_val(
                    datasets.KITTI, {"split": "val"}
                )
                loader = data.DataLoader(kitti, batch_size=1, **kwargs)
            elif val_dataset == "spring":
                spring = datasets.three_frame_wrapper_val(
                    datasets.SpringFlowDataset, {"split": "val"}
                )
                loader = data.DataLoader(spring, batch_size=4, **kwargs)
            else:
                raise ValueError(f"Unknown validation dataset: {val_dataset}")
            val_dataloaders.append(loader)
        return val_dataloaders
