import argparse
import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import timm
import torch
import wandb
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import ConcatDataset, DataLoader

from config.config import Config, load_config
from src.dataset import WhaleDataset, load_df
from src.metric_learning import ArcFaceLossAdaptiveMargin, ArcMarginProductSubcenter, GeM
from src.utils import WarmupCosineLambda, map_dict, topk_average_precision


def parse():
    parser = argparse.ArgumentParser(description="Training for HappyWhale")
    parser.add_argument("--out_base_dir", default="result")
    parser.add_argument("--in_base_dir", default="input")
    parser.add_argument("--exp_name", default="tmp")
    parser.add_argument("--load_snapshot", action="store_true")
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--wandb_logger", action="store_true")
    parser.add_argument("--config_path", default="config/debug.yaml")
    return parser.parse_args()


class WhaleDataModule(LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Config,
        image_dir: str,
        val_bbox_name: str,
        fold: int,
        additional_dataset: WhaleDataset = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.image_dir = image_dir
        self.val_bbox_name = val_bbox_name
        self.additional_dataset = additional_dataset
        if cfg.n_data != -1:
            df = df.iloc[: cfg.n_data]
        self.all_df = df
        if fold == -1:
            self.train_df = df
        else:
            skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=0)
            train_idx, val_idx = list(skf.split(df, df.individual_id))[fold]
            self.train_df = df.iloc[train_idx].copy()
            self.val_df = df.iloc[val_idx].copy()
            # relabel ids not included in training data as "new individual"
            new_mask = ~self.val_df.individual_id.isin(self.train_df.individual_id)
            self.val_df.individual_id.mask(new_mask, cfg.num_classes, inplace=True)
            print(f"new: {(self.val_df.individual_id == cfg.num_classes).sum()} / {len(self.val_df)}")

    def get_dataset(self, df, data_aug):
        return WhaleDataset(df, self.cfg, self.image_dir, self.val_bbox_name, data_aug)

    def train_dataloader(self):
        dataset = self.get_dataset(self.train_df, True)
        if self.additional_dataset is not None:
            dataset = ConcatDataset([dataset, self.additional_dataset])
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.cfg.n_splits == -1:
            return None
        return DataLoader(
            self.get_dataset(self.val_df, False),
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    def all_dataloader(self):
        return DataLoader(
            self.get_dataset(self.all_df, False),
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )


class SphereClassifier(LightningModule):
    def __init__(self, cfg: dict, id_class_nums=None, species_class_nums=None):
        super().__init__()
        if not isinstance(cfg, Config):
            cfg = Config(cfg)
        self.save_hyperparameters(cfg, ignore=["id_class_nums", "species_class_nums"])
        self.test_results_fp = None

        # NN architecture
        self.backbone = timm.create_model(
            cfg.model_name,
            in_chans=3,
            pretrained=cfg.pretrained,
            num_classes=0,
            features_only=True,
            out_indices=cfg.out_indices,
        )
        feature_dims = self.backbone.feature_info.channels()
        print(f"feature dims: {feature_dims}")
        self.global_pools = torch.nn.ModuleList(
            [GeM(p=cfg.global_pool.p, requires_grad=cfg.global_pool.train) for _ in cfg.out_indices]
        )
        self.mid_features = np.sum(feature_dims)
        if cfg.normalization == "batchnorm":
            self.neck = torch.nn.BatchNorm1d(self.mid_features)
        elif cfg.normalization == "layernorm":
            self.neck = torch.nn.LayerNorm(self.mid_features)
        self.head_id = ArcMarginProductSubcenter(self.mid_features, cfg.num_classes, cfg.n_center_id)
        self.head_species = ArcMarginProductSubcenter(self.mid_features, cfg.num_species_classes, cfg.n_center_species)
        if id_class_nums is not None and species_class_nums is not None:
            margins_id = np.power(id_class_nums, cfg.margin_power_id) * cfg.margin_coef_id + cfg.margin_cons_id
            margins_species = (
                np.power(species_class_nums, cfg.margin_power_species) * cfg.margin_coef_species
                + cfg.margin_cons_species
            )
            print("margins_id", margins_id)
            print("margins_species", margins_species)
            self.margin_fn_id = ArcFaceLossAdaptiveMargin(margins_id, cfg.num_classes, cfg.s_id)
            self.margin_fn_species = ArcFaceLossAdaptiveMargin(margins_species, cfg.num_species_classes, cfg.s_species)
            self.loss_fn_id = torch.nn.CrossEntropyLoss()
            self.loss_fn_species = torch.nn.CrossEntropyLoss()

    def get_feat(self, x: torch.Tensor) -> torch.Tensor:
        ms = self.backbone(x)
        h = torch.cat([global_pool(m) for m, global_pool in zip(ms, self.global_pools)], dim=1)
        return self.neck(h)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.get_feat(x)
        return self.head_id(feat), self.head_species(feat)

    def training_step(self, batch, batch_idx):
        x, ids, species = batch["image"], batch["label"], batch["label_species"]
        logits_ids, logits_species = self(x)
        margin_logits_ids = self.margin_fn_id(logits_ids, ids)
        loss_ids = self.loss_fn_id(margin_logits_ids, ids)
        loss_species = self.loss_fn_species(self.margin_fn_species(logits_species, species), species)
        self.log_dict({"train/loss_ids": loss_ids.detach()}, on_step=False, on_epoch=True)
        self.log_dict({"train/loss_species": loss_species.detach()}, on_step=False, on_epoch=True)
        with torch.no_grad():
            self.log_dict(map_dict(logits_ids, ids, "train"), on_step=False, on_epoch=True)
            self.log_dict(
                {"train/acc_species": topk_average_precision(logits_species, species, 1).mean().detach()},
                on_step=False,
                on_epoch=True,
            )
        return loss_ids * self.hparams.loss_id_ratio + loss_species * (1 - self.hparams.loss_id_ratio)

    def validation_step(self, batch, batch_idx):
        x, ids, species = batch["image"], batch["label"], batch["label_species"]
        out1, out_species1 = self(x)
        out2, out_species2 = self(x.flip(3))
        output, output_species = (out1 + out2) / 2, (out_species1 + out_species2) / 2
        self.log_dict(map_dict(output, ids, "val"), on_step=False, on_epoch=True)
        self.log_dict(
            {"val/acc_species": topk_average_precision(output_species, species, 1).mean().detach()},
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        backbone_params = list(self.backbone.parameters()) + list(self.global_pools.parameters())
        head_params = (
            list(self.neck.parameters()) + list(self.head_id.parameters()) + list(self.head_species.parameters())
        )
        params = [
            {"params": backbone_params, "lr": self.hparams.lr_backbone},
            {"params": head_params, "lr": self.hparams.lr_head},
        ]
        if self.hparams.optimizer == "Adam":
            optimizer = torch.optim.Adam(params)
        elif self.hparams.optimizer == "AdamW":
            optimizer = torch.optim.AdamW(params)
        elif self.hparams.optimizer == "RAdam":
            optimizer = torch.optim.RAdam(params)

        warmup_steps = self.hparams.max_epochs * self.hparams.warmup_steps_ratio
        cycle_steps = self.hparams.max_epochs - warmup_steps
        lr_lambda = WarmupCosineLambda(warmup_steps, cycle_steps, self.hparams.lr_decay_scale)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return [optimizer], [scheduler]

    def test_step(self, batch, batch_idx):
        x = batch["image"]
        feat1 = self.get_feat(x)
        out1, out_species1 = self.head_id(feat1), self.head_species(feat1)
        feat2 = self.get_feat(x.flip(3))
        out2, out_species2 = self.head_id(feat2), self.head_species(feat2)
        pred_logit, pred_idx = ((out1 + out2) / 2).cpu().sort(descending=True)
        return {
            "original_index": batch["original_index"],
            "label": batch["label"],
            "label_species": batch["label_species"],
            "pred_logit": pred_logit[:, :1000],
            "pred_idx": pred_idx[:, :1000],
            "pred_species": ((out_species1 + out_species2) / 2).cpu(),
            "embed_features1": feat1.cpu(),
            "embed_features2": feat2.cpu(),
        }

    def test_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        outputs = self.all_gather(outputs)
        if self.trainer.global_rank == 0:
            epoch_results: Dict[str, np.ndarray] = {}
            for key in outputs[0].keys():
                if torch.cuda.device_count() > 1:
                    result = torch.cat([x[key] for x in outputs], dim=1).flatten(end_dim=1)
                else:
                    result = torch.cat([x[key] for x in outputs], dim=0)
                epoch_results[key] = result.detach().cpu().numpy()
            np.savez_compressed(self.test_results_fp, **epoch_results)


def train(
    df: pd.DataFrame,
    args: argparse.Namespace,
    cfg: Config,
    fold: int,
    do_inference: bool = False,
    additional_dataset: WhaleDataset = None,
    optuna_trial: Optional[optuna.Trial] = None,
) -> Optional[float]:
    out_dir = f"{args.out_base_dir}/{args.exp_name}/{fold}"
    id_class_nums = df.individual_id.value_counts().sort_index().values
    species_class_nums = df.species.value_counts().sort_index().values
    model = SphereClassifier(cfg, id_class_nums=id_class_nums, species_class_nums=species_class_nums)
    data_module = WhaleDataModule(
        df, cfg, f"{args.in_base_dir}/train_images", cfg.val_bbox, fold, additional_dataset=additional_dataset
    )
    loggers = [pl_loggers.CSVLogger(out_dir)]
    if args.wandb_logger:
        loggers.append(
            pl_loggers.WandbLogger(
                project="kaggle-happywhale", group=args.exp_name, name=f"{args.exp_name}/{fold}", save_dir=out_dir
            )
        )
    callbacks = [LearningRateMonitor("epoch")]
    if optuna_trial is not None:
        callbacks.append(PyTorchLightningPruningCallback(optuna_trial, "val/mapNone"))
    if args.save_checkpoint:
        callbacks.append(ModelCheckpoint(out_dir, save_last=True, save_top_k=0))
    trainer = Trainer(
        gpus=torch.cuda.device_count(),
        max_epochs=cfg["max_epochs"],
        logger=loggers,
        callbacks=callbacks,
        checkpoint_callback=args.save_checkpoint,
        precision=16,
        sync_batchnorm=True,
    )
    ckpt_path = f"{out_dir}/last.ckpt"
    if not os.path.exists(ckpt_path) or not args.load_snapshot:
        ckpt_path = None
    trainer.fit(model, ckpt_path=ckpt_path, datamodule=data_module)
    if do_inference:
        for test_bbox in cfg.test_bboxes:
            # all train data
            model.test_results_fp = f"{out_dir}/train_{test_bbox}_results.npz"
            trainer.test(model, data_module.all_dataloader())
            # test data
            model.test_results_fp = f"{out_dir}/test_{test_bbox}_results.npz"
            df_test = load_df(args.in_base_dir, cfg, "sample_submission.csv", False)
            test_data_module = WhaleDataModule(df_test, cfg, f"{args.in_base_dir}/test_images", test_bbox, -1)
            trainer.test(model, test_data_module.all_dataloader())

    if args.wandb_logger:
        wandb.finish()
    if optuna_trial is not None:
        return trainer.callback_metrics["val/mapNone"].item()
    else:
        return None


def main():
    args = parse()
    warnings.filterwarnings("ignore", ".*does not have many workers.*")
    cfg = load_config(args.config_path, "config/default.yaml")
    print(cfg)
    df = load_df(args.in_base_dir, cfg, "train.csv", True)
    pseudo_dataset = None
    if cfg.pseudo_label is not None:
        pseudo_df = load_df(args.in_base_dir, cfg, cfg.pseudo_label, False)
        pseudo_dataset = WhaleDataset(
            pseudo_df[pseudo_df.conf > cfg.pseudo_conf_threshold], cfg, f"{args.in_base_dir}/test_images", "", True
        )
    if cfg["n_splits"] == -1:
        train(df, args, cfg, -1, do_inference=True, additional_dataset=pseudo_dataset)
    else:
        train(df, args, cfg, 0, do_inference=True, additional_dataset=pseudo_dataset)


if __name__ == "__main__":
    main()
