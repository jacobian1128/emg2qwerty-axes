# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
)
from emg2qwerty.transforms import Transform
import copy


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    # Feed the entire session at once without windowing/padding
                    # at test time for more realism
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=False,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=False,
        )


class TDSConvCTCModule(pl.LightningModule):
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        share_hand_weights: bool = False,
        split_decoding: bool = False, 
        spec_norm: str = 'BatchNorm2d',
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        spec_norm_channels = self.NUM_BANDS * self.ELECTRODE_CHANNELS

        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=spec_norm_channels, spec_norm_type=spec_norm),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
                share_hand_weights=share_hand_weights
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),
            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
                share_hand_weights=share_hand_weights
            ),
            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )

class TDSConvCTCFinetuneModule(TDSConvCTCModule):
    def __init__(self, *args, **kwargs):
        # Initialize the original model
        super().__init__(*args, **kwargs)

        # Replace shared hand_processor with separate left/right processors
        self.left_processor = copy.deepcopy(self.model[:-2])
        self.right_processor = copy.deepcopy(self.model[:-2])
        self.shared_part = copy.deepcopy(self.model[-2:])

        self.left_processor[0].channels = 16
        self.right_processor[0].channels = 16

        self.left_processor[1].num_bands = 1
        self.right_processor[1].num_bands = 1
        
        for layer_key in self.left_processor[3]._modules['tds_conv_blocks']._modules.keys():
            self.left_processor[3]._modules['tds_conv_blocks']._modules[layer_key].share_hand_weights = False
            self.right_processor[3]._modules['tds_conv_blocks']._modules[layer_key].share_hand_weights = False

            if int(layer_key) % 2 == 0:
                self.left_processor[3]._modules['tds_conv_blocks']._modules[layer_key].width //= 2
                self.right_processor[3]._modules['tds_conv_blocks']._modules[layer_key].width //= 2


        self.left_processor[1]
        del self.model  # Remove the original shared module

    def forward(self, input):
        # Process each hand independently
        left_input, right_input = torch.split(input,1,2)
        left_out = self.left_processor(left_input)
        right_out = self.right_processor(right_input)
        combined = torch.cat([left_out, right_out], dim=-1)
        outputs = self.shared_part(combined)  # Keep the original final layer
        return outputs

    def on_load_checkpoint(self, checkpoint):
        # Modify checkpoint to map "hand_processor" weights to left/right processors
        state_dict = checkpoint["state_dict"]
        new_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("model.") and int(key.split('.')[1])<4:
                print(key)
                # Copy weights to both left and right processors
                new_key_left = key.replace("model.", "left_processor.")
                new_key_right = key.replace("model.", "right_processor.")
                new_state_dict[new_key_left] = value
                new_state_dict[new_key_right] = value
            else:
                new_key_shared = key.replace("model.", "shared_part.")
                new_state_dict[new_key_shared] = value

        checkpoint["state_dict"] = new_state_dict
        
        return super().on_load_checkpoint(checkpoint)




from emg2qwerty.modules import AttnBranchFCBlock, TDSFullyConnectedBlock


class TDSAttnBranchCTCModule(TDSConvCTCModule):
    """Run D: released SplashNet + parallel zero-init attention branches on
    the FC mixing blocks. Warm-starts from a stock checkpoint (strict=False;
    all released tensors load, only br_* stay fresh) and optionally trains
    the branches alone (fc_block/layer_norm stay frozen at released values).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
        share_hand_weights: bool = True,
        split_decoding: bool = False,
        spec_norm: str = "RollingTimeNorm",
        branch_d_model: int = 128,
        branch_nhead: int = 8,
        warm_start_ckpt: str | None = None,
        train_branch_only: bool = False,
        mixing_mode: str = "branch",
    ) -> None:
        super().__init__(
            in_features=in_features,
            mlp_features=mlp_features,
            block_channels=block_channels,
            kernel_width=kernel_width,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            decoder=decoder,
            share_hand_weights=share_hand_weights,
            split_decoding=split_decoding,
            spec_norm=spec_norm,
        )

        num_features = self.NUM_BANDS * mlp_features[-1]
        encoder = self.model[3]
        blocks, pair = [], 0
        for block in encoder.tds_conv_blocks:
            if isinstance(block, TDSFullyConnectedBlock):
                blocks.append(
                    AttnBranchFCBlock(
                        num_features=num_features,
                        share_hand_weights=share_hand_weights,
                        channels=block_channels[pair],
                        d_model=branch_d_model,
                        nhead=branch_nhead,
                        use_fc=(mixing_mode == "branch"),
                    )
                )
                pair += 1
            else:
                blocks.append(block)
        encoder.tds_conv_blocks = nn.Sequential(*blocks)

        if warm_start_ckpt is not None:
            state_dict = torch.load(warm_start_ckpt, map_location="cpu")[
                "state_dict"
            ]
            result = self.load_state_dict(state_dict, strict=False)
            assert not result.unexpected_keys, result.unexpected_keys
            assert all(".br_" in k for k in result.missing_keys), (
                result.missing_keys
            )
            print(
                f"Warm start from {warm_start_ckpt}: "
                f"loaded {len(state_dict)}/{len(state_dict)} tensors, "
                f"{len(result.missing_keys)} branch tensors at fresh init"
            )

        if train_branch_only:
            for param in self.parameters():
                param.requires_grad_(False)
            for block in encoder.tds_conv_blocks:
                if isinstance(block, AttnBranchFCBlock):
                    for name, param in block.named_parameters():
                        if name.startswith("br_"):
                            param.requires_grad_(True)

    def configure_optimizers(self):
        return utils.instantiate_optimizer_and_scheduler(
            [p for p in self.parameters() if p.requires_grad],
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


from emg2qwerty.modules import RNNTPredictionNetwork, RNNTJoiner  # noqa: E402
from torchaudio.functional import rnnt_loss  # noqa: E402


class TDSTransducerModule(TDSConvCTCModule):
    """RNN-T head on the (optionally warm-started, frozen) SplashNet trunk.

    Encoder = parent's model minus final Linear+LogSoftmax (tap at
    num_features = NUM_BANDS * mlp_features[-1]). Blank = charset null class,
    which doubles as the prediction network's BOS token.
    Validation/test decode with streaming transducer greedy (no joint tensor),
    so full-session eval stays memory-safe.
    """

    def __init__(
        self,
        *args,
        pred_embed_dim: int = 256,
        pred_dim: int = 512,
        pred_layers: int = 1,
        joiner_dim: int = 384,
        max_emits_per_frame: int = 8,
        warm_start_ckpt: str | None = None,
        train_head_only: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        num_classes = charset().num_classes
        self.blank = charset().null_class
        enc_dim = self.NUM_BANDS * self.hparams.mlp_features[-1]
        self.prediction = RNNTPredictionNetwork(
            num_classes, pred_embed_dim, pred_dim, pred_layers
        )
        self.joiner = RNNTJoiner(enc_dim, pred_dim, joiner_dim, num_classes)
        self.max_emits_per_frame = max_emits_per_frame

        if warm_start_ckpt is not None:
            ckpt = torch.load(warm_start_ckpt, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            missing, unexpected = self.load_state_dict(state, strict=False)
            assert not unexpected, f"unexpected keys: {unexpected[:5]}"
            assert all(
                k.startswith(("prediction.", "joiner.")) for k in missing
            ), f"non-head keys missing: {[k for k in missing if not k.startswith(('prediction.', 'joiner.'))][:5]}"
            print(
                f"Transducer warm start: loaded {len(state)} tensors, "
                f"{len(missing)} head tensors fresh"
            )

        self.train_head_only = train_head_only
        if train_head_only:
            for p in self.model.parameters():
                p.requires_grad = False
            n_train = sum(
                p.numel() for p in self.parameters() if p.requires_grad
            )
            print(f"train_head_only: {n_train/1e6:.2f}M trainable")

    def train(self, mode: bool = True):
        super().train(mode)
        if getattr(self, "train_head_only", False) and mode:
            self.model.eval()
        return self

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model[:-2](inputs)  # (T, N, enc_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Kept for API compat (returns encoder features, not emissions).
        return self.encode(inputs)

    @torch.no_grad()
    def transducer_greedy(self, enc: torch.Tensor) -> list[np.ndarray]:
        """enc: (T, N, D). Streaming greedy decode, returns label arrays."""
        T, N, _ = enc.shape
        dev = enc.device
        results = []
        for n in range(N):
            labels: list[int] = []
            tok = torch.full((1, 1), self.blank, dtype=torch.long, device=dev)
            pred_out, state = self.prediction(tok)
            for t in range(T):
                emits = 0
                while emits < self.max_emits_per_frame:
                    logits = self.joiner.step(enc[t, n:n + 1], pred_out[:, -1])
                    k = int(logits.argmax(dim=-1))
                    if k == self.blank:
                        break
                    labels.append(k)
                    tok = torch.full((1, 1), k, dtype=torch.long, device=dev)
                    pred_out, state = self.prediction(tok, state)
                    emits += 1
            results.append(np.asarray(labels, dtype=np.int64))
        return results

    def _step(self, phase, batch, *args, **kwargs):
        inputs = batch["inputs"]
        targets = batch["targets"]  # (U_max, N)
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)

        enc = self.encode(inputs)  # (T, N, D)
        T_diff = inputs.shape[0] - enc.shape[0]
        enc_lengths = (input_lengths - T_diff).to(torch.int32)

        tgt = targets.transpose(0, 1).contiguous()  # (N, U_max)
        loss = None
        if phase == "train":
            bos = torch.full(
                (N, 1), self.blank, dtype=torch.long, device=tgt.device
            )
            pred_in = torch.cat([bos, tgt.clamp(min=0)], dim=1)  # (N, U+1)
            pred_out, _ = self.prediction(pred_in)
            logits = self.joiner(
                enc.permute(1, 0, 2).contiguous(), pred_out
            )  # (N, T, U+1, C)
            loss = rnnt_loss(
                logits=logits.float(),
                targets=tgt.to(torch.int32),
                logit_lengths=enc_lengths,
                target_lengths=target_lengths.to(torch.int32),
                blank=self.blank,
                reduction="mean",
            )
            self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        else:
            metrics = self.metrics[f"{phase}_metrics"]
            hyps = self.transducer_greedy(enc)
            t_np = targets.detach().cpu().numpy()
            tl_np = target_lengths.detach().cpu().numpy()
            for i in range(N):
                target = LabelData.from_labels(t_np[: tl_np[i], i])
                metrics.update(
                    prediction=LabelData.from_labels(hyps[i]), target=target
                )
            loss = torch.zeros((), device=enc.device)
            self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def on_train_epoch_end(self) -> None:
        # Transducer training tracks loss only; the train CER metric is never
        # updated, and computing an empty metric divides by zero.
        pass

    def configure_optimizers(self):
        return utils.instantiate_optimizer_and_scheduler(
            [p for p in self.parameters() if p.requires_grad],
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )


from emg2qwerty.modules import HATJoiner  # noqa: E402


def _charset_label_names() -> list[str]:
    cs = charset()
    ncls = cs.num_classes if hasattr(cs, "num_classes") else 99
    names = []
    for lab in range(ncls):
        name = None
        for attr in ("label_to_key", "keys", "_keys"):
            try:
                k = getattr(cs, attr)
                v = k(lab) if callable(k) else k[lab]
                name = str(v)
                break
            except Exception:
                continue
        names.append(name if name is not None else f"<{lab}>")
    return names


class TDSHATModule(TDSTransducerModule):
    """Hierarchical Anatomical Transducer: TDSTransducerModule with the
    flat joiner replaced by the anatomical-tree HATJoiner. Trunk warm-start
    and head-only freezing inherited from the parent."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        enc_dim = self.NUM_BANDS * self.hparams.mlp_features[-1]
        pred_dim = self.hparams.get("pred_dim", 512)
        self.joiner = HATJoiner(
            enc_dim, pred_dim, charset().num_classes, _charset_label_names()
        )
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"HAT head params trainable: {n_train/1e6:.2f}M")


class EMAWeightAveraging(pl.Callback):
    """Exponential moving average of model weights, updated once per
    optimizer step. Validation, checkpointing, and the post-fit test all see
    the EMA weights: EMA is swapped in at validation start and the raw
    training weights are restored at the next train-epoch start (so saved
    checkpoints contain EMA weights).

    NOTE: EMA state is not persisted across auto-resume; a resumed run
    restarts the average from the checkpoint weights (acceptable drift).
    """

    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.ema = None
        self.raw = None
        self._last_step = -1

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        step = trainer.global_step
        if step == self._last_step:
            return
        self._last_step = step
        with torch.no_grad():
            if self.ema is None:
                self.ema = {
                    k: v.detach().clone()
                    for k, v in pl_module.state_dict().items()
                    if v.dtype.is_floating_point
                }
            else:
                for k, v in pl_module.state_dict().items():
                    if k in self.ema:
                        self.ema[k].mul_(self.decay).add_(
                            v.detach(), alpha=1.0 - self.decay)

    def _load(self, pl_module, tensors):
        with torch.no_grad():
            sd = pl_module.state_dict()
            for k, v in tensors.items():
                sd[k].copy_(v)

    def on_validation_epoch_start(self, trainer, pl_module):
        if self.ema is None or trainer.sanity_checking:
            return
        self.raw = {k: v.detach().clone()
                    for k, v in pl_module.state_dict().items()
                    if k in self.ema}
        self._load(pl_module, self.ema)

    def on_train_epoch_start(self, trainer, pl_module):
        if self.raw is not None:
            self._load(pl_module, self.raw)
            self.raw = None
