import math
from typing import Any, ClassVar, Optional, Sequence, Tuple

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torchmetrics import MetricCollection

from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import MultiBandRotationInvariantMLP, SpectrogramNorm

# NOTE
### Append this modules into original modules.py of emg2qwerty repository to run.


class PositionalEncoding(nn.Module):
    """
    Implements the classic sinusoidal positional encoding.
    
    Args:
        d_model (int): Dimensionality of the input features.
        dropout (float): Dropout rate applied after adding the positional encoding.
        max_len (int): Maximum sequence length.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (T, N, d_model)
        
        Returns:
            Tensor: Output tensor with positional encodings added.
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor: # This is key masking (default)
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask


class _ConvolutionModule(torch.nn.Module):
    """Conformer convolution module.

    Args:
        input_dim (int): input dimension.
        num_channels (int): number of depthwise convolution layer input channels.
        depthwise_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        bias (bool, optional): indicates whether to add bias term to each convolution layer. (Default: ``False``)
        use_group_norm (bool, optional): use GroupNorm rather than BatchNorm. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        num_channels: int,
        depthwise_kernel_size: int,
        dropout: float = 0.0,
        bias: bool = False,
        use_group_norm: bool = False,
    ) -> None:
        super().__init__()
        if (depthwise_kernel_size - 1) % 2 != 0:
            raise ValueError("depthwise_kernel_size must be odd to achieve 'SAME' padding.")
        self.layer_norm = torch.nn.LayerNorm(input_dim)
        self.sequential = torch.nn.Sequential(
            torch.nn.Conv1d(
                input_dim,
                2 * num_channels,
                1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.GLU(dim=1),
            torch.nn.Conv1d(
                num_channels,
                num_channels,
                depthwise_kernel_size,
                stride=1,
                padding=(depthwise_kernel_size - 1) // 2,
                groups=num_channels,
                bias=bias,
            ),
            torch.nn.GroupNorm(num_groups=1, num_channels=num_channels)
            if use_group_norm
            else torch.nn.BatchNorm1d(num_channels),
            torch.nn.SiLU(),
            torch.nn.Conv1d(
                num_channels,
                input_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, D)`.

        Returns:
            torch.Tensor: output, with shape `(B, T, D)`.
        """
        x = self.layer_norm(input)
        x = x.transpose(1, 2)
        x = self.sequential(x)
        return x.transpose(1, 2)


class _FeedForwardModule(torch.nn.Module):
    r"""Positionwise feed forward layer.

    Args:
        input_dim (int): input dimension.
        hidden_dim (int): hidden dimension.
        dropout (float, optional): dropout probability. (Default: 0.0)
    """

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, input_dim, bias=True),
            torch.nn.Dropout(dropout),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): with shape `(*, D)`.

        Returns:
            torch.Tensor: output, with shape `(*, D)`.
        """
        return self.sequential(input)


class ConformerLayer(torch.nn.Module):
    """Conformer layer that constitutes Conformer.

    Args:
        input_dim (int): input dimension.
        ffn_dim (int): hidden layer dimension of feedforward network.
        num_attention_heads (int): number of attention heads.
        depthwise_conv_kernel_size (int): kernel size of depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)
    """

    def __init__(
        self,
        input_dim: int,
        ffn_dim: int,
        num_attention_heads: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ) -> None:
        super().__init__()
        

        self.ffn1 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)

        self.self_attn_layer_norm = torch.nn.LayerNorm(input_dim)

        self.self_attn = torch.nn.MultiheadAttention(input_dim, num_attention_heads, dropout=dropout)
        self.self_attn_dropout = torch.nn.Dropout(dropout)

        self.conv_module = _ConvolutionModule(
            input_dim=input_dim,
            num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
            bias=True,
            use_group_norm=use_group_norm,
        )

        self.ffn2 = _FeedForwardModule(input_dim, ffn_dim, dropout=dropout)
        self.final_layer_norm = torch.nn.LayerNorm(input_dim)
        self.convolution_first = convolution_first

    def _apply_convolution(self, input: torch.Tensor) -> torch.Tensor:
        residual = input
        input = input.transpose(0, 1)
        input = self.conv_module(input)
        input = input.transpose(0, 1)
        input = residual + input
        return input

    # Changed this part
    def forward(self, input: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None, is_causal: bool = False) -> torch.Tensor:
        r"""
        Args:
            input (torch.Tensor): input, with shape `(T, B, D)`.
            key_padding_mask (torch.Tensor or None): key padding mask to use in self attention layer.

        Returns:
            torch.Tensor: output, with shape `(T, B, D)`.
        """
        residual = input
        x = self.ffn1(input)
        # print("After FFN1:", torch.isnan(x).sum())
        # print('After FFN1:', torch.max(x), torch.min(x))

        x = x * 0.5 + residual

        if self.convolution_first:
            x = self._apply_convolution(x)
            # print("After Convolution:", torch.isnan(x).sum())

        residual = x

        # print('Before self_attn_layer_norm:', torch.max(x), torch.min(x))
        x = self.self_attn_layer_norm(x)
        # print("After self_attn_layer_norm:", torch.isnan(x).sum())
        # print("Pre-attn norm: max", x.max(), "min", x.min(), "std", x.std())


        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask, #attn_mask, # Add in attention mask and is_causal
            is_causal=is_causal, #is_causal,
            need_weights=False,
        )

        # print("Post-attn: max", x.max(), "min", x.min(), "std", x.std())

        # print("After self-attn:", torch.isnan(x).sum())

        
        x = self.self_attn_dropout(x)
        x = x + residual

        if not self.convolution_first:
            x = self._apply_convolution(x)
        # print("After Convolution:", torch.isnan(x).sum())

        residual = x
        x = self.ffn2(x)
        # print('After FFN2:', torch.isnan(x).sum())
        x = x * 0.5 + residual

        x = self.final_layer_norm(x)
        # print("After final_layer_norm:", torch.isnan(x).sum())
        return x

class Conformer(torch.nn.Module):
    r"""Conformer architecture introduced in
    *Conformer: Convolution-augmented Transformer for Speech Recognition*
    :cite:`gulati2020conformer`.

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Conformer layer.
        ffn_dim (int): hidden layer dimension of feedforward networks.
        num_layers (int): number of Conformer layers to instantiate.
        depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)

    Examples:
        >>> conformer = Conformer(
        >>>     input_dim=80,
        >>>     num_heads=4,
        >>>     ffn_dim=128,
        >>>     num_layers=4,
        >>>     depthwise_conv_kernel_size=31,
        >>> )
        >>> lengths = torch.randint(1, 400, (10,))  # (batch,)
        >>> input = torch.rand(10, int(lengths.max()), input_dim)  # (batch, num_frames, input_dim)
        >>> output = conformer(input, lengths)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ):
        super().__init__()

        self.conformer_layers = torch.nn.ModuleList(
            [
                ConformerLayer(
                    input_dim,
                    ffn_dim,
                    num_heads,
                    depthwise_conv_kernel_size,
                    dropout=dropout,
                    use_group_norm=use_group_norm,
                    convolution_first=convolution_first,
                )
                for _ in range(num_layers)
            ]
        )

        self.positional_encoding = PositionalEncoding(input_dim, dropout=dropout)

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input (torch.Tensor): with shape `(B, T, input_dim)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor)
                torch.Tensor
                    output frames, with shape `(B, T, input_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """


        # permute input (input is (T, N, num_features), where conformer takes in (N, T, num_features))
        input = input.transpose(0, 1)
        input = self.positional_encoding(input)

        encoder_padding_mask = _lengths_to_padding_mask(lengths)
        
        x = input.transpose(0, 1)
        for layer in self.conformer_layers:

            x = layer(x, encoder_padding_mask, is_causal=False)
        return x, lengths # output (T, B, num_features)
    

# NOTE
### Append this part into original lightning.py of emg2qwerty repository to run.

Acausal_Conformer = Conformer


class ConformerModule(pl.LightningModule):
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
        frontend: Optional[DictConfig] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Optional GPU front-end (log-spectrogram + SpecAugment on the batch). Applied in
        # _step only when the batch is raw EMG (4-D); spectrogram batches (5-D) pass through,
        # so checkpoints work with either the CPU or the GPU transform path.
        self.frontend = instantiate(frontend) if frontend is not None else None

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)

        self.norm = SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS)
        self.mlp = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=mlp_features,
            num_bands=self.NUM_BANDS,
        )
        self.flatten = nn.Flatten(start_dim=2)

        # Reduce input feature before inputting into conformer

        self.conformer = Acausal_Conformer(
                input_dim=num_features,
                num_heads=6,
                ffn_dim=256,
                num_layers=3,
                depthwise_conv_kernel_size=kernel_width, # original = 32, but it only allows odd numbers due to "same" padding
                convolution_first = False # This is default
            )
        
        self.output_layer = nn.Sequential(
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )


        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=charset().null_class)
        self.alpha = 1.0

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

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.norm(inputs)
        x = self.mlp(x)
        x = self.flatten(x)

        with torch.autograd.set_detect_anomaly(False):
            x, _ = self.conformer(x, lengths)

        x = self.output_layer(x)

        return x

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        if self.frontend is not None and inputs.dim() == 4:  # raw EMG (T, N, bands, channels)
            inputs, input_lengths = self.frontend(inputs, input_lengths, self.training)

        emissions = self.forward(inputs, input_lengths) 
        # print("Any NaN in output?", torch.isnan(emissions).any())

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        # CTC loss
        ctc_loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Cross-entropy loss
        ce_loss = 0

        loss = ctc_loss

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

        # Log losses
        self.log(f"{phase}/ctc_loss", ctc_loss, batch_size=N, sync_dist=True)
        self.log(f"{phase}/ce_loss", ce_loss, batch_size=N, sync_dist=True)
        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss
        # return predictions, loss, metrics
    
    
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


# ===== Causal variant (additive; reviewer #4) ==========================
# Same architecture and weights layout as ConformerModule; only three things
# change: (1) self-attention is masked to past/present frames, (2) the depthwise
# convolution is left-padded (kernel-1) instead of same, (3) the datamodule
# uses padding [1800, 0] (no future context) via config/model/conformer_ctc_causal.yaml.

class _CausalDepthwiseConv1d(torch.nn.Module):
    def __init__(self, conv: torch.nn.Conv1d) -> None:
        super().__init__()
        self.conv = conv
        self.left_pad = conv.kernel_size[0] - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, T)
        return self.conv(torch.nn.functional.pad(x, (self.left_pad, 0)))


class _CausalConvolutionModule(_ConvolutionModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        dw = self.sequential[2]
        causal = torch.nn.Conv1d(dw.in_channels, dw.out_channels, dw.kernel_size[0],
                                 stride=1, padding=0, groups=dw.groups, bias=dw.bias is not None)
        self.sequential[2] = _CausalDepthwiseConv1d(causal)


class CausalConformerLayer(ConformerLayer):
    def __init__(self, input_dim, ffn_dim, num_attention_heads, depthwise_conv_kernel_size,
                 dropout: float = 0.0, use_group_norm: bool = False, convolution_first: bool = False) -> None:
        super().__init__(input_dim, ffn_dim, num_attention_heads, depthwise_conv_kernel_size,
                         dropout=dropout, use_group_norm=use_group_norm, convolution_first=convolution_first)
        self.conv_module = _CausalConvolutionModule(
            input_dim=input_dim, num_channels=input_dim,
            depthwise_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout, bias=True, use_group_norm=use_group_norm)


class CausalConformer(Conformer):
    def __init__(self, input_dim, num_heads, ffn_dim, num_layers, depthwise_conv_kernel_size,
                 dropout: float = 0.0, use_group_norm: bool = False, convolution_first: bool = False):
        super().__init__(input_dim, num_heads, ffn_dim, num_layers, depthwise_conv_kernel_size,
                         dropout=dropout, use_group_norm=use_group_norm, convolution_first=convolution_first)
        self.conformer_layers = torch.nn.ModuleList([
            CausalConformerLayer(input_dim, ffn_dim, num_heads, depthwise_conv_kernel_size,
                                 dropout=dropout, use_group_norm=use_group_norm,
                                 convolution_first=convolution_first)
            for _ in range(num_layers)])

    def forward(self, input: torch.Tensor, lengths: torch.Tensor):
        # identical preprocessing to Conformer.forward; only the attention mask differs
        input = input.transpose(0, 1)
        input = self.positional_encoding(input)
        encoder_padding_mask = _lengths_to_padding_mask(lengths)
        x = input.transpose(0, 1)  # (T, B, D)
        T = x.size(0)
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        for layer in self.conformer_layers:
            x = layer(x, encoder_padding_mask, attn_mask=causal_mask, is_causal=True)
        return x, lengths


class CausalConformerModule(ConformerModule):
    def __init__(self, in_features, mlp_features, block_channels, kernel_width,
                 optimizer, lr_scheduler, decoder) -> None:
        super().__init__(in_features, mlp_features, block_channels, kernel_width,
                         optimizer, lr_scheduler, decoder)
        num_features = self.NUM_BANDS * mlp_features[-1]
        self.conformer = CausalConformer(
            input_dim=num_features, num_heads=6, ffn_dim=256, num_layers=3,
            depthwise_conv_kernel_size=kernel_width, convolution_first=False)


# ===== 100 ms protocol-matched variant (additive) =======================
# Future context unified to the benchmark's 100 ms budget: each of the 3
# blocks attends LOOKAHEAD frames ahead (banded mask), compounding to
# 3*4 = 12 frames = 96 ms at 125 Hz; convolutions are causal (0 future).
# Long inputs (the benchmark's un-windowed test sessions) are processed in
# deployment windows -- 4 s chunks with ~900 ms past and 96 ms future
# context -- because a T x T attention mask is infeasible at session scale.

class LookaheadConformer(CausalConformer):
    LOOKAHEAD: ClassVar[int] = 4  # frames per layer at 125 Hz (8 ms/frame)

    def forward(self, input: torch.Tensor, lengths: torch.Tensor):
        input = input.transpose(0, 1)
        input = self.positional_encoding(input)
        encoder_padding_mask = _lengths_to_padding_mask(lengths)
        x = input.transpose(0, 1)  # (T, B, D)
        T = x.size(0)
        band_mask = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device),
            diagonal=self.LOOKAHEAD + 1,
        )
        for layer in self.conformer_layers:
            x = layer(x, encoder_padding_mask, attn_mask=band_mask, is_causal=False)
        return x, lengths


class LookaheadConformerModule(ConformerModule):
    CHUNK: ClassVar[int] = 500   # 4 s at 125 Hz
    L_CTX: ClassVar[int] = 112   # ~900 ms past context per chunk
    R_CTX: ClassVar[int] = 12    # 96 ms future context (= 3 layers x 4 frames)

    def __init__(self, in_features, mlp_features, block_channels, kernel_width,
                 optimizer, lr_scheduler, decoder) -> None:
        super().__init__(in_features, mlp_features, block_channels, kernel_width,
                         optimizer, lr_scheduler, decoder)
        num_features = self.NUM_BANDS * mlp_features[-1]
        self.conformer = LookaheadConformer(
            input_dim=num_features, num_heads=6, ffn_dim=256, num_layers=3,
            depthwise_conv_kernel_size=kernel_width, convolution_first=False)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.norm(inputs)
        x = self.mlp(x)
        x = self.flatten(x)  # (T, N, D)
        T = x.shape[0]
        if T <= 700:  # training/val windows (625 frames) take the single pass
            x, _ = self.conformer(x, lengths)
        else:
            outs = []
            for s in range(0, T, self.CHUNK):
                a = max(0, s - self.L_CTX)
                b = min(T, s + self.CHUNK + self.R_CTX)
                seg = x[a:b]
                seg_len = (lengths - a).clamp(min=1, max=b - a)
                y, _ = self.conformer(seg, seg_len)
                outs.append(y[s - a : (s - a) + min(self.CHUNK, T - s)])
            x = torch.cat(outs, dim=0)
        return self.output_layer(x)


# ===== Causal chunked inference (additive) =============================
# The benchmark's un-windowed test sessions make the T x T causal mask
# infeasible (tens of GB at session scale). Long inputs are processed in
# deployment windows: 4 s chunks with ~900 ms past context and no future
# context (causal). Training/val windows (<= 700 frames) keep the single pass.

def _causal_chunked_forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    x = self.norm(inputs)
    x = self.mlp(x)
    x = self.flatten(x)  # (T, N, D)
    T = x.shape[0]
    if T <= 700:
        x, _ = self.conformer(x, lengths)
    else:
        CHUNK, L_CTX = 500, 112
        outs = []
        for s in range(0, T, CHUNK):
            a = max(0, s - L_CTX)
            b = min(T, s + CHUNK)
            seg = x[a:b]
            seg_len = (lengths - a).clamp(min=1, max=b - a)
            y, _ = self.conformer(seg, seg_len)
            outs.append(y[s - a : (s - a) + min(CHUNK, T - s)])
        x = torch.cat(outs, dim=0)
    return self.output_layer(x)

CausalConformerModule.forward = _causal_chunked_forward


# ===== Tail-aligned streaming inference (additive) ======================
# Window-level causality: the ACAUSAL encoder (unchanged weights) is run on
# sliding windows that END at the emission block, so attention is
# bidirectional within the window but the stream-level lookahead is bounded
# by the emission stride (12 frames = 96 ms <= the benchmark's 100 ms
# budget). No retraining: this evaluates an existing acausal checkpoint as
# it would be deployed with a 96 ms output delay. Training/short inputs
# take the ordinary single pass. Segments are right-padded and masked via
# lengths; the emission block is read from each segment's true tail.

class TailStreamConformerModule(ConformerModule):
    SEG: ClassVar[int] = 624     # frames per window (~5 s of context)
    STRIDE: ClassVar[int] = 12   # emission block; stream lookahead <= 96 ms
    SEG_BATCH: ClassVar[int] = 32

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.norm(inputs)
        x = self.mlp(x)
        x = self.flatten(x)  # (T, N, D)
        T, N, D = x.shape
        if self.training or T <= 700:
            x, _ = self.conformer(x, lengths)
            return self.output_layer(x)
        import os as _os
        SEG = int(_os.environ.get("TAIL_SEG", self.SEG))
        SEGB = int(_os.environ.get("TAIL_SEG_BATCH", self.SEG_BATCH))
        outs = []
        for n in range(N):
            pe = self.conformer.positional_encoding
            if hasattr(pe, "idx"): pe.idx = n
            Tn = int(lengths[n].item())
            xn = x[:Tn, n]  # (Tn, D)
            blocks = []
            starts = list(range(0, Tn, self.STRIDE))
            for i in range(0, len(starts), SEGB):
                group = starts[i : i + SEGB]
                segs, keeps = [], []
                import os as _os
                delay = int(_os.environ.get("TAIL_DELAY", "0"))
                offs = []
                for s in group:
                    b = min(s + self.STRIDE, Tn)   # emission block [s, b)
                    we = min(b + delay, Tn)        # window ends delay frames later
                    a = max(0, we - SEG)
                    segs.append(xn[a:we])
                    keeps.append(b - s)
                    offs.append((s - a, b - a))
                lens = [seg.shape[0] for seg in segs]
                if len(set(lens)) > 1:
                    # unequal windows (session start): run each alone so no padding leaks through conv layers
                    for seg, (lo, hi) in zip(segs, offs):
                        y1, _ = self.conformer(seg.unsqueeze(1), torch.tensor([seg.shape[0]], dtype=lengths.dtype, device=lengths.device))
                        blocks.append(y1[lo:hi, 0].clone())
                    continue
                L = max(lens)
                batch = xn.new_zeros(L, len(segs), D)
                for j, seg in enumerate(segs):
                    batch[: lens[j], j] = seg
                seg_len = torch.tensor(lens, dtype=lengths.dtype, device=lengths.device)
                y, _ = self.conformer(batch, seg_len)
                for j, k in enumerate(keeps):
                    lo, hi = offs[j]
                    blocks.append(y[lo:hi, j].clone())
            yn = torch.cat(blocks, dim=0)  # (Tn, D)
            if Tn < T:
                yn = torch.cat([yn, yn.new_zeros(T - Tn, D)], dim=0)
            outs.append(yn)
        x = torch.stack(outs, dim=1)  # (T, N, D)
        return self.output_layer(x)


# ===== Fixed-index positional offset for windowed inference (additive) ===
# The public Conformer code applies PositionalEncoding after transposing to
# (N, T, D), so pe[:N] is added along the BATCH axis: batch element i receives
# the constant vector pe[i] at every time step and no temporal position at all.
# Un-windowed evaluation therefore gives session n the constant pe[n]. To keep
# windowed inference on the same convention, every window of session n gets
# pe[n] rather than pe[window index within the batch].

class _FixedIndexPE(PositionalEncoding):
    idx: int = 0
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[self.idx])

def _install_fixed_pe(module: ConformerModule) -> None:
    old = module.conformer.positional_encoding
    new = _FixedIndexPE(old.pe.shape[2], dropout=old.dropout.p, max_len=old.pe.shape[0])
    new.pe = old.pe
    module.conformer.positional_encoding = new

_tail_forward_orig = TailStreamConformerModule.forward
def _tail_forward_fixed(self, inputs, lengths):
    import os as _os
    # TAIL_PE=cycle keeps the public code's behaviour (window i in a batch gets pe[i]); used
    # only for the fixed-vs-cycling comparison reported in the paper (+0.03 greedy CER).
    if _os.environ.get("TAIL_PE", "fixed") != "cycle" and not isinstance(self.conformer.positional_encoding, _FixedIndexPE):
        _install_fixed_pe(self)
    return _tail_forward_orig(self, inputs, lengths)
TailStreamConformerModule.forward = _tail_forward_fixed
