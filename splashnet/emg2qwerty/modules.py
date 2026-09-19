# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence

import torch
from torch import nn
import torch.nn.functional as F
import pdb


class SpectrogramNorm(nn.Module):
    """A `torch.nn.Module` that applies 2D batch normalization over spectrogram
    per electrode channel per band. Inputs must be of shape
    (T, N, num_bands, electrode_channels, frequency_bins).

    With left and right bands and 16 electrode channels per band, spectrograms
    corresponding to each of the 2 * 16 = 32 channels are normalized
    independently using `nn.BatchNorm2d` such that stats are computed
    over (N, freq, time) slices.

    Args:
        channels (int): Total number of electrode channels across bands
            such that the normalization statistics are calculated per channel.
            Should be equal to num_bands * electrode_chanels.
    """

    def __init__(self, channels: int, spec_norm_type: str) -> None:
        super().__init__()
        self.channels = channels
        self.spec_norm_type = spec_norm_type
        if self.spec_norm_type == 'RollingTimeNorm':
            self.spec_norm = RollingTimeNorm(eps = 0.1) 
        elif self.spec_norm_type == 'BatchNorm2d': 
            self.batch_norm = nn.BatchNorm2d(channels)
        elif self.spec_norm_type == 'None':
            self.spec_norm = None
        else:
            raise ValueError(f"Invalid spec_norm_type, got {spec_norm_type}")


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, bands, C, freq = inputs.shape  # (T, N, bands=2, C=16, freq)
        #assert self.channels == bands * C # probably best to just get rid of this assertion

        if self.spec_norm_type == 'RollingTimeNorm':
            x = self.spec_norm(inputs) 
            return x
        
        elif self.spec_norm_type == 'BatchNorm2d':
            x = inputs.movedim(0, -1)  # (N, bands=2, C=16, freq, T)
            x = x.reshape(N, bands * C, freq, T)
            x = self.batch_norm(x)
            x = x.reshape(N, bands, C, freq, T)
            return x.movedim(-1, 0)  # (T, N, bands=2, C=16, freq)
        
        if self.spec_norm is None:
            return inputs



class RotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that takes an input tensor of shape
    (T, N, electrode_channels, ...) corresponding to a single band, applies
    an MLP after shifting/rotating the electrodes for each positional offset
    in ``offsets``, and pools over all the outputs.

    Returns a tensor of shape (T, N, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input of
            shape (T, N, C, ...), this should be equal to C * ... (that is,
            the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
    ) -> None:
        super().__init__()

        assert len(mlp_features) > 0
        mlp: list[nn.Module] = []
        for out_features in mlp_features:
            mlp.extend(
                [
                    nn.Linear(in_features, out_features),
                    nn.ReLU(),
                ]
            )
            in_features = out_features
        self.mlp = nn.Sequential(*mlp)

        assert pooling in {"max", "mean"}, f"Unsupported pooling: {pooling}"
        self.pooling = pooling

        self.offsets = offsets if len(offsets) > 0 else (0,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs  # (T, N, C, ...)

        # Create a new dim for band rotation augmentation with each entry
        # corresponding to the original tensor with its electrode channels
        # shifted by one of ``offsets``:
        # (T, N, C, ...) -> (T, N, rotation, C, ...)
        x = torch.stack([x.roll(offset, dims=2) for offset in self.offsets], dim=2)

        # Flatten features and pass through MLP:
        # (T, N, rotation, C, ...) -> (T, N, rotation, mlp_features[-1])
        x = self.mlp(x.flatten(start_dim=3))

        # Pool over rotations:
        # (T, N, rotation, mlp_features[-1]) -> (T, N, mlp_features[-1])
        if self.pooling == "max":
            return x.max(dim=2).values
        else:
            return x.mean(dim=2)


class MultiBandRotationInvariantMLP(nn.Module):
    """A `torch.nn.Module` that applies a separate instance of
    `RotationInvariantMLP` per band for inputs of shape
    (T, N, num_bands, electrode_channels, ...).

    Returns a tensor of shape (T, N, num_bands, mlp_features[-1]).

    Args:
        in_features (int): Number of input features to the MLP. For an input
            of shape (T, N, num_bands, C, ...), this should be equal to
            C * ... (that is, the flattened size from the channel dim onwards).
        mlp_features (list): List of integers denoting the number of
            out_features per layer in the MLP.
        pooling (str): Whether to apply mean or max pooling over the outputs
            of the MLP corresponding to each offset. (default: "mean")
        offsets (list): List of positional offsets to shift/rotate the
            electrode channels by. (default: ``(-1, 0, 1)``).
        num_bands (int): ``num_bands`` for an input of shape
            (T, N, num_bands, C, ...). (default: 2)
        stack_dim (int): The dimension along which the left and right data
            are stacked. (default: 2)
    """

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        pooling: str = "mean",
        offsets: Sequence[int] = (-1, 0, 1),
        num_bands: int = 2,
        stack_dim: int = 2,
        share_hand_weights: bool = False,
    ) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.stack_dim = stack_dim
        self.share_hand_weights = share_hand_weights

        # One MLP per band
        self.mlps = nn.ModuleList(
            [
                RotationInvariantMLP(
                    in_features=in_features,
                    mlp_features=mlp_features,
                    pooling=pooling,
                    offsets=offsets,
                )
                for _ in range(num_bands if share_hand_weights == False else 1)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        assert inputs.shape[self.stack_dim] == self.num_bands

        inputs_per_band = inputs.unbind(self.stack_dim)
        if self.share_hand_weights == False:
            outputs_per_band = [
                mlp(_input) for mlp, _input in zip(self.mlps, inputs_per_band)
            ]
        else:
            outputs_per_band = [
                self.mlps[0](_input) for _input in inputs_per_band
            ]

        return torch.stack(outputs_per_band, dim=self.stack_dim)

    

class TDSConv2dBlock(nn.Module):
    """A 2D temporal convolution block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        channels (int): Number of input and output channels. For an input of
            shape (T, N, num_features), the invariant we want is
            channels * width = num_features.
        width (int): Input width. For an input of shape (T, N, num_features),
            the invariant we want is channels * width = num_features.
        kernel_width (int): The kernel size of the temporal convolution.
    """

    def __init__(self, channels: int, width: int, kernel_width: int, share_hand_weights: bool) -> None:
        super().__init__()
        self.channels = channels
        self.width = width
        self.share_hand_weights = share_hand_weights

        self.conv2d = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_width),
        )
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(channels * width // (2 if share_hand_weights else 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T_in, N, C = inputs.shape
        if self.share_hand_weights:
            # Reshape to process both halves in parallel
            inputs = inputs.view(T_in, N*2, C//2)  # [T, 2*N, C//2]

            # T(2N)(C/2) -> (2N)(C/2)T -> (2N)c(w/2)T ### w is half of what it would be without shared weights
            x = inputs.movedim(0, -1).reshape(N*2, self.channels, self.width//2, T_in)
            x = self.conv2d(x)
            x = self.relu(x)
            x = x.reshape(N*2, C//2, -1).movedim(-1, 0)  # (2N)c(w/2)T -> (2N)(C/2)T -> T(2N)(C/2)
                
            # Skip connection afer downsampling 
            T_out = x.shape[0]
            x = x + inputs[-T_out:]
            # Layer norm over C/2
            x = self.layer_norm(x)
            
            return x.view(T_out, N, C)
        
        else:
            # TNC -> NCT -> NcwT
            x = inputs.movedim(0, -1).reshape(N, self.channels, self.width, T_in)
            x = self.conv2d(x)
            x = self.relu(x)
            x = x.reshape(N, C, -1).movedim(-1, 0)  # NcwT -> NCT -> TNC

            # Skip connection after downsampling
            T_out = x.shape[0]
            x = x + inputs[-T_out:]

            # Layer norm over C
            return self.layer_norm(x)  # TNC
        

class TDSFullyConnectedBlock(nn.Module):
    """A fully connected block as per "Sequence-to-Sequence Speech
    Recognition with Time-Depth Separable Convolutions, Hannun et al"
    (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        share_hand_weights (bool): whether to share weights across hands
            and treat each hand independently
    """
    def __init__(self, num_features: int, share_hand_weights: bool) -> None:
        super().__init__()
        self.num_features = num_features
        self.share_hand_weights = share_hand_weights

        if self.share_hand_weights:
            half_features = num_features // 2
            self.fc_block = nn.Sequential(
                nn.Linear(half_features, half_features),
                nn.ReLU(),
                nn.Linear(half_features, half_features)
            )
            self.layer_norm = nn.LayerNorm(half_features)
        else:
            self.fc_block = nn.Sequential(
                nn.Linear(num_features, num_features),
                nn.ReLU(),
                nn.Linear(num_features, num_features)
            )
            self.layer_norm = nn.LayerNorm(num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.share_hand_weights:
            # Reshape to process both halves in parallel
            T, N, C = inputs.shape
            x = inputs.view(T, N*2, C//2)  # [T, 2*N, C//2]
            
            # Process both halves simultaneously
            x = self.fc_block(x) + x  # Residual connection
            x = self.layer_norm(x)
            
            # Restore original shape
            return x.view(T, N, C)
        else:
            x = self.fc_block(inputs)
            return self.layer_norm(x + inputs)
        

class TDSConvEncoder(nn.Module):
    """A time depth-separable convolutional encoder composing a sequence
    of `TDSConv2dBlock` and `TDSFullyConnectedBlock` as per
    "Sequence-to-Sequence Speech Recognition with Time-Depth Separable
    Convolutions, Hannun et al" (https://arxiv.org/abs/1904.02619).

    Args:
        num_features (int): ``num_features`` for an input of shape
            (T, N, num_features).
        block_channels (list): A list of integers indicating the number
            of channels per `TDSConv2dBlock`.
        kernel_width (int): The kernel size of the temporal convolutions.
    """

    def __init__(
        self,
        num_features: int,
        block_channels: Sequence[int] = (24, 24, 24, 24),
        kernel_width: int = 32,
        share_hand_weights: bool = False,
    ) -> None:
        super().__init__()

        assert len(block_channels) > 0
        tds_conv_blocks: list[nn.Module] = []
        for channels in block_channels:
            assert (
                num_features % channels == 0
            ), "block_channels must evenly divide num_features"
            tds_conv_blocks.extend(
                [
                    TDSConv2dBlock(channels, num_features // channels, kernel_width, share_hand_weights),
                    TDSFullyConnectedBlock(num_features, share_hand_weights),
                ]
            )
        self.tds_conv_blocks = nn.Sequential(*tds_conv_blocks)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.tds_conv_blocks(inputs)  # (T, N, num_features)


class RollingTimeNorm(nn.Module):
    r"""Causally normalize a 5D tensor (T, N, bands, channels, freq) along the time axis.

    For each sample (N) and for every (band, channel, frequency) location, this module
    normalizes the data as follows:

    - For the first `warmup` time steps (default 125), the statistics (mean and standard deviation)
      are computed over the entire warmup period (i.e. time indices 0 to 124) and are used for
      every time step in that period.
    - For any subsequent time step \(t \geq \texttt{warmup}\), the statistics are computed over
      all time steps from 0 up through \(t\). For example, at the 300th time step the normalization
      uses the statistics from the first 300 time steps.

    A small epsilon is added inside the square root for numerical stability.
    
    Args:
        warmup (int): Number of time bins to use for warmup (default: 125).
        eps (float): A small constant added for numerical stability.
    """
    def __init__(self, warmup: int = 125, eps: float = 1e-5):
        super().__init__()
        self.warmup = warmup
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (T, N, bands, channels, freq).

        Returns:
            torch.Tensor: Normalized tensor of the same shape.
        """
        T, N, bands, channels, freq = x.shape
        device, dtype = x.device, x.dtype

        # Compute cumulative sum and cumulative sum of squares along time (dim=0).
        # These have shape (T, N, bands, channels, freq).
        cumsum = x.cumsum(dim=0)
        cumsum2 = (x ** 2).cumsum(dim=0)

        # Create a denominator tensor: [1, 2, 3, ..., T] reshaped for broadcasting.
        time_range = torch.arange(1, T + 1, device=device, dtype=dtype).view(T, 1, 1, 1, 1)

        # Compute the cumulative mean and mean-of-squares.
        cum_mean = cumsum / time_range
        cum_mean2 = cumsum2 / time_range
        cum_std = torch.sqrt(cum_mean2 - cum_mean ** 2 + self.eps)

        # For the warmup period, override the cumulative stats with those computed from
        # the entire warmup period.
        if T >= self.warmup:
            # Use the statistics at time index (warmup - 1) which uses the first 'warmup' time steps.
            warmup_mean = cum_mean[self.warmup - 1]  # shape: (N, bands, channels, freq)
            warmup_mean2 = cum_mean2[self.warmup - 1]
            warmup_std = torch.sqrt(warmup_mean2 - warmup_mean ** 2 + self.eps)

            # For t in [0, warmup), assign the warmup statistics.
            cum_mean[:self.warmup] = warmup_mean.unsqueeze(0).expand(self.warmup, N, bands, channels, freq)
            cum_std[:self.warmup] = warmup_std.unsqueeze(0).expand(self.warmup, N, bands, channels, freq)
        else:
            # If the entire sequence is shorter than the warmup period,
            # use the stats computed from all available time steps.
            warmup_mean = cum_mean[-1]
            warmup_std = torch.sqrt(cum_mean2[-1] - warmup_mean ** 2 + self.eps)
            cum_mean[:] = warmup_mean
            cum_std[:] = warmup_std

        # Normalize: For each time step t, subtract the corresponding mean and divide by std.
        return (x - cum_mean) / cum_std
    



class AttnBranchFCBlock(TDSFullyConnectedBlock):
    """Run D: TDSFullyConnectedBlock + parallel zero-init self-attention branch.

    y = LayerNorm( x + fc(x) + attn_branch(x) ), computed in the hand-folded
    view [T, 2N, num_features//2] mirroring the split-and-share FC path.
    With the branch out-projection zero-initialized, the block is function-
    identical to the released FC block at init. Requires share_hand_weights.

    Args:
        num_features (int): full feature dim (both hands).
        share_hand_weights (bool): must be True (SplashNet released arch).
        channels (int): the paired TDSConv2dBlock's total channel count;
            per-hand tokens = channels // 2.
        d_model (int): branch attention dim. (default: 128)
        nhead (int): branch attention heads. (default: 8)
    """

    INFERENCE_CHUNK = 8192  # rows of (T*2N) per chunk when grad is disabled

    def __init__(
        self,
        num_features: int,
        share_hand_weights: bool,
        channels: int,
        d_model: int = 128,
        nhead: int = 8,
        use_fc: bool = True,
    ) -> None:
        super().__init__(num_features, share_hand_weights)
        self.use_fc = use_fc
        assert share_hand_weights, "AttnBranchFCBlock requires share_hand_weights"
        half = num_features // 2
        ch = channels // 2
        assert half % ch == 0
        assert d_model % nhead == 0
        self.branch_channels = ch
        self.branch_width = half // ch

        self.br_in = nn.Linear(self.branch_width, d_model)
        self.br_emb = nn.Parameter(torch.randn(ch, d_model) * 0.02)
        self.br_attn_norm = nn.LayerNorm(d_model)
        self.br_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.br_ffn_norm = nn.LayerNorm(d_model)
        self.br_ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.br_out = nn.Linear(d_model, self.branch_width)
        # Zero-init: branch contributes nothing at init -> exact released fn.
        nn.init.zeros_(self.br_out.weight)
        nn.init.zeros_(self.br_out.bias)

    def _branch_mix(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, branch_channels, branch_width)
        h = self.br_in(tokens) + self.br_emb
        a = self.br_attn_norm(h)
        a, _ = self.br_attn(a, a, a, need_weights=False)
        h = h + a
        h = h + self.br_ffn(self.br_ffn_norm(h))
        return self.br_out(h)

    def _branch(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, 2N, half) hand-folded
        T, B, H = x.shape
        tokens = x.reshape(T * B, self.branch_channels, self.branch_width)
        if torch.is_grad_enabled():
            mixed = self._branch_mix(tokens)
        else:
            mixed = torch.cat(
                [self._branch_mix(t) for t in tokens.split(self.INFERENCE_CHUNK)]
            )
        return mixed.reshape(T, B, H)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        T, N, C = inputs.shape
        x = inputs.view(T, N * 2, C // 2)
        if self.use_fc:
            x = self.fc_block(x) + self._branch(x) + x
        else:  # replace mode: attention-only mixing (fc_block unused)
            x = self._branch(x) + x
        x = self.layer_norm(x)
        return x.view(T, N, C)


class RNNTPredictionNetwork(nn.Module):
    """Transducer prediction network: embedding + LSTM over label history.

    Blank doubles as the BOS token. Input: (N, U) label ids ->
    output: (N, U, pred_dim). Also supports single-step decoding with state.
    """

    def __init__(
        self,
        num_classes: int,
        embed_dim: int = 256,
        pred_dim: int = 512,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, pred_dim, num_layers=num_layers, batch_first=True
        )

    def forward(self, tokens, state=None):
        emb = self.embedding(tokens)  # (N, U, E)
        out, state = self.lstm(emb, state)
        return out, state


class RNNTJoiner(nn.Module):
    """Additive joiner: proj(enc) + proj(pred) -> tanh -> logits."""

    def __init__(
        self,
        enc_dim: int,
        pred_dim: int,
        joiner_dim: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.enc_proj = nn.Linear(enc_dim, joiner_dim)
        self.pred_proj = nn.Linear(pred_dim, joiner_dim)
        self.out = nn.Linear(joiner_dim, num_classes)

    def forward(self, enc, pred):
        """enc: (N, T, D_enc), pred: (N, U, P) -> logits (N, T, U, C)."""
        e = self.enc_proj(enc).unsqueeze(2)   # (N, T, 1, J)
        p = self.pred_proj(pred).unsqueeze(1)  # (N, 1, U, J)
        return self.out(torch.tanh(e + p))

    def step(self, enc_t, pred_u):
        """enc_t: (N, D_enc), pred_u: (N, P) -> logits (N, C)."""
        return self.out(torch.tanh(self.enc_proj(enc_t) + self.pred_proj(pred_u)))


class HATJoiner(nn.Module):
    """Hierarchical Anatomical Transducer joiner.

    Emission log-probs factor over an anatomical tree:
      level 1 (acoustics only): finger-column groups + singleton keys + blank
      level 2 (acoustics + label history): letter WITHIN its finger group
    History can therefore only redistribute within-group mass — the acoustic
    stream owns "which region moved", the vocabulary stream owns "which key
    of that column". Output is a valid log-distribution over all classes.

    Interface mirrors RNNTJoiner: forward(enc (N,T,D), pred (N,U,P)) ->
    (N,T,U,C) log-probs; step(enc_t (N,D), pred_u (N,P)) -> (N,C).
    """

    FINGER_GROUPS = ("qaz", "wsx", "edc", "rfvtgb", "yhnujm", "ik", "ol", "p")

    def __init__(self, enc_dim, pred_dim, num_classes, names):
        super().__init__()
        letter_group = {}
        for gi, g in enumerate(self.FINGER_GROUPS):
            for ch in g:
                letter_group[ch] = gi
        flat = [ch for g in self.FINGER_GROUPS for ch in g]
        pos_of_char = {ch: i for i, ch in enumerate(flat)}
        lens = [len(g) for g in self.FINGER_GROUPS]
        self.group_slices = []
        off = 0
        for L in lens:
            self.group_slices.append((off, off + L))
            off += L
        self.n_letters = off  # 26

        lvl1_of = torch.zeros(num_classes, dtype=torch.long)
        letter_pos = torch.zeros(num_classes, dtype=torch.long)
        letter_mask = torch.zeros(num_classes)
        nxt = len(self.FINGER_GROUPS)
        for lab, name in enumerate(names):
            if len(name) == 1 and name in letter_group:
                lvl1_of[lab] = letter_group[name]
                letter_pos[lab] = pos_of_char[name]
                letter_mask[lab] = 1.0
            else:
                lvl1_of[lab] = nxt
                nxt += 1
        self.n_lvl1 = nxt
        self.register_buffer("lvl1_of", lvl1_of)
        self.register_buffer("letter_pos", letter_pos)
        self.register_buffer("letter_mask", letter_mask)

        self.lvl1 = nn.Linear(enc_dim, self.n_lvl1)
        self.lvl2 = nn.Linear(enc_dim, self.n_letters)
        self.hist2 = nn.Linear(pred_dim, self.n_letters)

    def _group_logsoftmax(self, x):
        outs = []
        for a, b in self.group_slices:
            outs.append(torch.log_softmax(x[..., a:b], dim=-1))
        return torch.cat(outs, dim=-1)

    def _assemble(self, a1, l2):
        """a1 (..., n_lvl1) log-probs, l2 (..., 26) within-group log-probs
        -> (..., num_classes) log-probs."""
        shape = a1.shape[:-1]
        idx1 = self.lvl1_of.expand(*shape, -1)
        out = a1.gather(-1, idx1)
        idx2 = self.letter_pos.expand(*shape, -1)
        out = out + l2.gather(-1, idx2) * self.letter_mask
        return out

    def forward(self, enc, pred):
        a1 = torch.log_softmax(self.lvl1(enc), dim=-1).unsqueeze(2)  # (N,T,1,L1)
        l2 = self.lvl2(enc).unsqueeze(2) + self.hist2(pred).unsqueeze(1)
        l2 = self._group_logsoftmax(l2)  # (N,T,U,26)
        U = pred.shape[1]
        a1 = a1.expand(-1, -1, U, -1)
        return self._assemble(a1, l2)

    def step(self, enc_t, pred_u):
        a1 = torch.log_softmax(self.lvl1(enc_t), dim=-1)
        l2 = self._group_logsoftmax(self.lvl2(enc_t) + self.hist2(pred_u))
        return self._assemble(a1, l2)
