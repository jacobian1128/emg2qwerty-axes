"""GPU front-end: log-spectrogram (+ SpecAugment in training) computed on the batch on the GPU.

Moves the two most expensive per-sample CPU transforms of the emg2qwerty pipeline
(LogSpectrogram, SpecAugment) out of the data-loader workers. Numerically the
log-spectrogram is the same computation as transforms.LogSpectrogram
(torchaudio Spectrogram(n_fft, hop, normalized=True, center=False, power=2),
log10(x + 1e-6)); it is implemented as unfold + rfft, which agrees with the
torchaudio path to ~5e-7 on real windows. SpecAugment mirrors transforms.SpecAugment
sample by sample (mask count drawn per sample, iid masks per band/electrode).

Input : raw EMG batch (T, N, bands, channels) and input_lengths in samples.
Output: (T', N, bands, channels, freq) and lengths in frames, T' = (T - n_fft)//hop + 1.

No parameters or buffers: the module's state_dict is unchanged, so checkpoints
trained with the front-end load into the plain module (CPU transforms) and vice versa.
"""
from typing import Optional
import numpy as np
import torch
import torchaudio


class GPUFrontend:
    def __init__(
        self,
        n_fft: int = 64,
        hop_length: int = 16,
        n_time_masks: int = 0,
        time_mask_param: int = 0,
        iid_time_masks: bool = True,
        n_freq_masks: int = 0,
        freq_mask_param: int = 0,
        iid_freq_masks: bool = True,
        mask_value: float = 0.0,
    ) -> None:
        self.n_fft, self.hop = n_fft, hop_length
        self._window = torch.hann_window(n_fft)  # periodic, torchaudio default
        self._norm = float(self._window.pow(2).sum())  # normalized=True: |X|^2 / sum(w^2)
        self._win_cache: dict = {}
        self.n_time_masks, self.n_freq_masks = n_time_masks, n_freq_masks
        self.mask_value = mask_value
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param, iid_masks=iid_time_masks)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param, iid_masks=iid_freq_masks)

    def _window_on(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._win_cache:
            self._win_cache[key] = self._window.to(device=device, dtype=dtype)
        return self._win_cache[key]

    def log_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        """(T, N, B, C) -> (T', N, B, C, F)."""
        T, N, B, C = x.shape
        flat = x.permute(1, 2, 3, 0).reshape(-1, T)                 # (N*B*C, T) contiguous
        frames = flat.unfold(-1, self.n_fft, self.hop)               # (N*B*C, T', n_fft), center=False
        X = torch.fft.rfft(frames * self._window_on(x.device, x.dtype), dim=-1)
        logspec = torch.log10(X.real.pow(2).add_(X.imag.pow(2)).div_(self._norm).add_(1e-6))
        Tp, F = logspec.shape[-2], logspec.shape[-1]
        return logspec.reshape(N, B, C, Tp, F).permute(3, 0, 1, 2, 4)  # (T', N, B, C, F)

    def spec_augment(self, spec: torch.Tensor) -> torch.Tensor:
        """Vectorized SpecAugment on (T', N, B, C, F), same distribution as transforms.SpecAugment applied per
        window: per sample n_masks ~ U{0..n_max}; per (band, channel) width ~ U[0, param), start ~ U[0, T-width),
        both truncated to int as torchaudio.functional.mask_along_axis_iid does. ~2 ms per batch of 32."""
        Tp, N, B, C, F = spec.shape
        dev = spec.device
        x = spec
        if self.n_time_masks:
            n_t = torch.randint(0, self.n_time_masks + 1, (N,), device=dev)
            t = torch.arange(Tp, device=dev).view(Tp, 1, 1, 1, 1)
            for k in range(self.n_time_masks):
                w = torch.rand(N, B, C, device=dev) * self.time_mask.mask_param
                s = (torch.rand(N, B, C, device=dev) * (Tp - w)).long()
                w = torch.where((k < n_t).view(N, 1, 1), w, torch.zeros_like(w)).long()
                mask = (t >= s.view(1, N, B, C, 1)) & (t < (s + w).view(1, N, B, C, 1))
                x = x.masked_fill(mask, self.mask_value)
        if self.n_freq_masks:
            n_f = torch.randint(0, self.n_freq_masks + 1, (N,), device=dev)
            f = torch.arange(F, device=dev).view(1, 1, 1, 1, F)
            for k in range(self.n_freq_masks):
                w = torch.rand(N, B, C, device=dev) * self.freq_mask.mask_param
                s = (torch.rand(N, B, C, device=dev) * (F - w)).long()
                w = torch.where((k < n_f).view(N, 1, 1), w, torch.zeros_like(w)).long()
                mask = (f >= s.view(1, N, B, C, 1)) & (f < (s + w).view(1, N, B, C, 1))
                x = x.masked_fill(mask, self.mask_value)
        return x

    def spec_augment_loop(self, spec: torch.Tensor) -> torch.Tensor:
        """Reference implementation: the per-sample torchaudio loop (identical semantics, ~7x slower)."""
        x = spec.permute(1, 2, 3, 4, 0)
        out = []
        for n in range(x.shape[0]):
            xn = x[n]
            for _ in range(np.random.randint(self.n_time_masks + 1)):
                xn = self.time_mask(xn, mask_value=self.mask_value)
            for _ in range(np.random.randint(self.n_freq_masks + 1)):
                xn = self.freq_mask(xn, mask_value=self.mask_value)
            out.append(xn)
        return torch.stack(out, 0).permute(4, 0, 1, 2, 3)

    def frame_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(lengths - self.n_fft, self.hop, rounding_mode="floor") + 1

    def __call__(self, inputs: torch.Tensor, lengths: torch.Tensor, training: bool):
        spec = self.log_spectrogram(inputs)
        if training and (self.n_time_masks or self.n_freq_masks):
            spec = self.spec_augment(spec)
        return spec, self.frame_lengths(lengths)
