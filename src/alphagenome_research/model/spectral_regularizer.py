
# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Spectral regularization utilities for genomic contact maps."""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralContactRegularizer(nn.Module):
    """
    Spectral regularization for genomic contact maps.

    Principle: Penalize high-frequency noise while preserving
    low-frequency structures (TAD domains, loops).

    Based on:
    - Fourier-domain regularization (Chiang et al. 2021)
    - Multi-scale genomic signals (Zhou et al. 2022)
    """

    def __init__(
        self,
        lambda_high: float = 0.1,
        lambda_low: float = 0.05,
        lambda_sym: float = 0.01,
        freq_threshold_high: float = 0.3,
        freq_threshold_low: float = 0.1,
    ) -> None:
        super().__init__()

        self.lambda_high = lambda_high
        self.lambda_low = lambda_low
        self.lambda_sym = lambda_sym

        # These are computed in forward (input size dependent)
        self.freq_threshold_high = freq_threshold_high
        self.freq_threshold_low = freq_threshold_low

        # Cache for frequency masks (optimization)
        self.register_buffer("_freq_masks", None)

    def _create_frequency_masks(self, length: int, device: torch.device) -> torch.Tensor:
        """Creates frequency masks once per input length."""
        if self._freq_masks is not None:
            if self._freq_masks.shape[-1] == length and self._freq_masks.device == device:
                return self._freq_masks

        # 2D frequency space
        frequencies = torch.fft.fftfreq(length, device=device)
        kx, ky = torch.meshgrid(frequencies, frequencies, indexing="ij")

        # Radial frequency (distance from center)
        radial_freq = torch.sqrt(kx**2 + ky**2)
        radial_freq = torch.fft.fftshift(radial_freq)  # Center at 0

        # Masks
        low_freq_mask = (radial_freq <= self.freq_threshold_low).float()
        high_freq_mask = (radial_freq >= self.freq_threshold_high).float()

        # Mid-band mask (optional for analysis)
        mid_freq_mask = (
            (radial_freq > self.freq_threshold_low)
            & (radial_freq < self.freq_threshold_high)
        ).float()

        masks = torch.stack([low_freq_mask, mid_freq_mask, high_freq_mask])
        self._freq_masks = masks

        return masks

    def forward(
        self,
        contact_pred: torch.Tensor,
        contact_target: torch.Tensor | None = None,
        weight_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Args:
            contact_pred: Tensor [batch, L, L] - Model prediction
            contact_target: Tensor [batch, L, L] - Ground truth (optional)
            weight_mask: Tensor [batch, L, L] - Weight mask (e.g., valid regions)

        Returns:
            loss: Scalar tensor - Regularization loss
            metrics: Dict - Diagnostic metrics
        """
        _, length, _ = contact_pred.shape

        # 1. Ensure symmetry (Hi-C is naturally symmetric)
        sym_loss = torch.tensor(0.0, device=contact_pred.device)
        if self.lambda_sym > 0:
            sym_diff = contact_pred - contact_pred.transpose(-1, -2)
            sym_loss = torch.mean(sym_diff**2)

        # 2. 2D Fourier Transform
        pred_fft = torch.fft.fft2(contact_pred.float())
        pred_fft = torch.fft.fftshift(pred_fft)  # Centered frequencies

        # 3. Retrieve frequency masks
        masks = self._create_frequency_masks(length, contact_pred.device)
        low_mask, _, high_mask = masks

        # 4. High-frequency loss (penalize noise or differentiate from target)
        high_freq_loss = torch.tensor(0.0, device=contact_pred.device)
        if contact_target is None:
            high_freq_energy = torch.abs(pred_fft * high_mask.unsqueeze(0)) ** 2
            if weight_mask is not None:
                # Weighted by importance (e.g., high coverage regions)
                high_freq_energy = high_freq_energy * weight_mask
            high_freq_loss = torch.mean(high_freq_energy)
        else:
            target_fft = torch.fft.fft2(contact_target.float())
            target_fft = torch.fft.fftshift(target_fft)

            high_freq_pred = pred_fft * high_mask.unsqueeze(0)
            high_freq_target = target_fft * high_mask.unsqueeze(0)

            high_freq_diff = torch.abs(high_freq_pred - high_freq_target) ** 2
            if weight_mask is not None:
                high_freq_diff = high_freq_diff * weight_mask
            high_freq_loss = torch.mean(high_freq_diff)

        # 5. Low-frequency loss (preserve structure)
        low_freq_loss = torch.tensor(0.0, device=contact_pred.device)
        if contact_target is not None and self.lambda_low > 0:
            target_fft = torch.fft.fft2(contact_target.float())
            target_fft = torch.fft.fftshift(target_fft)

            low_freq_pred = pred_fft * low_mask.unsqueeze(0)
            low_freq_target = target_fft * low_mask.unsqueeze(0)

            low_freq_diff = torch.abs(low_freq_pred - low_freq_target) ** 2

            if weight_mask is not None:
                low_freq_diff = low_freq_diff * weight_mask

            low_freq_loss = torch.mean(low_freq_diff)

        # 6. Total loss
        total_loss = (
            self.lambda_high * high_freq_loss
            + self.lambda_low * low_freq_loss
            + self.lambda_sym * sym_loss
        )

        # 7. Diagnostic metrics
        metrics = {
            "high_freq_loss": high_freq_loss,
            "low_freq_loss": low_freq_loss,
            "symmetry_loss": sym_loss,
            "total_spectral_loss": total_loss,
        }

        return total_loss, metrics

    def analyze_frequency_content(self, contact_map: torch.Tensor) -> dict[str, float]:
        """
        Analyzes frequency content for diagnostic purposes.

        Returns:
            dict with energy distribution across frequency bands.
        """
        length = contact_map.shape[-1]

        # Transform
        contact_fft = torch.fft.fft2(contact_map.float())
        contact_fft = torch.fft.fftshift(contact_fft)
        power_spectrum = torch.abs(contact_fft) ** 2

        # Masks
        masks = self._create_frequency_masks(length, contact_map.device)
        low_mask, mid_mask, high_mask = masks

        # Energy per band
        total_energy = torch.sum(power_spectrum)
        low_energy = torch.sum(power_spectrum * low_mask.unsqueeze(0))
        mid_energy = torch.sum(power_spectrum * mid_mask.unsqueeze(0))
        high_energy = torch.sum(power_spectrum * high_mask.unsqueeze(0))

        return {
            "energy_low_freq": (low_energy / total_energy).item(),
            "energy_mid_freq": (mid_energy / total_energy).item(),
            "energy_high_freq": (high_energy / total_energy).item(),
            "total_energy": total_energy.item(),
        }


class MultiScaleSpectralRegularizer(nn.Module):
    """
    Multi-scale spectral regularization.

    Applies regularization at different resolutions to capture
    structures at various genomic scales:
    - Fine scale: Loops (~1-10kb)
    - Medium scale: TAD domains (~50-500kb)
    - Coarse scale: A/B Compartments (~1Mb+)
    """

    def __init__(self, scales: list[float] | None = None, **kwargs: float) -> None:
        super().__init__()

        if scales is None:
            scales = [1.0, 0.5, 0.25]

        self.scales = scales
        self.regularizers = nn.ModuleList(
            [SpectralContactRegularizer(**kwargs) for _ in scales]
        )

    def forward(
        self,
        contact_pred: torch.Tensor,
        contact_target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        losses = []
        all_metrics: dict[str, torch.Tensor] = {}

        for i, scale in enumerate(self.scales):
            # Resize to scale
            if scale != 1.0:
                length_orig = contact_pred.shape[-1]
                length_scaled = int(length_orig * scale)

                # Simple interpolation (could be improved)
                pred_scaled = torch.nn.functional.interpolate(
                    contact_pred.unsqueeze(1),
                    size=(length_scaled, length_scaled),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

                if contact_target is not None:
                    target_scaled = torch.nn.functional.interpolate(
                        contact_target.unsqueeze(1),
                        size=(length_scaled, length_scaled),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                else:
                    target_scaled = None
            else:
                pred_scaled = contact_pred
                target_scaled = contact_target

            # Apply regularizer to this scale
            loss, metrics = self.regularizers[i](pred_scaled, target_scaled)

            losses.append(loss)

            # Rename metrics by scale
            for key, value in metrics.items():
                all_metrics[f"scale_{scale}_{key}"] = value

        # Weighted average (weights could be learnable)
        total_loss = sum(losses) / len(losses)

        return total_loss, all_metrics
