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

"""Contact map predictor with spectral regularization integration."""

from __future__ import annotations

import torch
import torch.nn as nn

from alphagenome_research.model.spectral_regularizer import (
    SpectralContactRegularizer,
)


class SpectralEnhancedContactPredictor(nn.Module):
    """
    Contact predictor with integrated spectral regularization.
    Can be used as a direct replacement in AlphaGenome.
    """

    def __init__(
        self,
        base_predictor: nn.Module,
        use_spectral_reg: bool = True,
        spectral_kwargs: dict[str, float] | None = None,
    ) -> None:
        super().__init__()

        self.base_predictor = base_predictor
        self.use_spectral_reg = use_spectral_reg

        self.spectral_reg: SpectralContactRegularizer | None = None
        if use_spectral_reg:
            spectral_kwargs = spectral_kwargs or {}
            self.spectral_reg = SpectralContactRegularizer(**spectral_kwargs)

        # Post-prediction adjustment layer (optional)
        self.post_process = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Contacts are treated as probabilities
        )

    def forward(
        self, sequence_embedding: torch.Tensor, return_spectral_metrics: bool = False
    ) -> dict[str, torch.Tensor | dict[str, float]]:
        """
        Args:
            sequence_embedding: [batch, L, D] - Sequence embedding
            return_spectral_metrics: If True, returns diagnostic metrics

        Returns:
            contact_map: [batch, L, L] - Predicted contact map
            spectral_loss: Regularization loss (if used)
            metrics: Diagnostic metrics (if requested)
        """
        # 1. Base prediction
        base_pred = self.base_predictor(sequence_embedding)

        # 2. Post-processing
        if self.post_process is not None:
            # Add channel dimension
            base_pred = base_pred.unsqueeze(1)  # [batch, 1, L, L]
            contact_map = self.post_process(base_pred).squeeze(1)
        else:
            contact_map = base_pred

        # 3. Apply spectral regularization (training only)
        spectral_loss = torch.tensor(0.0, device=contact_map.device)
        spectral_metrics: dict[str, float] = {}

        if self.use_spectral_reg and self.training and self.spectral_reg is not None:
            spectral_loss, spectral_metrics = self.spectral_reg(contact_map)

        # 4. Return results
        outputs: dict[str, torch.Tensor | dict[str, float]] = {
            "contact_map": contact_map
        }

        if return_spectral_metrics:
            outputs["spectral_metrics"] = spectral_metrics

        if self.training:
            outputs["spectral_loss"] = spectral_loss

        return outputs

    def predict_with_uncertainty(
        self, sequence_embedding: torch.Tensor, n_samples: int = 10
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Prediction with uncertainty estimation using dropout.

        Returns:
            mean_pred: Average prediction
            uncertainty: Per-pixel uncertainty
        """
        self.train()  # Activate dropout

        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(sequence_embedding)["contact_map"]
                predictions.append(pred)

        self.eval()

        predictions_stack = torch.stack(predictions)  # [n_samples, batch, L, L]
        mean_pred = predictions_stack.mean(dim=0)
        uncertainty = predictions_stack.std(dim=0)

        return mean_pred, uncertainty
