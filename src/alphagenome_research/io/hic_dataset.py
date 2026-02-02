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

"""Datasets for Hi-C/Micro-C contact map modeling."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class HiCDataset(Dataset):
  """
  Dataset for Hi-C/Micro-C contact maps.

  Loads data from .cool or .mcool files (standard format).
  """

  def __init__(
      self,
      cool_file: str,
      resolution: int = 5000,
      chromosome: str = "chr1",
      region: str | None = None,
      balance: bool = True,
      transform: torch.nn.Module | None = None,
      cache_dir: str | None = None,
  ) -> None:
    super().__init__()

    self.cool_file = cool_file
    self.resolution = resolution
    self.chromosome = chromosome
    self.region = region
    self.balance = balance
    self.transform = transform

    # Load Cooler data
    self._load_cooler_data()

    # Cache to speed up access
    self.cache_dir = cache_dir
    self.cache: dict[str, np.ndarray] = {}

  def _load_cooler_data(self) -> None:
    """Loads data from the cooler file."""
    try:
      import cooler
    except ImportError as exc:
      raise ImportError(
          "cooler is required for HiCDataset; install it to read .cool files."
      ) from exc

    try:
      # Try to load as multi-cooler first
      if self.cool_file.endswith(".mcool"):
        uri = f"{self.cool_file}::/resolutions/{self.resolution}"
      else:
        uri = self.cool_file

      self.cool = cooler.Cooler(uri)

      # Verify that the chromosome exists
      if self.chromosome not in self.cool.chromnames:
        available = self.cool.chromnames
        raise ValueError(f"Chromosome {self.chromosome} not in {available}")

      # Get chromosome information
      chrom_length = self.cool.chromsizes[self.chromosome]
      self.num_bins = chrom_length // self.resolution
    except Exception as exc:
      raise RuntimeError(
          f"Error loading cooler file {self.cool_file}: {exc}"
      ) from exc

  def _fetch_region(self, start_bin: int, end_bin: int) -> np.ndarray:
    """Extracts a region from the contact map."""
    cache_key = f"{start_bin}_{end_bin}"

    if cache_key in self.cache:
      return self.cache[cache_key].copy()

    # Extract contact matrix
    matrix = self.cool.matrix(balance=self.balance).fetch(
        f"{self.chromosome}:{start_bin * self.resolution}-{end_bin * self.resolution}"
    )

    # Convert to float32 and replace NaN
    matrix = np.nan_to_num(matrix.astype(np.float32), nan=0.0)

    # Log-transform (contacts are super-linear)
    matrix = np.log1p(matrix)

    if self.cache_dir:
      self.cache[cache_key] = matrix.copy()

    return matrix

  def __len__(self) -> int:
    # Number of sliding windows
    window_size = 256  # Fixed size for our model
    stride = 128  # 50% overlap
    return max(1, (self.num_bins - window_size) // stride + 1)

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, int | str]]:
    # Calculate window
    window_size = 256
    stride = 128
    start_bin = idx * stride
    end_bin = start_bin + window_size

    # Ensure boundaries are respected
    if end_bin > self.num_bins:
      end_bin = self.num_bins
      start_bin = max(0, end_bin - window_size)

    # Extract matrix
    contact_matrix = self._fetch_region(start_bin, end_bin)

    # Ensure correct size (pad if necessary)
    if contact_matrix.shape[0] < window_size:
      pad_size = window_size - contact_matrix.shape[0]
      contact_matrix = np.pad(
          contact_matrix,
          ((0, pad_size), (0, pad_size)),
          mode="constant",
      )

    # Apply transformations
    if self.transform:
      contact_matrix = self.transform(contact_matrix)

    # Convert to tensor
    contact_tensor = torch.tensor(contact_matrix, dtype=torch.float32).unsqueeze(
        0
    )  # [1, L, L]

    # Region information
    region_info = {
        "chromosome": self.chromosome,
        "start_bp": start_bin * self.resolution,
        "end_bp": end_bin * self.resolution,
        "start_bin": start_bin,
        "end_bin": end_bin,
        "resolution": self.resolution,
    }

    return contact_tensor, region_info


class HiCDataModule:
  """
  Complete module for Hi-C data management.
  Includes loading, preprocessing, and train/val/test splitting.
  """

  def __init__(
      self,
      train_files: list[str],
      val_files: list[str] | None = None,
      test_files: list[str] | None = None,
      batch_size: int = 4,
      num_workers: int = 4,
      train_transform: torch.nn.Module | None = None,
      val_transform: torch.nn.Module | None = None,
  ) -> None:
    self.train_files = train_files
    self.val_files = val_files or []
    self.test_files = test_files or []
    self.batch_size = batch_size
    self.num_workers = num_workers

    # Transformations
    self.train_transform = train_transform
    self.val_transform = val_transform

  def setup(self, stage: str | None = None) -> None:
    """Prepares the datasets."""
    # Training dataset (combine multiple files)
    train_datasets = []
    for cool_file in self.train_files:
      for chrom in ["chr1", "chr2", "chr3"]:  # Chromosomes for training
        dataset = HiCDataset(
            cool_file=cool_file,
            chromosome=chrom,
            transform=self.train_transform,
        )
        train_datasets.append(dataset)

    self.train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    # Validation dataset
    if self.val_files:
      val_datasets = []
      for cool_file in self.val_files:
        for chrom in ["chr4", "chr5"]:  # Different chromosomes for validation
          dataset = HiCDataset(
              cool_file=cool_file,
              chromosome=chrom,
              transform=self.val_transform,
          )
          val_datasets.append(dataset)

      self.val_dataset = torch.utils.data.ConcatDataset(val_datasets)

    # Test dataset
    if self.test_files:
      test_datasets = []
      for cool_file in self.test_files:
        for chrom in ["chr6", "chr7"]:  # Chromosomes for testing
          dataset = HiCDataset(
              cool_file=cool_file,
              chromosome=chrom,
              transform=self.val_transform,
          )
          test_datasets.append(dataset)

      self.test_dataset = torch.utils.data.ConcatDataset(test_datasets)

  def train_dataloader(self) -> DataLoader:
    return DataLoader(
        self.train_dataset,
        batch_size=self.batch_size,
        shuffle=True,
        num_workers=self.num_workers,
        pin_memory=True,
    )

  def val_dataloader(self) -> DataLoader:
    return DataLoader(
        self.val_dataset,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=self.num_workers,
        pin_memory=True,
    )

  def test_dataloader(self) -> DataLoader:
    return DataLoader(
        self.test_dataset,
        batch_size=self.batch_size,
        shuffle=False,
        num_workers=self.num_workers,
        pin_memory=True,
    )
