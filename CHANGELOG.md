# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0]

### Added

-   Non-zero mean computation to fine-tuning notebook.
-   Gene LFC loss for GenomeTracksHead.
-   Align alternate predictions for splice junctions when making indel
    predictions.

### Changed

-   Fix splice-site and splice-site usage scoring to use the correct gene mask
    extractor.
-   Only consider splice sites that are within a gene body, and overlaps with
    the provided variant.
-   Pass requested outputs to the model prediction call, significantly reducing
    roofline memory consumption by eliding redundant computation.
-   Pad gene masks in gene scoring. This reduces the number of jit
    recompilations for different numbers of genes.

## [0.1.0]

### Added

-   Add code and notebook example for fine-tuning AlphaGenome.
-   Better support for loading pre-trained model with different heads and
    metadata.

### Changed

-   Various fixes to better align predictions with the AlphaGenome API (e.g.
    metadata column names).
-   Fixed variant extraction to correctly insert variants on the negative
    strand.

## [0.0.1]

Initial release.
