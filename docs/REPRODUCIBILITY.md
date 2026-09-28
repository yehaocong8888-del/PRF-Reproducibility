# Reproducibility Guide

The repository distinguishes three asset types:

1. **Canonical assets** — publication-facing frozen outputs or exact execution sources.
2. **Supporting assets** — protocol, provenance, or confirmatory evidence.
3. **Derived assets** — convenience exports generated from preserved originals.

The public repository was assembled from frozen historical experiment assets.
Source files were first audited and SHA256-hashed. Same-target files with
identical SHA256 values were treated as historical mirrors and stored once
physically while retaining all provenance aliases in the source manifest.

The builder did not execute models or notebooks.

For integrity verification, compare files against:

`metadata/SHA256SUMS.txt`

For mapping between paper components and repository directories, see:

`docs/PAPER_TO_REPOSITORY_MAP.md`
