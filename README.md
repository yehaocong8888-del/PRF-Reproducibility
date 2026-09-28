# PRF Reproducibility Release

This repository contains the publication-facing code, frozen result assets,
model adapters, evaluation artifacts, and revision experiments associated
with the PRF study.

## Repository organization

- `01_prototype_303/` — early 303-sample prototype code and original notebooks.
- `02_full_prf_pipeline/` — full PRF pipeline assets.
- `03_core_validation/` — core validation, ablation, and robustness evidence.
- `04_distillation_dataset/` — Student distillation-dataset construction.
- `05_student_lora/` — original distilled Student training/evaluation and final LoRA adapter.
- `06_external_validation/` — frozen Phase-1 PRF/Teacher and Phase-2 Student external validation.
- `07_boundary_repair/` — boundary-repair experiments and final repaired Student adapter.
- `08_revision_experiments/` — major-revision sensitivity, fusion, and efficiency analyses.
- `09_representation_baselines/` — representation-model comparison evidence.
- `metadata/` — public release manifests, provenance aliases, and SHA256 checksums.
- `docs/` — reproducibility, data/model availability, and paper-to-repository mapping.

## Important scope

Original third-party image datasets are not redistributed in this repository.
Base foundation-model weights are not redistributed. Final LoRA adapters that
were produced by this study are included where permitted.

Historical development checkpoints, failed attempts, model caches, and
superseded experiment outputs were intentionally excluded from the
publication-facing release.

The `.ipynb` notebooks are preserved unchanged. Corresponding
`*_exported.py` files are derived convenience exports generated without
executing the notebooks.

See `metadata/SHA256SUMS.txt` for release-file integrity checks.
