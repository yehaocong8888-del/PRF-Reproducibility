# Paper-to-Repository Map

| Paper / experimental component | Repository location |
|---|---|
| 303-sample prototype | `01_prototype_303/` |
| Full PRF pipeline | `02_full_prf_pipeline/` |
| Core validation and ablation | `03_core_validation/` |
| Distillation-data construction | `04_distillation_dataset/` |
| Original distilled Student | `05_student_lora/` |
| PRF/Teacher external validation | `06_external_validation/phase1_teacher_prf/` |
| Student external generalization | `06_external_validation/phase2_student/` |
| Nine-domain evaluation contract | `06_external_validation/phase2_student/canonical_9set/` |
| Boundary-repair experiments | `07_boundary_repair/` |
| A-threshold sensitivity | `08_revision_experiments/A_threshold_sensitivity/` |
| Adaptive A comparator | `08_revision_experiments/A_adaptive_comparator/` |
| Candidate-ranking / top-k sensitivity | `08_revision_experiments/topk_sensitivity/` |
| C-module sensitivity | `08_revision_experiments/C_gate_sensitivity/` |
| Reliability-weighted fusion | `08_revision_experiments/weighted_fusion/` |
| Computational efficiency | `08_revision_experiments/computational_efficiency/` |
| Representation baselines | `09_representation_baselines/` |

The repository deliberately excludes raw third-party image datasets,
base-model weights, development caches, failed attempts, and superseded
publication outputs.
