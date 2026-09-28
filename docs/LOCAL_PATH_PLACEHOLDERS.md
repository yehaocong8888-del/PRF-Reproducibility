# Local Path Placeholders

Some frozen experimental assets originally contained absolute paths from
the machines on which the experiments were executed.

The public release replaces only those machine-specific path roots with
semantic placeholders. Numerical results, labels, predictions, pair IDs,
scores, thresholds, and decision outputs are unchanged.

The placeholders have the following meanings:

| Placeholder | Meaning |
|---|---|
| `<PHASE1_EXTERNAL_ROOT>` | Root for Phase-1 external validation assets |
| `<PHASE2_STUDENT_ROOT>` | Root for Phase-2 Student external validation |
| `<PHASE3_BOUNDARY_ROOT>` | Root for Phase-3 boundary-repair experiments |
| `<PHASE4_REVISION_ROOT>` | Root for Phase-4 revision experiments |
| `<STUDENT_DISTILLATION_ROOT>` | Original Student distillation/training root |
| `<STUDENT_GENERALIZATION_ROOT>` | Student validation/generalization root |
| `<FULL_PRF_ROOT>` | Full PRF experimental-result root |
| `<PROTOTYPE303_ROOT>` | 303-sample prototype root |
| `<LORA_HISTORY_ROOT>` | Historical LoRA training root |
| `<PRF_CACHE_ROOT>` | Runtime image/cache root |
| `<LOCAL_USER_HOME>` | Local Windows user-home directory |
| `<LOCAL_LINUX_HOME>` | Local Linux user-home directory |
| `<LOCAL_DRIVE_X>` | Generic fallback for an unmapped local drive |

These placeholders are intentionally not resolved to downloadable raw
datasets. Third-party raw image datasets and base-model weights are not
redistributed by this repository.

For provenance, the internal release audit retains the original source SHA256
values. The public release manifest records hashes of the sanitized public
copies separately.
