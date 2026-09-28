# PRF Phase-2 Student External Generalization — Final Canonical Freeze

Freeze time: 2026-09-23T10:45:53.151337+08:00

This document freezes the final publication-facing statistics for Phase-2 Student external generalization validation. No additional Student model inference is permitted after this freeze unless a separately versioned corrective protocol is explicitly justified.

The nine external evaluation settings contain 13,401 frozen image pairs. The Student produced 13,401 successful predictions with zero format failures.

Per-setting results are the primary reporting unit. Any statistics aggregated across all 13,401 pairs are descriptive only because the external benchmarks differ in source distribution, task construction, class balance, and sampling protocol.

## Canonical per-setting results

| Dataset | N | PRF Acc. | Student Acc. | Macro-F1 | Coverage | Selective Acc. | Student↔PRF | U retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PIPAL | 2000 | 0.9105 | 0.6810 | 0.8066 | 0.6880 | 0.9898 | 0.7375 | 0.9120 |
| LIVE | 810 | 1.0000 | 0.9988 | 0.9994 | 0.9988 | 1.0000 | 0.9988 | NA |
| Waterloo | 3999 | 0.7994 | 0.7384 | 0.8350 | 0.7409 | 0.9966 | 0.7809 | 0.8240 |
| INRIA_Copydays | 3284 | 0.9059 | 0.7275 | 0.8377 | 0.7369 | 0.9872 | 0.7762 | 0.8750 |
| COVERAGE200 | 200 | 0.9600 | 0.9400 | 0.9634 | 0.9500 | 0.9895 | 0.9650 | 0.8000 |
| ISC2021 | 200 | 0.7650 | 0.6200 | 0.7029 | 0.6900 | 0.8986 | 0.7200 | 1.0000 |
| CLWD_controlled | 80 | 0.9250 | 0.8750 | 0.9316 | 0.8750 | 1.0000 | 0.8750 | 0.5000 |
| LVW | 2000 | 0.9830 | 0.9200 | 0.9543 | 0.9245 | 0.9951 | 0.9330 | 0.9474 |
| IMD2020 | 828 | 0.6727 | 0.6618 | 0.6758 | 0.8418 | 0.7862 | 0.8237 | 0.8333 |

## Final policy-transfer findings

- Student–PRF disagreements: 2,490.
- Direct ACCEPT↔REJECT flips: 28 (1.12% of disagreements).
- U-involved disagreements: 2,462 (98.88% of disagreements).
- Student strict errors: 3,118; 2,889 were AMBIGUOUS outputs (92.66%).
- Teacher-routed Student U-rate exceeded non-routed U-rate in 9/9 datasets.
- The routed/non-routed U-rate association reached Fisher p<0.05 in 7/9 settings.
- Student versus PRF paired strict correctness differed at McNemar p<0.05 in 5/9 settings.
- Student selective accuracy was at least 0.98 in 7/9 settings.

## Boundary interpretation

Across most external settings, the dominant Student degradation mode was coverage contraction through AMBIGUOUS prediction rather than direct ACCEPT/REJECT reversal. Teacher-routed cases consistently elicited more Student abstention than non-routed cases, linking the Student's uncertainty behavior to the pre-existing PRF difficulty boundary. This supports policy transfer of risk-sensitive behavior, but not complete reproduction of the teacher's coverage.

IMD2020 is the principal exception: its lower selective accuracy and higher number of direct ACCEPT/REJECT discrepancies indicate genuine domain-specific decision error in addition to abstention.

The two frozen COVERAGE-200 Teacher-intervention cases were both B=AMBIGUOUS -> Teacher/PRF=ACCEPT=Gold, and the Student independently returned ACCEPT in both cases.

## Final methodological status

Phase-2 external inference, statistical evaluation, policy-transfer analysis, confidence intervals, routed-boundary analysis, and intervention-case forensics are complete. No additional Student external model inference is required.