# PRF External Validation Final Canonical Audit

## 1. Main external validation

| Dataset | N | B_Accuracy | PRF_Accuracy | Delta_Accuracy | Teacher_Routed | Teacher_ne_B | Correction | Harm | Strong_C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PIPAL | 2000 | 0.910500 | 0.910500 | 0.000000 | 1165 | 0 | 0 | 0 | 0 |
| LIVE | 810 | 1.000000 | 1.000000 | 0.000000 | 74 | 0 | 0 | 0 | 0 |
| Waterloo | 3999 | 0.799450 | 0.799450 | 0.000000 | 752 | 0 | 0 | 0 | 0 |
| INRIA Copydays | 3284 | 0.905907 | 0.905907 | 0.000000 | 1133 | 0 | 0 | 0 | 0 |
| COVERAGE-200 | 200 | 0.950000 | 0.960000 | 0.010000 | 29 | 2 | 2 | 0 | 0 |
| ISC2021 | 200 | 0.765000 | 0.765000 | 0.000000 | 76 | 0 | 0 | 0 | 0 |
| CLWD-controlled | 80 | 0.925000 | 0.925000 | 0.000000 | 24 | 0 | 0 | 0 | 0 |
| LVW | 2000 | 0.983000 | 0.983000 | 0.000000 | 744 | 0 | 0 | 0 | 0 |
| IMD2020 | 828 | 0.672705 | 0.672705 | 0.000000 | 129 | 0 | 0 | 0 | 0 |

The nine accuracy-evaluable external settings contain 13,401 pair observations. COVERAGE-200 is the only setting with a positive PRF-minus-B accuracy point estimate; the other eight settings are decision-equivalent between B and Full PRF at the reported accuracy level.

Across these nine heterogeneous settings, 4,126 pair observations were routed to the 30B teacher. Only 2 teacher decisions differed from B; both occurred in COVERAGE-200, and both were corrections rather than harms. These totals are descriptive mechanism counts rather than a pooled benchmark estimate.

Strong-C was not activated in any of the frozen main external settings.

## 2. COVERAGE-200 arbitration

- B accuracy: 0.9500
- Full PRF accuracy: 0.9600
- Accuracy difference: +0.0100
- Teacher routed: 29/200
- Teacher != B: 2
- Corrections: 2
- Harms: 0
- Net corrections: +2
- Exact two-sided McNemar p: 0.5000

The two corrections demonstrate that the arbitration layer is not strictly redundant with B. However, the small discordant count does not support a claim of statistically significant overall superiority.

## 3. COVERAGE-1000 supplementary expansion

| Subset | N_pairs | Query_groups | B_Accuracy | PRF_Accuracy | Delta_Accuracy | Teacher_Routed | Teacher_ne_B | Correction | Harm | Net | Strong_C |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Nested COVERAGE-200 | 200 | 100 | 0.950000 | 0.960000 | 0.010000 | 29 | 2 | 2 | 0 | 2 | 0 |
| NEW-800 Rank 2-9 negatives | 800 | 100 | 0.995000 | 0.995000 | 0.000000 | 23 | 0 | 0 | 0 | 0 | 0 |
| Full COVERAGE-1000 | 1000 | 100 | 0.986000 | 0.988000 | 0.002000 | 52 | 2 | 2 | 0 | 2 | 0 |

The original COVERAGE-200 result is reproduced exactly inside COVERAGE-1000. The newly added 800 Rank-2–9 negative pairs yield B=PRF=0.995 with no additional teacher override. The expansion therefore serves as a robustness and mechanism-boundary analysis rather than an independent 1000-query benchmark.

## 4. KADID-10k / TID2013 route-only audit

KADID-10k and TID2013 do contain original reference and quality ground truth. However, the current external-validation asset contains 3,198 constructed PRF pairs, and a verified frozen mapping from those pair constructions to the PRF pair-level ACCEPT/REJECT gold labels is not available. Therefore these data are retained for routing/mechanism analysis but are excluded from the canonical accuracy table.

For these 3,198 pair observations, A and B differ on 824 cases; 842 cases are routed to the teacher; the teacher agrees with B on all 842; Strong-C=0.

## 5. Publication interpretation

The frozen external results support an interpretation in which B acts as the primary semantic reviewer, while A/C/D provide conditional auxiliary evidence and escalation. The external evidence does not support describing Full PRF as a universal accuracy booster. Its observed decision-level benefit is sparse and conditional, with COVERAGE providing the direct external evidence of non-destructive teacher correction.