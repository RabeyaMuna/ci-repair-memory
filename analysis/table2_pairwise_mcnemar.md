# Table 2 pairwise statistical analysis

Two-sided exact McNemar tests with Holm correction across all 21 pairwise comparisons. Differences are percentage points (pp). Only a `success` conclusion is counted as a successful repair; `failure`, `cancelled`, `waiting`, and a missing result are counted as failures. See the data note below.

| Configuration A | Configuration B | A only | B only | |Δ Pass@1| (pp) | Raw p | Holm p | Sig. |
|---|---|---:|---:|---:|---:|---:|:---:|
| CI Repair Framework + DeepSeek-Coder | CI Repair Framework + DeepSeek-Chat | 36 | 19 | 3.01 | 0.030 | 0.300 |  |
| CI Repair Framework + DeepSeek-Coder | CI Repair Framework + GPT-4o-mini | 70 | 15 | 9.73 | 1.172e-09 | 2.345e-08 | * |
| CI Repair Framework + DeepSeek-Coder | CI Repair Framework + GPT-5-mini | 53 | 60 | 1.24 | 0.573 | 1.000 |  |
| CI Repair Framework + DeepSeek-Coder | CI Repair Framework + DeepSeek-V4-Flash | 65 | 22 | 7.61 | 4.349e-06 | 5.798e-05 | * |
| CI Repair Framework + DeepSeek-Coder | mini-swe-agent + GPT-5-mini | 46 | 50 | 0.71 | 0.760 | 1.000 |  |
| CI Repair Framework + DeepSeek-Coder | mini-swe-agent + DeepSeek-V4-Flash | 51 | 47 | 0.71 | 0.762 | 1.000 |  |
| CI Repair Framework + DeepSeek-Chat | CI Repair Framework + GPT-4o-mini | 53 | 15 | 6.73 | 4.116e-06 | 5.798e-05 | * |
| CI Repair Framework + DeepSeek-Chat | CI Repair Framework + GPT-5-mini | 42 | 66 | 4.25 | 0.026 | 0.291 |  |
| CI Repair Framework + DeepSeek-Chat | CI Repair Framework + DeepSeek-V4-Flash | 51 | 25 | 4.60 | 0.004 | 0.046 | * |
| CI Repair Framework + DeepSeek-Chat | mini-swe-agent + GPT-5-mini | 37 | 58 | 3.72 | 0.040 | 0.356 |  |
| CI Repair Framework + DeepSeek-Chat | mini-swe-agent + DeepSeek-V4-Flash | 41 | 54 | 2.30 | 0.218 | 1.000 |  |
| CI Repair Framework + GPT-4o-mini | CI Repair Framework + GPT-5-mini | 23 | 85 | 10.97 | 1.520e-09 | 2.888e-08 | * |
| CI Repair Framework + GPT-4o-mini | CI Repair Framework + DeepSeek-V4-Flash | 27 | 39 | 2.12 | 0.175 | 1.000 |  |
| CI Repair Framework + GPT-4o-mini | mini-swe-agent + GPT-5-mini | 15 | 74 | 10.44 | 1.534e-10 | 3.222e-09 | * |
| CI Repair Framework + GPT-4o-mini | mini-swe-agent + DeepSeek-V4-Flash | 25 | 76 | 9.03 | 3.723e-07 | 6.329e-06 | * |
| CI Repair Framework + GPT-5-mini | CI Repair Framework + DeepSeek-V4-Flash | 83 | 33 | 8.85 | 3.865e-06 | 5.798e-05 | * |
| CI Repair Framework + GPT-5-mini | mini-swe-agent + GPT-5-mini | 64 | 61 | 0.53 | 0.858 | 1.000 |  |
| CI Repair Framework + GPT-5-mini | mini-swe-agent + DeepSeek-V4-Flash | 65 | 54 | 1.95 | 0.359 | 1.000 |  |
| CI Repair Framework + DeepSeek-V4-Flash | mini-swe-agent + GPT-5-mini | 26 | 73 | 8.32 | 2.484e-06 | 3.975e-05 | * |
| CI Repair Framework + DeepSeek-V4-Flash | mini-swe-agent + DeepSeek-V4-Flash | 9 | 48 | 6.90 | 1.520e-07 | 2.736e-06 | * |
| mini-swe-agent + GPT-5-mini | mini-swe-agent + DeepSeek-V4-Flash | 57 | 49 | 1.42 | 0.497 | 1.000 |  |

## Data note

The DeepSeek-Chat file contains 564 records and lacks `glances/255`. It is counted as a failure so that the analysis matches the reported 83/565 Pass@1. All other files contain all 565 instances. A complete-case analysis on the 564 common instances produces the same 21 significance decisions.

Missing records detected: `{'CI Repair Framework + DeepSeek-Chat': [('glances', '255')]}`
