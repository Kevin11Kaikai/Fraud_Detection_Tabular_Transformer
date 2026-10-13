# Paired extended training: BCE and XGBoost distillation

Both arms resumed from their complete epoch-5 checkpoints, adding 20 matched epochs; total continuation: 25 epochs, or 41 epochs including the parent. Model weights, AdamW momentum and random states were restored without resetting the optimizer.

Stopping reason: ADDITIONAL_EPOCH_CAP. Validation stability rule satisfied: False; this does not establish mathematical convergence.

| Model | Starting AP | Selected AP after extension | Selected continuation epoch | Fraud found at Top1% | Brier |
|---|---:|---:|---:|---:|---:|
| bce_control | 0.614284 | 0.636633 | 22 | 830 | 0.018585 |
| xgb_kd_lambda1 | 0.613028 | 0.634935 | 22 | 828 | 0.019854 |

The distillation-minus-BCE AP difference was -0.001697; all epochs and negative results are retained. The delivered model is bce_control, AP=0.636633.

Original strong LightGBM AP=0.633859; the strong B1C same-input reference AP=0.687708. Exceeded, respectively: True / False.

Both arms used identical inputs, 335 tokens, seed42, effective batch256/microbatch32, BF16, AdamW at fixed learning rate3e-5, the original MI-weighted mixing, past-prefix teachers and temperature1. Only the teacher-loss coefficient differed (0 or1). Sampling and mixing hashes matched at every epoch.

Stopping rules were frozen before new validation scores: run at least 10 additional matched epochs, then stop both arms only after each has 8 consecutive epochs without an AP gain greater than 0.0001; otherwise run to the20-epoch extension cap or GPU budget cap. Selection for each arm also includes its original epoch5.

Added GPU time: 272.72 minutes; added queue wait: 0.02 minutes; prior-stage GPU time: 73.73 minutes; total: 346.45 minutes. The old ledger was preserved; the added cap was 5.00 hours.

Fresh models reloaded both selected checkpoints and reran inference on all 88,581 validation rows. AP, ROC, Top-K and Brier for every new complete epoch were recomputed using separate NumPy formulas. Original sealed evidence and budgets were preserved.

Neither the final test split nor the official unlabeled test was accessed in this campaign. Validation was reused for selection, so results are single-seed development evidence. Matched stopping conditions and complete execution do not establish theoretical convergence, independent generalization or deployment guarantees. Input maps fitted on the entire training partition are not strictly past-only streaming preprocessing.

Unseen UID, missing identity and all five chronological validation blocks are retained in temporal_uid_metrics.csv. Stratified Top-K uses within-group budgets. Other Excel experiments remain paused as requested and are not resumed automatically.

## Reproduction and checkpoint recovery

Use the existing environment from the original fraud-detection-mvp root:

```powershell
& .\.venv\Scripts\python.exe -X utf8 -B -u -m experiments.excel_xgb_convergence_20261012.campaign
```

Reuse completed experiments. Resume incomplete work with model, AdamW and RNG states from a complete epoch. If one arm has finished a paired epoch, complete only the other arm without repeating the first update. Stop on code, input or parent-model hash mismatch; charge interrupted GPU windows conservatively without resetting budgets.
