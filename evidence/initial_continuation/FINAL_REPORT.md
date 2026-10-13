# Paired XGBoost-to-ExcelFormer B1C distillation report

Result status: **CONTINUATION_TARGET_ACHIEVED_DISTILLATION_GAIN_NOT_ESTABLISHED**. Verification status: **PASS**.

**The single-Transformer pipeline exceeded the predefined LightGBM reference; distillation provided no additional gain over matched BCE continuation in this study.**

## From inputs to outputs

Raw transaction and identity covariates pass through frozen B1C preprocessing to produce 335 input tokens; ExcelFormer outputs fraud probabilities for ranking. During training, three past-prefix XGBoost teachers produce soft targets for later TRAIN blocks, mixed with the original Hid-Mix weights. XGBoost is unnecessary at inference. Teacher feature maps use their own past prefixes; the student retains its original batch TRAIN maps, so the complete pipeline is not claimed to use streaming preprocessing.

## Paired results

| Model | AP ↑ | Recall@Top1% ↑ | Precision@Top1% ↑ | Recall@Top5% ↑ | Brier ↓ |
|---|---:|---:|---:|---:|---:|
| ExcelFormer B1C parent | 0.601916 | 27.02% | 92.78% | 62.62% | 0.019731 |
| bce_control | 0.614284 | 27.42% | 94.13% | 64.56% | 0.019484 |
| xgb_kd_lambda1 | 0.613028 | 27.15% | 93.23% | 64.40% | 0.020573 |
| Regular LightGBM (same rich_all inputs) | 0.608338 | 26.46% | 90.86% | 65.29% | 0.020009 |
| Original strong LightGBM (rich_all) | 0.633859 | 27.05% | 92.89% | 66.50% | 0.019143 |
| LightGBM B1C (same-input Excel B1C control) | 0.687708 | 27.98% | 96.05% | 70.58% | 0.016913 |
| Champion-derived public XGB component (train-only) | 0.671557 | 28.53% | 97.97% | 67.23% | 0.019389 |

Both arms completed 5 matched epochs. The primary attribution comparison selects checkpoints only within this common range. Distillation-minus-BCE AP: -0.0012559647462662138. The original checkpoint remains an epoch0 candidate so a degraded model is not delivered as an improvement.

BCE AP change relative to the original B1C parent: +0.012368; change in fraud found at Top1%: +12. Distillation AP change relative to BCE: -0.001256; change in fraud found at Top1%: -8; Brier change: +0.001089. Distillation had higher ROC-AUC, but AP was the predefined primary metric; AUC was not substituted to declare a win.

The original B1C peak occurred at its final training epoch, so additional training itself could help; the BCE control isolates this gain. The full-TRAIN teacher AP0.671557 is not a guarantee of student performance and is not attributed to prefix teachers. Temperature was fixed at 1 and lambda at 1; no extra teacher, architecture or seed search was performed.

Teacher AP on the three subsequent TRAIN blocks: 0.589873, 0.678231, 0.611154; mean predicted risk: 1.24%, 1.50%, 1.65%; observed fraud prevalence: 3.52%, 4.09%, 3.93%. These are descriptive OOF results from different TRAIN blocks, not entries for the main validation ranking, and were not used to select teacher epochs. Low teacher risk probabilities are an observed signal limitation; the paired results assess the net impact on the student. See teacher_diagnostics.json.

The control measures the net effect of the complete masked distillation objective. The teacher-availability mask also changes effective loss weights for the last75% of TRAIN rows, so the result cannot be isolated as teacher knowledge or calibration. Low teacher probabilities, temporal drift across prefixes and rule-representation transfer remain hypotheses; no additional experiments tested these mechanisms.

## Comparison scope and temporal generalization

The regular rich_all LightGBM AP0.608338 is the declared pipeline reference. B1C adds identity/UID representations, so exceeding it does not demonstrate architectural superiority with identical inputs. The original strong LightGBM AP0.633859 and same-input B1C LightGBM AP0.687708 are retained without weakening or retraining baselines. See temporal_uid_metrics.csv for the five chronological validation blocks and UID/identity strata. Stratified Top-K uses within-group budgets and is not a global investigation-budget gain.

BCE AP over the five chronological blocks: 0.739567, 0.645878, 0.625303, 0.536364, 0.513576; performance still declines in later blocks. UID seen/unseen AP: 0.716807/0.570083; identity observed/missing AP: 0.782685/0.426678. These groups differ in prevalence and composition, so their gaps cannot be attributed entirely to one field. Distillation AP in the last block: 0.523726, above BCE in that block at 0.513576, while its overall validation AP remains lower. This is exploratory heterogeneity, not grounds for changing the primary method or evaluation roles. See temporal_uid_diagnostics.png.

This is development validation reused for selection; neither the remaining test labels nor the official competition test was accessed. One seed, short continuation and validation-based selection do not establish statistical significance, independent future generalization or deployment guarantees. Recall changes are fully disclosed and are not new acceptance criteria.

## Resources and verification

Active GPU lease time in this stage: 73.73 minutes, within a 120-minute cap; queue wait is recorded separately. Lease setup and release overhead are included in whole-lease costs; see budget.json for original windows. Per-trial training, wait, memory and parameter counts appear in ablation_results.csv and each result.json. Teacher CPU preparation costs appear in teacher/features_receipt.json. Historical budgets were preserved, not reset.

Cumulative queue wait: 126.91 minutes; both models have 470738 parameters and approximately3.09GiB peak reserved training memory. All GPU work used the existing FIFO without terminating other threads.

See verification.json for code, original-model and input hashes, matched sampling/Hid-Mix, saved-prediction rescoring and fresh-model checkpoint reloads. This was author verification, not independent third-party validation. Incomplete and failed work is retained in campaign_state.json, failure records and budget.json.

See REPRODUCTION.md for entry points. Fallback LightGBM distillation was not executed.

After comparison within the common training range, the highest-AP single Transformer is saved as selected_model.pt; its source, input cache, metrics and hashes appear in selected_model_card.json. Retaining the original checkpoint or selecting BCE continuation is not described as a distillation result.
