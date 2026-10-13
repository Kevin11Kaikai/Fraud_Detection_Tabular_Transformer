# MVP source map

Follow the implemented data flow in this order:

1. `src/data.py` and `configs/mvp.json`: initial chronological split and train-only
   baseline preprocessing. The earliest neural benchmark was FT-Transformer.
2. `experiments/kaggle_fraudsquad_reproduction_20261011/features.py`: audited
   train-fitted entity frequency, numeric statistics and unique-value counts.
3. `experiments/identity_uid_ablation_20261011/common.py`: common/identity/UID
   feature groups and the B00/B10/B1C information sets.
4. `experiments/identity_uid_transformer_transfer_20261011/prepare.py`: missing
   and unknown handling, chronological categorical encoding, numeric transforms
   and train-only mutual-information ordering.
5. `experiments/identity_uid_transformer_transfer_20261011/models.py`: adaptation
   of the existing three-layer ExcelFormer, with gated feature embeddings and
   semi-permeable attention. Cross-transaction information enters through entity
   statistics, not a separate row-attention branch.
6. `experiments/excel_xgb_distillation_20261012/loss.py` and `training.py`: MI-weighted
   feature-mixing BCE, BF16 gradient accumulation, checkpoint and RNG state.
7. `experiments/excel_xgb_convergence_20261012/train.py`, `common.py`, `report.py`:
   paired continuation, budget/early-stopping controls, fresh reload and rescoring.

The FT helper and FIFO scheduler are included only because these source modules
import them. No additional model searches or unrelated research results are
published. The distillation arm is retained as the completed negative control.

## Historical dependencies

The feature builder reads constants from its cached public Kaggle notebook;
`fetch_sources.py` documents the upstream retrieval. ExcelFormer imports the
pinned external implementation recorded in `UPSTREAM_SOURCES.json`. Upstream
ExcelFormer had no LICENSE at the recorded revision; its implementation is
referenced rather than redistributed in this snapshot.

Archived campaign guards refer to original local absolute paths, feature caches,
preflight/protocol receipts, checkpoints and their hashes. Those source files
are preserved as run, not rewritten into a new training implementation. A fresh
experiment needs its own configuration and artifacts; publishing these scripts
does not establish that such a fresh experiment has been executed.

## Feature counts

The 192 per-record source fields and 38 entity statistics describe the current
pipeline input grouping. They are not all newly invented features: B00 already
included 25 entity aggregates. B1C adds 26 identity fields and 13 UID frequency/
unique-count fields. Missing/unknown indicators and categorical encoding yield
335 numeric input tokens. Category encoding uses earlier training blocks; other
entity maps use the complete training partition, so the full pipeline is not
described as strictly streaming.

The source inventory is in `SOURCE_MANIFEST.json`; private fingerprints remain local.
Row data and weights remain local. Reports distinguish saved-prediction rescore,
fresh checkpoint replay and independent replication.

## Complete results and figure reproduction

The final publication adds all parent/continuation epoch logs and both matched
training arms, plus the original benchmark curves. Results from the earliest
held-out evaluation remain explicitly separate from the later development-
validation comparison. No new model run was performed for publication.

`reports/training_complete_20261012/render_complete.py` can reproduce the full
figure from the published aggregate snapshot, without excluded datasets or
checkpoints. Its unchanged historical source is next to it as
`render_complete_original.py`. The figure-specific README documents this small
input-loading adaptation. The source inventory and adaptation are documented
in `SOURCE_MANIFEST.json`; historical source copies remain unchanged.
Public aggregate records omit local paths and private artifact fingerprints.
