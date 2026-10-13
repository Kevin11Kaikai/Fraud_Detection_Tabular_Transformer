# Complete training visualization

`training_complete.png` and `.svg` are the visually checked English exports.
`figure_data.json` contains 16 parent epochs and 25 continuation epochs for each
arm, the selected-checkpoint metrics and fixed tree references.
Raw per-transaction records and predictions are not included.

The four panels show complete validation AP, the shared hard BCE training loss,
fraud captured within 886 reviews, and the matched feature-ablation stages.
Both the original LightGBM and the stronger enriched-field tree reference are
marked. Distillation is retained even though it did not improve on BCE.

## Reproduce the figure without a GPU or dataset

With Python, NumPy and Matplotlib installed, run from the branch root:

```powershell
python reports/training_complete_20261012/render_complete.py --output rendered_training_results
```

The English renderer defaults to Matplotlib's bundled DejaVu Sans and requires
no Chinese font. An optional local font file can be supplied with
`--font /path/to/font.ttf`. All outputs go into the separate folder; the
published artifacts are preserved. The SVG keeps selectable English text.

`render_complete_original.py` preserves the historical data-loading strategy;
its display text is localized into English. The original frozen code and
artifacts remain local. The public `render_complete.py` reads the exported
aggregate snapshot and published evidence, with English labels and adjusted
spacing for readability. It verifies complete epoch sequences and
selected-metric consistency before rendering. Reproduction renders saved metrics;
it does not train, run inference, or independently replicate the model.

The visual-review outcome is preserved. Local paths and model/data fingerprints
remain in private local audit records. The
initial benchmark's separate held-out results are archived under
`evidence/original_mvp/`; they are not the development-validation headline.
