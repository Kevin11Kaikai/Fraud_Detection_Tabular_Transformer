# Fixed raw feature subset: 52 candidates

Selected by field semantics before looking at any validation/test outcome.
No target encoding, identity join, TransactionID, TransactionDT predictor,
V-feature selection, label-based feature ranking, or hyperparameter sweep.
Training-only constant/all-missing fields are removed and recorded.

| Group | Candidates | Rationale |
|---|---|---|
| Transaction | TransactionAmt; ProductCD | Amount and product context |
| Payment card | card1, card2, card3, card4, card5, card6 | Anonymized card/payment attributes; all treated as categories |
| Address/distance | addr1, addr2; dist1, dist2 | Address categories and numeric distance context |
| Email | P_emaildomain, R_emaildomain | Purchaser/recipient domain categories |
| Counts | C1 through C14 | Anonymized count variables; numeric |
| Delays | D1 through D15 | Anonymized elapsed/delay variables; numeric |
| Match flags | M1 through M9 | Anonymized match flags; categorical |

The anonymization limits interpretation. C/D values are used as supplied;
their availability at the original decision time cannot be independently verified.
No retrospective aggregations are computed. Repeat-entity overlap remains possible.

Both models receive the same retained raw fields and training-selected numeric
missingness indicators (missing fraction in [0.01, 0.99]). Both use training-only
median imputation and log1p(TransactionAmt). LightGBM uses unstandardized numeric
values and categorical codes; FT-Transformer standardizes the numeric values
using training means/stds and embeds those same categorical codes. Category
vocabularies retain at most 2048 training values with count >=2; rare/unseen ->0,
missing ->1. Missing is a reserved category, not a learned numeric quantity.
