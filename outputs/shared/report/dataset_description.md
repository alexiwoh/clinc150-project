## Dataset Description

**Source**: `clinc/clinc_oos` (subset: `plus`) (data/artifacts/dataset_summary.json).

The dataset contains 151 intent classes (150 in-scope + 1 OOS). The OOS class uses label name `oos` (label ID 42).

### Split Sizes

| Split      | Total  | In-Scope | OOS   |
| ---------- | ------ | -------- | ----- |
| Train      | 15,250 | 15,000   | 250   |
| Validation | 3,100  | 3,000    | 100   |
| Test       | 5,500  | 4,500    | 1,000 |

### Class Balance

In-scope classes are balanced in the `plus` subset at 100 training examples per class. OOS support varies across splits: train has 250 OOS examples (~1.6%), while test has 1000 OOS examples (~18.2%), creating a significant distribution shift.

### Distribution Quirks

- Test split has ~18.2% OOS examples vs ~1.6% in train, creating a substantial distribution shift.
- In-scope classes are perfectly balanced in the 'plus' subset (100 training examples per class).
- OOS has 250 training examples, 2.50x the per-class in-scope training count (100).
