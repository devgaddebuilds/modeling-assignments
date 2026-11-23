# Modeling assignments

Code and analysis for three modeling assignments: linear regression, a
classification tree, and k-means clustering. Each folder holds one assignment.
Running its `analysis.py` reproduces every number, table and chart used on the
slides.

| Folder | Question | Model |
|---|---|---|
| `regression_used_cars/` | Should the dealer buy this car at auction? | Linear regression |
| `decision_tree_clinic/` | Should the clinic overbook this slot? | Classification tree |
| `clustering_gym/` | Which member segments get the monthly credit? | k-means |

## How to run

Python 3.12.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python regression_used_cars/analysis.py
.venv/bin/python decision_tree_clinic/analysis.py
.venv/bin/python clustering_gym/analysis.py
```

Each script prints its results and rewrites the files in its `outputs/` folder.

## What is in each folder

- `data/`: the CSV files supplied with the assignment, unchanged.
- `analysis.py`: the whole analysis, top to bottom, in the order of the assignment's tasks.
- `outputs/results.md`: every number the slides use, written by the script.
- `outputs/*.png`: the charts, drawn by the script.
- `outputs/*.csv`: row-by-row detail behind the totals.

| Folder | Chart | Detail file |
|---|---|---|
| `regression_used_cars/` | `predicted_vs_actual_test.png` | `auction_decisions.csv` |
| `decision_tree_clinic/` | `tree.png` | none |
| `clustering_gym/` | `elbow_train_test.png` | `member_segments.csv` |

## Choices that apply to all three

- The random seed is 42 everywhere, fixed before any result was seen.
- The split is 70% training, 30% test. The clinic split keeps the no-show rate
  the same in both parts.
- Every modeling choice (which feature to drop, tree depth, the number of
  groups, which segments get the credit) is made from the training rows only.
  The test rows, and the 40 auction lots, are used only to score the result.
- Each choice is a named setting at the top of its script, with the rule that
  set it written next to it.
