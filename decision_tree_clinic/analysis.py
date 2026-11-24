"""Clinic no-shows: should the clinic overbook this slot?

One classification tree predicting no-shows, evaluated on a 30% stratified test
set, turned into an overbook/hold rule and valued with the clinic's cost table.

Run from anywhere:  python decision_tree_clinic/analysis.py
Writes every number used on the slides to outputs/results.md.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

# ---------------------------------------------------------------- settings
SEED = 42            # fixed before looking at any result
TEST_SHARE = 0.30

# Cost table from the brief (value per appointment, dollars).
COST_OVERBOOK_AND_PATIENT_CAME = -60    # false positive: both patients came
COST_HOLD_AND_PATIENT_MISSED = -120     # false negative: slot sat empty

# An empty slot costs twice what a double booking costs, so a missed no-show
# counts twice as much as a false alarm when the tree is grown.
CLASS_WEIGHT = {0: 1, 1: 2}             # 0 = checked-in, 1 = no-show
MIN_LEAF = 10                           # no leaf smaller than 10 training appointments
DEPTH_CANDIDATES = [1, 2, 3, 4, 5, 6]

# Depth: None picks the depth with the lowest cross-validated cost inside the
# training set (the test set is never used). Set a number to override by hand.
DEPTH_OVERRIDE = None

# Appointments per year at the clinic. This is an assumption, stated on the
# slide. Set to None to skip the per-year line.
ANNUAL_APPOINTMENTS = 10_000

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

FEATURES = ["lead_time_days", "prior_no_shows", "age", "distance_miles",
            "sms_reminder", "new_patient", "appointment_hour"]
CLASS_NAMES = ["show", "no-show"]


def md_table(df, formats=None):
    formats = formats or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            f = formats.get(c)
            cells.append(f.format(row[c]) if f else str(row[c]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_tree(depth):
    return DecisionTreeClassifier(max_depth=depth, min_samples_leaf=MIN_LEAF,
                                  class_weight=CLASS_WEIGHT, random_state=SEED)


def total_value(tn, fp, fn, tp):
    return fp * COST_OVERBOOK_AND_PATIENT_CAME + fn * COST_HOLD_AND_PATIENT_MISSED


def metrics(actual, predicted):
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    n = tn + fp + fn + tp
    value = total_value(tn, fp, fn, tp)
    return {
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "accuracy": (tn + tp) / n,
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "total_value": float(value),
        "value_per_appointment": float(value) / n,
    }


# ---------------------------------------------------------------- 1. split
appts = pd.read_csv(DATA / "clinic_appointments.csv")
appts["y"] = (appts["no_show"].str.strip() == "no-show").astype(int)
train, test = train_test_split(appts, test_size=TEST_SHARE, random_state=SEED,
                               stratify=appts["y"])
X_train, y_train = train[FEATURES], train["y"]
X_test, y_test = test[FEATURES], test["y"]

# ---------------------------------------------------------------- 2. fit
# Depth is chosen inside the training set: 5-fold cross-validation repeated 10
# times, scored with the clinic's own cost table.
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
cv_rows = []
for depth in DEPTH_CANDIDATES:
    costs = []
    for fit_idx, val_idx in cv.split(X_train, y_train):
        m = make_tree(depth).fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        pred = m.predict(X_train.iloc[val_idx])
        tn, fp, fn, tp = confusion_matrix(y_train.iloc[val_idx], pred, labels=[0, 1]).ravel()
        costs.append(total_value(tn, fp, fn, tp) / len(val_idx))
    cv_rows.append({"max_depth": depth, "cv_value_per_appointment": float(np.mean(costs))})
cv_table = pd.DataFrame(cv_rows)
# Highest value (least negative) wins; idxmax returns the shallowest on a tie.
auto_depth = int(cv_table.loc[cv_table["cv_value_per_appointment"].round(4).idxmax(), "max_depth"])
depth = DEPTH_OVERRIDE if DEPTH_OVERRIDE is not None else auto_depth

v1 = make_tree(depth).fit(X_train, y_train)
tree = v1.tree_
train_leaf = v1.apply(X_train)

# Unweighted training counts in every node, for readable descriptions.
node_members = v1.decision_path(X_train).toarray().astype(bool)
node_n = node_members.sum(axis=0)
node_noshow = (node_members & (y_train.to_numpy()[:, None] == 1)).sum(axis=0)
node_pred = np.argmax(tree.value[:, 0, :], axis=1)   # weighted majority = v1's call


def node_rule(i):
    return f"{FEATURES[tree.feature[i]]} <= {tree.threshold[i]:g}"


used_features = []
order = [0]
while order:                                   # breadth-first, root first
    i = order.pop(0)
    if tree.children_left[i] != -1:
        if FEATURES[tree.feature[i]] not in used_features:
            used_features.append(FEATURES[tree.feature[i]])
        order += [tree.children_left[i], tree.children_right[i]]
unused_features = [f for f in FEATURES if f not in used_features]


def leaf_path(leaf):
    """Conditions from the root down to `leaf`, as readable text."""
    steps, i = [], 0
    parents = {}
    for p in range(tree.node_count):
        for child, sign in ((tree.children_left[p], "<="), (tree.children_right[p], ">")):
            if child != -1:
                parents[child] = (p, sign)
    i = leaf
    while i in parents:
        p, sign = parents[i]
        steps.append(f"{FEATURES[tree.feature[p]]} {sign} {tree.threshold[p]:g}")
        i = p
    return list(reversed(steps))


leaves = [i for i in range(tree.node_count) if tree.children_left[i] == -1]
leaf_table = pd.DataFrame([{
    "leaf": i,
    "path": " and ".join(leaf_path(i)) or "(root)",
    "training_appointments": int(node_n[i]),
    "training_no_show_share": node_noshow[i] / node_n[i],
    "v1_predicts": CLASS_NAMES[node_pred[i]],
    "decision": "overbook" if node_pred[i] == 1 else "hold",
} for i in leaves])

# ---------------------------------------------------------------- 3. test fit
baseline_pred = np.zeros(len(y_test), dtype=int)       # everyone shows
v1_test_pred = v1.predict(X_test)
base_m = metrics(y_test, baseline_pred)
v1_m = metrics(y_test, v1_test_pred)
train_accuracy = float((v1.predict(X_train) == y_train).mean())


# ---------------------------------------------------------------- 4. decision
def decide(appointment):
    """Takes one appointment as a dict of the seven features.
    Returns ("overbook" or "hold", leaf id)."""
    row = pd.DataFrame([appointment])[FEATURES]
    leaf = int(v1.apply(row)[0])
    return ("overbook" if v1.predict(row)[0] == 1 else "hold"), leaf


UPCOMING = {
    "D": {"lead_time_days": 21, "prior_no_shows": 1, "age": 27, "distance_miles": 9.5,
          "sms_reminder": 0, "new_patient": 1, "appointment_hour": 15},
    "E": {"lead_time_days": 3, "prior_no_shows": 0, "age": 61, "distance_miles": 2.0,
          "sms_reminder": 1, "new_patient": 0, "appointment_hour": 9},
}
upcoming_rows = []
for name, appt in UPCOMING.items():
    decision, leaf = decide(appt)
    info = leaf_table[leaf_table["leaf"] == leaf].iloc[0]
    upcoming_rows.append({
        "appointment": name, "leaf": leaf, "path": info["path"],
        "training_appointments_in_leaf": info["training_appointments"],
        "training_no_show_share_in_leaf": info["training_no_show_share"],
        "decision": decision,
    })
upcoming_table = pd.DataFrame(upcoming_rows)

# ---------------------------------------------------------------- tree picture
INK, MUTED = "#0b0b0b", "#52514e"
HOLD_FILL, OVERBOOK_FILL, SPLIT_FILL = "#cde2fb", "#fbd9c9", "#ffffff"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})

n_leaves = len(leaves)


def node_question(i):
    """The split at node i as a plain yes/no question. Yes goes left."""
    feature, threshold = FEATURES[tree.feature[i]], tree.threshold[i]
    if feature == "sms_reminder":
        return "No SMS reminder?"
    if feature == "new_patient":
        return "Returning patient?"
    if feature == "prior_no_shows":
        most = int(np.floor(threshold))
        return "No prior no-shows?" if most == 0 else f"Prior no-shows: {most} or fewer?"
    label = {"lead_time_days": "Lead time (days)", "age": "Age",
             "distance_miles": "Distance (miles)",
             "appointment_hour": "Appointment hour"}[feature]
    return f"{label} at most {threshold:g}?"


def draw_tree(filename, figsize, dpi, font, title):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    annotations = plot_tree(v1, feature_names=FEATURES, ax=ax, impurity=False, fontsize=font)
    # plot_tree also returns the "True"/"False" branch labels; drop them and keep
    # the node boxes, which come back in node-id order.
    branch_labels = [a for a in annotations if a.get_text().strip() in ("True", "False")]
    for a in branch_labels:
        a.set_visible(False)
    node_boxes = [a for a in annotations if a not in branch_labels]
    assert len(node_boxes) == tree.node_count
    for i, ann in enumerate(node_boxes):
        assert f"samples = {node_n[i]}" in ann.get_text(), "node order mismatch"
        share = node_noshow[i] / node_n[i]
        body = f"{node_n[i]} appointments\n{share:.0%} no-show"
        if tree.children_left[i] == -1:
            decision = "OVERBOOK" if node_pred[i] == 1 else "HOLD"
            ann.set_text(f"Leaf {i}\n{body}\n{decision}")
            fill = OVERBOOK_FILL if node_pred[i] == 1 else HOLD_FILL
        else:
            ann.set_text(f"{node_question(i)}\n{body}")
            fill = SPLIT_FILL
        ann.set_color(INK)
        ann.set_bbox(dict(boxstyle="round,pad=0.4", facecolor=fill, edgecolor=MUTED,
                          linewidth=0.8))
        if ann.arrow_patch is not None:
            ann.arrow_patch.set_color(MUTED)
    ax.set_title(title, loc="left", fontsize=font + 1, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / filename, facecolor="white")
    plt.close(fig)


# Full-size picture, and a compact one with larger text for a small spot on a slide.
draw_tree("tree.png", (max(8.0, 2.3 * n_leaves), 1.9 * (v1.get_depth() + 1)), 200, 9,
          f"v1 tree (depth {v1.get_depth()}), counts from the {len(train)} training "
          "appointments.  Left branch = yes, right branch = no.")
draw_tree("tree_compact.png", (1.6 * n_leaves, 1.05 * (v1.get_depth() + 1)), 300, 8,
          f"Counts from the {len(train)} training appointments. Left = yes, right = no.")

# ---------------------------------------------------------------- value chart
BLUE, ORANGE, GRID = "#2a78d6", "#eb6834", "#e4e3df"
with plt.rc_context({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False}):
    costs = [-base_m["value_per_appointment"], -v1_m["value_per_appointment"]]
    fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=300)
    bars = ax.bar(["Baseline\nhold every slot", "v1\noverbook predicted no-shows"], costs,
                  width=0.5, color=[ORANGE, BLUE])
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2, cost + max(costs) * 0.03, f"${cost:,.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold", color=INK)
    ax.set_ylim(0, max(costs) * 1.2)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Cost per appointment")
    fig.suptitle(f"Cost per appointment ({len(test)} test appointments)",
                 x=0.03, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "cost_per_appointment.png", facecolor="white")
    plt.close(fig)

# ---------------------------------------------------------------- results file
lines = []
add = lines.append
add("# Clinic no-shows: results\n")
add("Generated by `analysis.py`. Every number below is reproduced by running that script.\n")
add("## 1. Data and split\n")
add(f"- Rows in `clinic_appointments.csv`: {len(appts)}, no-shows: {int(appts['y'].sum())} "
    f"({appts['y'].mean():.1%})")
add(f"- Training appointments (70%): {len(train)}, no-shows: {int(y_train.sum())} "
    f"({y_train.mean():.1%})")
add(f"- Test appointments (30%): {len(test)}, no-shows: {int(y_test.sum())} "
    f"({y_test.mean():.1%})")
add(f"- Split is stratified on the outcome. Random seed: {SEED}\n")

add("## 2. Model (v1)\n")
add("- Outcome: no-show (positive class) vs checked-in")
add("- Model family: classification tree (scikit-learn `DecisionTreeClassifier`, Gini)")
add(f"- Features offered to the tree: {', '.join(FEATURES)}")
add(f"- Depth: {v1.get_depth()} (max_depth = {depth}), leaves: {n_leaves}, "
    f"minimum {MIN_LEAF} training appointments per leaf")
add(f"- Class weights: no-show errors count {CLASS_WEIGHT[1]}x, matching the cost table "
    f"(${-COST_HOLD_AND_PATIENT_MISSED} empty slot vs ${-COST_OVERBOOK_AND_PATIENT_CAME} "
    "double booking). In effect a leaf is called no-show when more than one third of its "
    "training appointments were no-shows.")
add(f"- Features the tree splits on, root first: {', '.join(used_features)}")
add(f"- Features offered but never used: {', '.join(unused_features) or 'none'}")
add(f"- Root split: {node_rule(0)}\n")
add("### How the depth was chosen (training set only)\n")
add("5-fold cross-validation repeated 10 times inside the training set, scored with the "
    "cost table. Higher (closer to zero) is better.\n")
add(md_table(cv_table, {"max_depth": "{:.0f}", "cv_value_per_appointment": "${:,.2f}"}))
add(f"\nSelected depth: {depth}"
    + (" (set by DEPTH_OVERRIDE)" if DEPTH_OVERRIDE is not None else " (best cross-validated value)")
    + "\n")
add("### Tree rules\n")
add("```\n" + export_text(v1, feature_names=FEATURES, class_names=CLASS_NAMES).rstrip() + "\n```\n")
add("### Leaves (counts are unweighted training appointments)\n")
add(md_table(leaf_table, {"training_no_show_share": "{:.1%}"}))
add("\nTree picture: `outputs/tree.png`\n")

add("## 3. Fit on the test appointments: baseline vs v1\n")
add("Positive class is no-show. Baseline predicts every patient shows.\n")
fit_table = pd.DataFrame([
    {"model": "baseline (everyone shows)", **base_m},
    {"model": "v1 (tree)", **v1_m},
])[["model", "TN", "FP", "FN", "TP", "accuracy", "precision", "recall"]]


def pct(v):
    return "undefined (no no-shows predicted)" if pd.isna(v) else f"{v:.1%}"


fit_table["accuracy"] = fit_table["accuracy"].map(pct)
fit_table["precision"] = fit_table["precision"].map(pct)
fit_table["recall"] = fit_table["recall"].map(pct)
add(md_table(fit_table))
add(f"\n- v1 training accuracy: {train_accuracy:.1%}")
add(f"- v1 test accuracy: {v1_m['accuracy']:.1%}")
add(f"- Gap (training minus test): {train_accuracy - v1_m['accuracy']:+.1%}\n")

add("## 4. Decision for the two upcoming appointments\n")
add("Rule: overbook the slot if v1 predicts a no-show, hold it if v1 predicts a show.\n")
add(md_table(upcoming_table, {"training_no_show_share_in_leaf": "{:.1%}"}))
add("")

add("## 5. Value on the test appointments\n")
add(f"Cost table: overbooked and patient came ${COST_OVERBOOK_AND_PATIENT_CAME}, "
    f"held and patient missed ${COST_HOLD_AND_PATIENT_MISSED}, otherwise $0.\n")
value_table = pd.DataFrame([
    {"policy": "baseline (hold every slot)", "total_value": base_m["total_value"],
     "value_per_appointment": base_m["value_per_appointment"]},
    {"policy": "v1 (overbook predicted no-shows)", "total_value": v1_m["total_value"],
     "value_per_appointment": v1_m["value_per_appointment"]},
])
add(md_table(value_table, {"total_value": "${:,.0f}", "value_per_appointment": "${:,.2f}"}))
gain_total = v1_m["total_value"] - base_m["total_value"]
gain_each = v1_m["value_per_appointment"] - base_m["value_per_appointment"]
add(f"\n- v1 minus baseline, {len(test)} test appointments: ${gain_total:,.0f}")
add(f"- v1 minus baseline, per appointment: ${gain_each:,.2f}")
if ANNUAL_APPOINTMENTS:
    add(f"\nPer year, at an assumed {ANNUAL_APPOINTMENTS:,} appointments:\n")
    add(f"- Baseline: ${base_m['value_per_appointment'] * ANNUAL_APPOINTMENTS:,.0f}")
    add(f"- v1: ${v1_m['value_per_appointment'] * ANNUAL_APPOINTMENTS:,.0f}")
    add(f"- v1 minus baseline: ${gain_each * ANNUAL_APPOINTMENTS:,.0f}")
else:
    add("- Per year: not computed. Set ANNUAL_APPOINTMENTS at the top of `analysis.py` to "
        "the yearly volume you assume; the figure is the per-appointment gain times that volume.")

(OUT / "results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
