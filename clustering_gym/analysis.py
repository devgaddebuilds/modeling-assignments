"""Gym member segments: who should get the monthly offer?

k-means on scaled member features, k chosen from the elbow on the training
members and checked on a 30% test set without refitting, then used to decide
which segments get the $10 extras credit.

Run from anywhere:  python clustering_gym/analysis.py
Writes every number used on the slides to outputs/results.md.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- settings
SEED = 42            # fixed before looking at any result
TEST_SHARE = 0.30
K_RANGE = list(range(1, 9))

# k: None takes the elbow of the training curve (the point furthest from the
# straight line joining k = 1 and k = 8). Set a number to override by hand.
K_OVERRIDE = None

# Finance's assumptions from the brief.
CREDIT_COST = 10          # dollars per member per month
SPEND_THRESHOLD = 20      # members at or above this spend the same without the credit
NET_IF_BELOW = 5          # +$15 margin - $10 credit
NET_IF_AT_OR_ABOVE = -10  # pure giveaway

# Your own names for the segments, e.g. {"A": "...", "B": "..."}. Segments are
# lettered A, B, C... from highest to lowest average extras spend.
SEGMENT_NAMES = {}

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

# member_id is an identifier, not a trait, so it is left out.
FEATURES = ["age", "visits_per_month", "avg_visit_minutes", "classes_per_month",
            "months_member", "monthly_extras_spend"]
LEFT_OUT = ["member_id"]


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


def spread(scaled_rows, centers):
    """Sum of squared distances from each row to its nearest center."""
    d2 = ((scaled_rows[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return float(d2.min(axis=1).sum())


def elbow(ks, values):
    """k whose point sits furthest below the straight line from first to last."""
    x = (np.array(ks) - ks[0]) / (ks[-1] - ks[0])
    y = (np.array(values) - values[-1]) / (values[0] - values[-1])
    return int(ks[int(np.argmax((1 - x) - y))])


def credit_net(spend):
    """Monthly net to the gym from giving this member the credit."""
    return NET_IF_BELOW if spend < SPEND_THRESHOLD else NET_IF_AT_OR_ABOVE


# ---------------------------------------------------------------- 1. split
members = pd.read_csv(DATA / "gym_members.csv")
train, test = train_test_split(members, test_size=TEST_SHARE, random_state=SEED)
scaler = StandardScaler().fit(train[FEATURES])          # scaling fitted on training only
Z_train = scaler.transform(train[FEATURES])
Z_test = scaler.transform(test[FEATURES])

# ---------------------------------------------------------------- 2. fit
models = {k: KMeans(n_clusters=k, n_init=25, random_state=SEED).fit(Z_train) for k in K_RANGE}
train_spread = [spread(Z_train, models[k].cluster_centers_) for k in K_RANGE]

# ---------------------------------------------------------------- 3. test fit
# No refitting: test members go to the nearest center fitted on training members.
test_spread = [spread(Z_test, models[k].cluster_centers_) for k in K_RANGE]

elbow_train = elbow(K_RANGE, train_spread)
elbow_test = elbow(K_RANGE, test_spread)
K = K_OVERRIDE if K_OVERRIDE is not None else elbow_train
v1 = models[K]

spread_table = pd.DataFrame({"k": K_RANGE, "training_spread": train_spread,
                             "test_spread": test_spread})
spread_table["training_drop_from_previous_k"] = -spread_table["training_spread"].diff()
spread_table["test_drop_from_previous_k"] = -spread_table["test_spread"].diff()

# Letter the segments A, B, C... by average extras spend of training members.
train_cluster = v1.predict(Z_train)
mean_spend = pd.Series(train["monthly_extras_spend"].to_numpy()).groupby(train_cluster).mean()
order = list(mean_spend.sort_values(ascending=False).index)
letter_of = {cluster: "ABCDEFGH"[rank] for rank, cluster in enumerate(order)}


def label(letter):
    name = SEGMENT_NAMES.get(letter)
    return f"{letter} ({name})" if name else letter


def segments_for(frame):
    clusters = v1.predict(scaler.transform(frame[FEATURES]))
    return [letter_of[c] for c in clusters]


train = train.assign(segment=segments_for(train), part="train")
test = test.assign(segment=segments_for(test), part="test")
everyone = pd.concat([train, test]).sort_values("member_id")
everyone["net_if_credited"] = everyone["monthly_extras_spend"].map(credit_net)

# ---------------------------------------------------------------- 4. decision
# A segment gets the credit if crediting its training members nets the gym
# money under finance's assumptions (applied member by member).
train_net = train.assign(net=train["monthly_extras_spend"].map(credit_net))
segment_net_train = train_net.groupby("segment")["net"].sum()
credited = sorted(s for s, net in segment_net_train.items() if net > 0)


def assign(member):
    """Takes one member as a dict of the six features.
    Returns (segment letter, True if the segment gets the credit)."""
    row = pd.DataFrame([member])[FEATURES]
    segment = letter_of[int(v1.predict(scaler.transform(row))[0])]
    return segment, segment in credited


NEW_MEMBER = {"age": 30, "visits_per_month": 4, "avg_visit_minutes": 42,
              "classes_per_month": 1, "months_member": 5, "monthly_extras_spend": 10}
new_segment, new_gets_credit = assign(NEW_MEMBER)

# ---------------------------------------------------------------- 5. quantify
everyone["gets_credit_v1"] = everyone["segment"].isin(credited)
baseline_net = int(everyone["net_if_credited"].sum())
v1_net = int(everyone.loc[everyone["gets_credit_v1"], "net_if_credited"].sum())
below = everyone["monthly_extras_spend"] < SPEND_THRESHOLD

centers_original = pd.DataFrame(scaler.inverse_transform(v1.cluster_centers_), columns=FEATURES)
centers_original.insert(0, "segment", [letter_of[c] for c in range(K)])
centers_original = centers_original.sort_values("segment").reset_index(drop=True)

segment_rows = []
for s in sorted(letter_of.values()):
    grp = everyone[everyone["segment"] == s]
    segment_rows.append({
        "segment": label(s),
        "members_all": len(grp),
        "members_train": int((grp["part"] == "train").sum()),
        "members_test": int((grp["part"] == "test").sum()),
        "avg_extras_spend": grp["monthly_extras_spend"].mean(),
        "share_spending_under_20": (grp["monthly_extras_spend"] < SPEND_THRESHOLD).mean(),
        "net_if_credited_training_members": int(segment_net_train[s]),
        "net_if_credited_all_members": int(grp["net_if_credited"].sum()),
        "gets_credit": "yes" if s in credited else "no",
    })
segment_table = pd.DataFrame(segment_rows)

profile = everyone.groupby("segment")[FEATURES].mean().reset_index()
profile["segment"] = profile["segment"].map(label)
overall = everyone[FEATURES].mean()

everyone[["member_id", "part", "segment", "gets_credit_v1", "monthly_extras_spend",
          "net_if_credited"]].to_csv(OUT / "member_segments.csv", index=False)

# ---------------------------------------------------------------- elbow chart
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=200)
panels = [
    (axes[0], train_spread, BLUE, f"Training members ({len(train)})"),
    (axes[1], test_spread, ORANGE, f"Test members ({len(test)}), training centers, no refit"),
]
for ax, values, color, title in panels:
    ax.plot(K_RANGE, values, color=color, linewidth=2, marker="o", markersize=6,
            markeredgecolor="white", markeredgewidth=1, zorder=3)
    ax.axvline(K, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.scatter([K], [values[K - 1]], s=110, facecolor="none", edgecolor=INK,
               linewidth=1.5, zorder=4)
    ax.annotate(f"k = 1 (baseline)\n{values[0]:,.0f}", (1, values[0]),
                xytext=(12, 2), textcoords="offset points", ha="left", va="top",
                fontsize=10)
    ax.annotate(f"k = {K}: {values[K - 1]:,.0f}", (K, values[K - 1]),
                xytext=(10, 12), textcoords="offset points", ha="left", va="bottom",
                fontsize=10)
    ax.set_ylim(0, values[0] * 1.08)
    ax.set_xticks(K_RANGE)
    ax.set_xlabel("Number of groups (k)")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Within-group spread (sum of squared distances)")
fig.tight_layout()
fig.savefig(OUT / "elbow_train_test.png", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- results file
lines = []
add = lines.append
add("# Gym member segments: results\n")
add("Generated by `analysis.py`. Every number below is reproduced by running that script.\n")
add("## 1. Data and split\n")
add(f"- Rows in `gym_members.csv`: {len(members)} (averages over the last 3 months)")
add(f"- Training members (70%): {len(train)}")
add(f"- Test members (30%): {len(test)}")
add(f"- Random seed: {SEED}")
add(f"- Average extras spend, all members: ${members['monthly_extras_spend'].mean():.2f}\n")

add("## 2. Model\n")
add("- Model family: k-means clustering (scikit-learn `KMeans`, 25 restarts)")
add(f"- Features used: {', '.join(FEATURES)}")
add(f"- Left out: {', '.join(LEFT_OUT)} (an identifier, not a trait)")
add("- Scaling: each feature standardised to mean 0 and standard deviation 1, with the "
    "means and standard deviations taken from the training members only")
add(f"- k chosen: {K}"
    + (" (set by K_OVERRIDE)" if K_OVERRIDE is not None else " (elbow of the training curve)"))
if not 3 <= K <= 5:
    add(f"- Note: k = {K} is outside the three to five segments the operations lead asked for.")
add("")

add("## 3. Spread by k: training and test members\n")
add("Spread is the sum of squared distances from each member to the nearest group center, "
    "in scaled units. Test members are assigned to centers fitted on training members; "
    "nothing is refitted. The two columns are sums over different numbers of members "
    f"({len(train)} vs {len(test)}), so compare their shapes, not their levels.\n")
shown = spread_table.copy()
shown["k"] = shown["k"].map("{:.0f}".format)
for col in ["training_spread", "test_spread", "training_drop_from_previous_k",
            "test_drop_from_previous_k"]:
    shown[col] = shown[col].map(lambda v: "" if pd.isna(v) else f"{v:,.1f}")
add(md_table(shown))
add(f"\n- Elbow of the training curve: k = {elbow_train}")
add(f"- Elbow of the test curve (same rule): k = {elbow_test}")
add(f"- Test curve agrees with the chosen k: {'yes' if elbow_test == K else 'NO'}")
k1_test, kK_test = test_spread[0], test_spread[K - 1]
k1_train, kK_train = train_spread[0], train_spread[K - 1]
add(f"- Test spread at k = 1 (baseline): {k1_test:,.1f}")
add(f"- Test spread at k = {K} (v1): {kK_test:,.1f} "
    f"({1 - kK_test / k1_test:.1%} lower than baseline)")
add(f"- Training spread at k = 1: {k1_train:,.1f}; at k = {K}: {kK_train:,.1f} "
    f"({1 - kK_train / k1_train:.1%} lower)")
add("\nChart: `outputs/elbow_train_test.png`\n")

add(f"## 4. The {K} segments\n")
add("Segments are lettered from highest to lowest average extras spend. Names are left "
    "for you to fill in via SEGMENT_NAMES.\n")
add("### Group centers (the average member of each group, original units)\n")
add(md_table(centers_original, {f: "{:.1f}" for f in FEATURES}))
add("\n### Average of all members assigned to each segment (training and test)\n")
add(md_table(profile, {f: "{:.1f}" for f in FEATURES}))
add("\nAll-member averages for comparison: "
    + ", ".join(f"{f} {overall[f]:.1f}" for f in FEATURES) + "\n")
add("### Size, extras spend and credit decision\n")
add(md_table(segment_table, {"avg_extras_spend": "${:.2f}",
                             "share_spending_under_20": "{:.1%}"}))
add(f"\nCredit rule: a segment gets the ${CREDIT_COST} credit if crediting its training "
    "members nets the gym money under finance's assumptions, applied member by member "
    f"(+${NET_IF_BELOW} for a member spending under ${SPEND_THRESHOLD}, "
    f"−${-NET_IF_AT_OR_ABOVE} otherwise).")
add(f"\nSegments that get the credit: {', '.join(credited) or 'none'}\n")

add("## 5. New member\n")
add("- " + ", ".join(f"{k} {v}" for k, v in NEW_MEMBER.items()))
add(f"- Segment: {label(new_segment)}")
add(f"- Gets the credit: {'yes' if new_gets_credit else 'no'}\n")

add("## 6. Monthly net from the credit, all 330 members\n")
add(f"- Members spending under ${SPEND_THRESHOLD}: {int(below.sum())}; "
    f"at or above ${SPEND_THRESHOLD}: {int((~below).sum())}\n")
value_table = pd.DataFrame([
    {"plan": "baseline (credit to everyone)", "members_credited": len(everyone),
     "monthly_net": baseline_net},
    {"plan": f"v1 (credit to segments {', '.join(credited) or 'none'})",
     "members_credited": int(everyone["gets_credit_v1"].sum()), "monthly_net": v1_net},
])
add(md_table(value_table, {"monthly_net": "${:,.0f}"}))
add(f"\n- v1 minus baseline: ${v1_net - baseline_net:,.0f} per month")
add("- Member-by-member detail: `outputs/member_segments.csv`")

(OUT / "results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
