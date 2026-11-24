"""Used car prices: should the dealer buy this car?

Linear regression of sale price on car features, evaluated on a 30% test set,
then turned into a buy/pass rule and scored on 40 auction lots.

Run from anywhere:  python regression_used_cars/analysis.py
Writes every number used on the slides to outputs/results.md.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------- settings
SEED = 42            # fixed before looking at any result
TEST_SHARE = 0.30
PREP_COST = 1500     # dollars to get a car ready for the lot
P_CUTOFF = 0.05      # a feature "earns its place" if its p-value is below this
ADJ_R2_TOLERANCE = 0.005  # keep the smaller model if adjusted R2 falls by less than this

# Which model to hand over as v1: "auto" applies the rule above on training
# data only. Set to "full" or "reduced" to override the choice by hand.
HANDOVER = "auto"

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

OUTCOME = "Sale_Price"
FEATURES = ["Mileage_K", "Age_Years", "Horsepower", "Accident_Yes", "Num_Owners"]


def load(path):
    df = pd.read_csv(path)
    df["Accident_Yes"] = (df["Accident"].str.strip().str.lower() == "yes").astype(int)
    return df


def fit_ols(train, features):
    X = sm.add_constant(train[features].astype(float))
    return sm.OLS(train[OUTCOME].astype(float), X).fit()


def predict(model, df, features):
    X = sm.add_constant(df[features].astype(float), has_constant="add")
    return model.predict(X)


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2)))


def coef_table(model):
    t = pd.DataFrame({
        "coefficient": model.params,
        "std_error": model.bse,
        "p_value": model.pvalues,
    })
    t.index.name = "term"
    return t


def md_table(df, formats=None):
    """Small markdown table writer (avoids needing the tabulate package)."""
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


# ---------------------------------------------------------------- 1. split
cars = load(DATA / "used_car_sales.csv")
train, test = train_test_split(cars, test_size=TEST_SHARE, random_state=SEED)

# ---------------------------------------------------------------- 2. fit
full = fit_ols(train, FEATURES)
weak = [f for f in FEATURES if full.pvalues[f] >= P_CUTOFF]
kept = [f for f in FEATURES if f not in weak]
reduced = fit_ols(train, kept) if weak else None

if HANDOVER == "full" or reduced is None:
    choice = "full"
elif HANDOVER == "reduced":
    choice = "reduced"
else:
    drop_in_adj_r2 = full.rsquared_adj - reduced.rsquared_adj
    choice = "reduced" if drop_in_adj_r2 < ADJ_R2_TOLERANCE else "full"

v1 = reduced if choice == "reduced" else full
v1_features = kept if choice == "reduced" else FEATURES

# ---------------------------------------------------------------- 3. test fit
baseline_price = float(train[OUTCOME].mean())   # one guess for every car
test_actual = test[OUTCOME].to_numpy(dtype=float)
test_pred_base = np.full(len(test), baseline_price)
test_pred_v1 = predict(v1, test, v1_features).to_numpy()

fit_rows = pd.DataFrame([
    {"model": "baseline (training average)",
     "test_R2": r2_score(test_actual, test_pred_base),
     "test_RMSE": rmse(test_actual, test_pred_base)},
    {"model": "v1 (linear regression)",
     "test_R2": r2_score(test_actual, test_pred_v1),
     "test_RMSE": rmse(test_actual, test_pred_v1)},
])


# ---------------------------------------------------------------- 4. decision
def decide(car, auction_price):
    """v1 rule. `car` is a dict with Mileage_K, Age_Years, Horsepower,
    Accident ("yes"/"no") and Num_Owners. Returns "buy" or "pass"."""
    row = dict(car)
    row["Accident_Yes"] = 1 if str(row.get("Accident", "no")).strip().lower() == "yes" else 0
    predicted = float(predict(v1, pd.DataFrame([row]), v1_features).iloc[0])
    return "buy" if auction_price + PREP_COST < predicted else "pass"


def decide_baseline(car, auction_price):
    """Baseline rule: every car is predicted to sell for the training average."""
    return "buy" if auction_price + PREP_COST < baseline_price else "pass"


lots = load(DATA / "auction_lots.csv")
lots["Predicted_Sale_Price"] = predict(v1, lots, v1_features).round(0)
lots["Profit_If_Bought"] = lots["Later_Sale_Price"] - lots["Auction_Price"] - PREP_COST
feature_cols = ["Mileage_K", "Age_Years", "Horsepower", "Accident", "Num_Owners"]
lots["Baseline_Decision"] = [
    decide_baseline(r[feature_cols].to_dict(), r["Auction_Price"]) for _, r in lots.iterrows()]
lots["V1_Decision"] = [
    decide(r[feature_cols].to_dict(), r["Auction_Price"]) for _, r in lots.iterrows()]


def score(decisions):
    bought = lots[decisions == "buy"]
    return {
        "cars_bought": int(len(bought)),
        "money_losing_cars_bought": int((bought["Profit_If_Bought"] < 0).sum()),
        "total_profit": float(bought["Profit_If_Bought"].sum()),
    }


value_rows = pd.DataFrame([
    {"rule": "baseline", **score(lots["Baseline_Decision"])},
    {"rule": "v1", **score(lots["V1_Decision"])},
])
buy_everything = float(lots["Profit_If_Bought"].sum())
lots.drop(columns=["Accident_Yes"]).to_csv(OUT / "auction_decisions.csv", index=False)

# ---------------------------------------------------------------- chart
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})

lo = min(test_actual.min(), test_pred_v1.min()) - 1500
hi = max(test_actual.max(), test_pred_v1.max()) + 1500
fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k")


def draw_chart(filename, figsize, dpi, font, marker, tick_step, title_across_figure=False):
    """Predicted vs actual sale price for the test cars."""
    with plt.rc_context({"font.size": font}):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.axhline(baseline_price, color=ORANGE, linewidth=2, zorder=2)
        ax.scatter(test_actual, test_pred_v1, s=marker, color=BLUE, edgecolor="white",
                   linewidth=0.8, zorder=3)
        ax.text(lo + 600, baseline_price + 400,
                f"Baseline: ${baseline_price:,.0f}\nfor every car",
                ha="left", va="bottom", color=INK, fontsize=font - 1)
        ax.text(hi - 600, lo + 1200, "Dashed line = perfect prediction",
                ha="right", va="bottom", color=MUTED, fontsize=font - 1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(matplotlib.ticker.MultipleLocator(tick_step))
            axis.set_major_formatter(fmt)
        ax.set_xlabel("Actual sale price")
        ax.set_ylabel("Predicted sale price (v1)")
        title = f"Predicted vs actual sale price, {len(test)} test cars"
        if title_across_figure:      # the title is wider than the plot, so span the figure
            fig.suptitle(title, x=0.03, ha="left", fontsize=font + 1, fontweight="bold")
        else:
            ax.set_title(title, loc="left", fontsize=font + 1, fontweight="bold")
        fig.tight_layout()
        fig.savefig(OUT / filename, facecolor="white")
        plt.close(fig)


# Full-size chart, and a compact one with larger text for a small spot on a slide.
draw_chart("predicted_vs_actual_test.png", (7.2, 5.4), 200, 11, 34, 5000)
draw_chart("predicted_vs_actual_test_compact.png", (4.0, 3.5), 300, 9, 20, 10000,
           title_across_figure=True)

# ---------------------------------------------------------------- results file
lines = []
add = lines.append
add("# Used car prices: results\n")
add("Generated by `analysis.py`. Every number below is reproduced by running that script.\n")
add("## 1. Data and split\n")
add(f"- Rows in `used_car_sales.csv`: {len(cars)}")
add(f"- Training cars (70%): {len(train)}")
add(f"- Test cars (30%): {len(test)}")
add(f"- Random seed: {SEED}")
add(f"- Auction lots (not used for fitting): {len(lots)}\n")

add("## 2. Model fit on the training cars\n")
add(f"Outcome: `{OUTCOME}` (dollars). Model family: multiple linear regression (OLS).")
add("`Accident_Yes` is 1 if the history report shows an accident, 0 if not.\n")
add("### Full model (all five features)\n")
add(md_table(coef_table(full).reset_index(),
             {"coefficient": "{:,.2f}", "std_error": "{:,.2f}", "p_value": "{:.4f}"}))
add(f"\n- R²: {full.rsquared:.4f}")
add(f"- Adjusted R²: {full.rsquared_adj:.4f}\n")
if reduced is not None:
    add(f"Features with p-value at or above {P_CUTOFF}: {', '.join(weak)}\n")
    add(f"### Reduced model (without {', '.join(weak)})\n")
    add(md_table(coef_table(reduced).reset_index(),
                 {"coefficient": "{:,.2f}", "std_error": "{:,.2f}", "p_value": "{:.4f}"}))
    add(f"\n- R²: {reduced.rsquared:.4f}")
    add(f"- Adjusted R²: {reduced.rsquared_adj:.4f}")
    add(f"- Change in adjusted R² from dropping: {reduced.rsquared_adj - full.rsquared_adj:+.4f}\n")
else:
    add(f"Every feature has a p-value below {P_CUTOFF}, so no reduced model was fitted.\n")
add(f"### Model handed over as v1: {choice}\n")
add(f"- Features in v1: {', '.join(v1_features)}")
add(f"- Dropped: {', '.join(weak) if choice == 'reduced' else 'none'}")
add(f"- Adjusted R² of the training fit: {v1.rsquared_adj:.4f}")
add(f"- Selection setting: HANDOVER = \"{HANDOVER}\" (auto keeps the smaller model if "
    f"adjusted R² falls by less than {ADJ_R2_TOLERANCE}; uses training data only)\n")

add("## 3. Fit on the test cars: baseline vs v1\n")
add(f"Baseline prediction for every car (training average): ${baseline_price:,.2f}\n")
add(md_table(fit_rows, {"test_R2": "{:.4f}", "test_RMSE": "${:,.0f}"}))
add("\nThe baseline R² is at or just below zero by construction: it predicts one "
    "number, and the test cars' own average differs slightly from the training average.\n")
add("Chart: `outputs/predicted_vs_actual_test.png`\n")

add("## 4. Decision on the 40 auction lots\n")
add(f"Rule: buy if auction price + ${PREP_COST:,} is below the predicted sale price, "
    "otherwise pass.")
add(f"Profit on a bought car: later sale price − auction price − ${PREP_COST:,}.\n")
add(md_table(value_rows, {"total_profit": "${:,.0f}"}))
gain = value_rows.loc[1, "total_profit"] - value_rows.loc[0, "total_profit"]
add(f"\n- v1 total profit minus baseline total profit: ${gain:,.0f}")
add(f"- For reference, buying all {len(lots)} lots would total ${buy_everything:,.0f}")
add("- Lot-by-lot detail: `outputs/auction_decisions.csv`")

(OUT / "results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
