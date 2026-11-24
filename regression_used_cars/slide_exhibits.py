"""Builds one web page with the tables and charts for the used car slides.

Runs analysis.py first, so every figure on the page comes from the same fit.

Run from anywhere:  python regression_used_cars/slide_exhibits.py
Writes outputs/slide_exhibits.html and outputs/auction_profit.png.
"""

import html

import matplotlib.pyplot as plt

import analysis as a   # runs the full analysis and exposes its results

OUT = a.OUT

FEATURE_LABELS = {
    "Mileage_K": "Mileage, per 1,000 miles",
    "Age_Years": "Age, per year",
    "Horsepower": "Horsepower, per 1 hp",
    "Accident_Yes": "Accident on record",
    "Num_Owners": "Number of owners, per owner",
}


def dollars(value, signed=False):
    sign = "−" if value < 0 else ("+" if signed else "")
    return f"{sign}${abs(value):,.0f}"


def p_text(p):
    return "below 0.001" if p < 0.001 else f"{p:.2f}"


# ---------------------------------------------------------------- profit chart
base, v1 = a.value_rows.iloc[0], a.value_rows.iloc[1]
fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200)
labels = [
    f"Baseline\n{base['cars_bought']} cars bought, {base['money_losing_cars_bought']} lose money",
    f"v1\n{v1['cars_bought']} cars bought, {v1['money_losing_cars_bought']} lose money",
]
profits = [base["total_profit"], v1["total_profit"]]
bars = ax.bar(labels, profits, width=0.5, color=[a.ORANGE, a.BLUE])
ax.axhline(0, color=a.MUTED, linewidth=1)
span = max(profits) - min(profits)
for bar, profit in zip(bars, profits):
    offset = span * 0.03 * (1 if profit >= 0 else -1)
    ax.text(bar.get_x() + bar.get_width() / 2, profit + offset, dollars(profit, signed=True),
            ha="center", va="bottom" if profit >= 0 else "top", fontsize=13,
            fontweight="bold", color=a.INK)
ax.set_ylim(min(profits) - span * 0.18, max(profits) + span * 0.15)
ax.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda v, _: dollars(v) if v else "$0"))
ax.grid(axis="y", color=a.GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["bottom"].set_visible(False)
ax.tick_params(axis="x", length=0, labelsize=11, colors=a.INK)
ax.set_ylabel("Total profit")
ax.set_title(f"Total profit on the {len(a.lots)} auction lots", loc="left", fontsize=12,
             fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "auction_profit.png", facecolor="white")
plt.close(fig)


# ---------------------------------------------------------------- tables
def table(headers, rows, numeric_from=1, muted_rows=()):
    out = ["<table>", "<thead><tr>"]
    for i, h in enumerate(headers):
        cls = ' class="num"' if i >= numeric_from else ""
        out.append(f"<th{cls}>{html.escape(h)}</th>")
    out.append("</tr></thead><tbody>")
    for r, row in enumerate(rows):
        out.append('<tr class="muted">' if r in muted_rows else "<tr>")
        for i, cell in enumerate(row):
            cls = ' class="num"' if i >= numeric_from else ""
            out.append(f"<td{cls}>{html.escape(str(cell))}</td>")
        out.append("</tr>")
    out.append("</tbody></table>")
    return "".join(out)


dropped = a.weak if a.choice == "reduced" else []
data_model = table(["", ""], [
    ["Data", f"{len(a.cars)} cars sold by the dealer in the last year"],
    ["Split", f"{len(a.train)} training cars (70%), {len(a.test)} test cars (30%)"],
    ["Outcome", "Sale price, in dollars"],
    ["Model", "Linear regression"],
    ["Features", ", ".join(FEATURE_LABELS[f].split(",")[0] for f in a.v1_features)],
    ["Dropped", ", ".join(FEATURE_LABELS[f].split(",")[0] for f in dropped) or "None"],
    ["Adjusted R², training cars", f"{a.v1.rsquared_adj:.2f}"],
], numeric_from=9)

coef_rows = [[FEATURE_LABELS[f], dollars(a.v1.params[f], signed=True), p_text(a.v1.pvalues[f])]
             for f in a.v1_features]
for f in dropped:
    coef_rows.append([FEATURE_LABELS[f] + " (dropped)", dollars(a.full.params[f], signed=True),
                      p_text(a.full.pvalues[f])])
coefficients = table(["Feature", "Effect on sale price", "p-value"], coef_rows,
                     muted_rows=range(len(a.v1_features), len(coef_rows)))

fit = table(["", "R²", "Typical error"], [
    [f"Baseline (every car at {dollars(a.baseline_price)})",
     f"{a.fit_rows.loc[0, 'test_R2']:.2f}".replace("-", "−"),
     dollars(a.fit_rows.loc[0, "test_RMSE"])],
    ["v1 (linear regression)",
     f"{a.fit_rows.loc[1, 'test_R2']:.2f}".replace("-", "−"),
     dollars(a.fit_rows.loc[1, "test_RMSE"])],
])

value = table(["", "Cars bought", "Money-losing cars", "Total profit"], [
    ["Baseline", base["cars_bought"], base["money_losing_cars_bought"],
     dollars(base["total_profit"], signed=True)],
    ["v1", v1["cars_bought"], v1["money_losing_cars_bought"],
     dollars(v1["total_profit"], signed=True)],
])

# ---------------------------------------------------------------- page
PAGE = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Used car slides: exhibits</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ margin: 0; background: #ffffff; color: #000000;
         font-family: Helvetica, Arial, sans-serif; font-size: 18px; line-height: 1.35; }}
  main {{ max-width: 860px; margin: 0 auto; padding: 32px 24px 96px; }}
  h1 {{ font-size: 20px; margin: 0 0 4px; }}
  .note {{ color: #6b6b6b; font-size: 14px; margin: 0 0 8px; }}
  section {{ margin-top: 72px; }}
  .tag {{ color: #6b6b6b; font-size: 13px; letter-spacing: 0.04em; text-transform: uppercase;
         margin: 0 0 20px; padding-bottom: 8px; border-bottom: 1px solid #dddddd; }}
  .exhibit {{ display: inline-block; background: #ffffff; padding: 12px 16px; }}
  .title {{ font-weight: 700; margin: 0 0 10px; }}
  table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; }}
  th, td {{ text-align: left; padding: 9px 28px 9px 0; border-bottom: 1px solid #c8c8c8;
           white-space: nowrap; }}
  th:last-child, td:last-child {{ padding-right: 0; }}
  thead th {{ border-bottom: 2px solid #000000; font-weight: 700; }}
  tbody tr:last-child td {{ border-bottom: 2px solid #000000; }}
  .num {{ text-align: right; padding-left: 28px; }}
  tr.muted td {{ color: #6b6b6b; }}
  img {{ display: block; width: 720px; max-width: 100%; height: auto; }}
</style>
</head>
<body>
<main>
  <h1>Used car slides: exhibits</h1>
  <p class="note">Built by slide_exhibits.py from the same fit as outputs/results.md.
  Screenshot the white block under each grey label.</p>

  <section>
    <p class="tag">Slide 1 · Data and model</p>
    <div class="exhibit">{data_model}</div>
  </section>

  <section>
    <p class="tag">Slide 1 · What each feature is worth</p>
    <div class="exhibit">{coefficients}</div>
  </section>

  <section>
    <p class="tag">Slide 1 · Fit on the {len(a.test)} test cars</p>
    <div class="exhibit">{fit}</div>
  </section>

  <section>
    <p class="tag">Slide 1 · Predicted vs actual chart</p>
    <div class="exhibit"><img src="predicted_vs_actual_test.png"
      alt="Predicted against actual sale price for the test cars"></div>
  </section>

  <section>
    <p class="tag">Slide 2 · Value on the {len(a.lots)} auction lots</p>
    <div class="exhibit">{value}</div>
  </section>

  <section>
    <p class="tag">Slide 2 · Profit chart</p>
    <div class="exhibit"><img src="auction_profit.png"
      alt="Total profit on the auction lots, baseline against v1"></div>
  </section>
</main>
</body>
</html>
"""
(OUT / "slide_exhibits.html").write_text(PAGE)
print(f"Wrote {OUT / 'slide_exhibits.html'}")
