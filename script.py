import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm  # NEW: for quantile regression

# 1. Load the data
df = pd.read_csv("nagpra_data_full.csv")

# Make sure numeric columns are actually numeric
num_cols = ["Remains_Not_Made_Available", "Remains_Made_Available", "Percent_Made_Available"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with missing numeric data
df = df.dropna(subset=num_cols)

# Add a total column for convenience
df["Total_Remains"] = df["Remains_Not_Made_Available"] + df["Remains_Made_Available"]

# 2. Filter to "large holders" (you can change 200 if you want)
large = df[df["Remains_Not_Made_Available"] >= 0].copy()

print("Number of large institutions:", len(large))

# 3. Overall statistics for the dataset
total_not = df["Remains_Not_Made_Available"].sum()
total_made = df["Remains_Made_Available"].sum()
total_all = total_not + total_made

overall_pct = (total_made / total_all) * 100 if total_all > 0 else np.nan

print("\n=== Overall summary (all institutions in this file) ===")
print("Total remains not made available for return:", int(total_not))
print("Total remains made available for return:", int(total_made))
print("Overall repatriation rate (percent made available): {:.1f}%".format(overall_pct))

# 4. Stats just for large holders
large_not = large["Remains_Not_Made_Available"].sum()
large_made = large["Remains_Made_Available"].sum()
large_total = large_not + large_made
large_pct = (large_made / large_total) * 100 if large_total > 0 else np.nan

print("\n=== Summary for large holders (>= 200 not made available) ===")
print("Total remains not made available for return (large):", int(large_not))
print("Total remains made available for return (large):", int(large_made))
print("Overall repatriation rate among large holders: {:.1f}%".format(large_pct))

# Distribution of percentages among large holders
pct_min = large["Percent_Made_Available"].min()
pct_max = large["Percent_Made_Available"].max()
pct_mean = large["Percent_Made_Available"].mean()
pct_std = large["Percent_Made_Available"].std()

print("\n=== Distribution of repatriation percentages (large holders) ===")
print("Min percent made available: {:.1f}%".format(pct_min))
print("Max percent made available: {:.1f}%".format(pct_max))
print("Mean percent made available: {:.1f}%".format(pct_mean))
print("Std dev of percent made available: {:.1f}".format(pct_std))

# 4.5 QUANTIFY THE "LOWER TRIANGLE" STORY
# ----------------------------------------

# (a) Bucket institutions by size and compare average repatriation rates
bins = [0, 100, 500, 2000, np.inf]
labels = ["0–99", "100–499", "500–1999", "2000+"]

df["Size_Bin"] = pd.cut(df["Total_Remains"], bins=bins, labels=labels, right=False)

bin_stats = (
    df.groupby("Size_Bin")
      .agg(
          Count=("Institution", "count"),
          Median_Percent_Made=("Percent_Made_Available", "median")
      )
)

print("\n=== Repatriation rates by size of holdings (all institutions) ===")
print(bin_stats.to_string())

# (b) Check how many very large institutions have high repatriation rates
very_large_threshold = 2000     # you can change this
high_rate_threshold = 80        # 80%+

very_large = df[df["Total_Remains"] >= very_large_threshold].copy()
high_rate_very_large = very_large[very_large["Percent_Made_Available"] >= high_rate_threshold]

print("\n=== High performers among very large holders ===")
print("Very large threshold (Total_Remains >= {}):".format(very_large_threshold))
print("Number of very large institutions:", len(very_large))
print("Number of very large institutions with >= {}% repatriation: {}".format(
    high_rate_threshold, len(high_rate_very_large))
)

if len(very_large) > 0:
    frac_high = len(high_rate_very_large) / len(very_large) * 100
    print("Share of very large institutions with >= {}% repatriation: {:.1f}%".format(
        high_rate_threshold, frac_high
    ))

# (c) Correlation with log(total remains) to capture scale effect
df["Log_Total_Remains"] = np.log10(df["Total_Remains"] + 1)
log_corr = df["Log_Total_Remains"].corr(df["Percent_Made_Available"])
print("\nCorrelation between log10(Total_Remains) and Percent_Made_Available:", log_corr)

# (d) Quantile regression on full dataset (this is the "fancy math" bit)
print("\n=== Quantile regression: Percent_Made_Available ~ log10(Total_Remains) ===")
X = sm.add_constant(df["Log_Total_Remains"])
y = df["Percent_Made_Available"]

quantiles = [0.5, 0.9]  # median and upper envelope

qr_results = {}
for q in quantiles:
    model = sm.QuantReg(y, X)
    res = model.fit(q=q)
    qr_results[q] = res
    print(f"Quantile {q:.2f}: Percent = {res.params['const']:.2f} + {res.params['Log_Total_Remains']:.2f} * log10(Total_Remains)")
    
# 5. Rankings: who is holding the most and who is doing the best/worst?

# Top 5 by remains not made available
top_unrepat = large.sort_values("Remains_Not_Made_Available", ascending=False).head(5)
print("\n=== Top 5 institutions by remains NOT made available for return ===")
print(top_unrepat[["Institution", "Remains_Not_Made_Available", "Percent_Made_Available"]].to_string(index=False))

# To identify meaningful "best" and "worst", require at least 500 total remains
eligible = large[large["Total_Remains"] >= 500].copy()

best = eligible.sort_values("Percent_Made_Available", ascending=False).head(5)
worst = eligible.sort_values("Percent_Made_Available", ascending=True).head(5)

print("\n=== Top 5 'best' repatriation performers (among large, >= 500 total) ===")
print(best[["Institution", "Total_Remains", "Percent_Made_Available"]].to_string(index=False))

print("\n=== Top 5 'worst' repatriation performers (among large, >= 500 total) ===")
print(worst[["Institution", "Total_Remains", "Percent_Made_Available"]].to_string(index=False))

# ========= CUSTOM AXIS RANGES =========
# Set these to numbers if you want to fix the range,
# or leave as None to let matplotlib auto-scale.

SCATTER_X_MIN = 0      # for large-holders scatter
SCATTER_X_MAX = 9000
SCATTER_Y_MIN = 0
SCATTER_Y_MAX = 100

HIST_X_MIN = None
HIST_X_MAX = None

# For the quantile-regression scatter using Total_Remains (all institutions)
QR_X_MIN = 0
QR_X_MAX = df["Total_Remains"].max()  # or set manually, e.g., 8000
QR_Y_MIN = 0
QR_Y_MAX = 100
# =====================================

# 6. Correlation and scatter plot (large holders only)
x = large["Remains_Not_Made_Available"]
y = large["Percent_Made_Available"]

corr = x.corr(y)
print("\nCorrelation between remains not made available and percent made available (large):", corr)

plt.figure(figsize=(8, 6))
plt.scatter(x, y)

plt.xlabel("Remains Not Made Available for Return")
plt.ylabel("Percent of Remains Made Available for Return")
plt.title("NAGPRA Repatriation Rates for Large Holding Institutions")

# Apply custom axis limits if provided
if SCATTER_X_MIN is not None and SCATTER_X_MAX is not None:
    plt.xlim(SCATTER_X_MIN, SCATTER_X_MAX)
if SCATTER_Y_MIN is not None and SCATTER_Y_MAX is not None:
    plt.ylim(SCATTER_Y_MIN, SCATTER_Y_MAX)

key_names = [
    "Ohio History Connection",
    "Illinois State Museum",
    "Harvard Univ.",
    "Univ. of California, Berkeley",
    "Univ. of Alabama",
    "Robert S. Peabody Institute of Archaeology",
    "Univ. of Arkansas"
]

for _, row in large[large["Institution"].isin(key_names)].iterrows():
    plt.annotate(
        row["Institution"],
        (row["Remains_Not_Made_Available"], row["Percent_Made_Available"]),
        textcoords="offset points",
        xytext=(5, 5),
        fontsize=8
    )

plt.tight_layout()
plt.show()

# 6.5 Quantile regression plot (all institutions, shows "lower triangle" more clearly)
plt.figure(figsize=(8, 6))
plt.scatter(df["Total_Remains"], df["Percent_Made_Available"], alpha=0.4, label="Institutions")

# Create a grid of x-values (Total_Remains) and plot quantile regression lines
x_grid = np.linspace(QR_X_MIN, QR_X_MAX, 100)
log_x_grid = np.log10(x_grid + 1)
X_grid = sm.add_constant(log_x_grid)

colors = {0.5: "red", 0.9: "green"}
labels_q = {0.5: "Median (50th percentile)", 0.9: "Upper envelope (90th percentile)"}

for q, res in qr_results.items():
    y_pred = X_grid @ res.params
    plt.plot(x_grid, y_pred, color=colors[q], label=labels_q[q])

plt.xlabel("Total Native American Remains Reported")
plt.ylabel("Percent of Remains Made Available for Return")
plt.title("Quantile Regression of Repatriation Rates vs. Size of Holdings")

if QR_X_MIN is not None and QR_X_MAX is not None:
    plt.xlim(QR_X_MIN, QR_X_MAX)
if QR_Y_MIN is not None and QR_Y_MAX is not None:
    plt.ylim(QR_Y_MIN, QR_Y_MAX)

plt.legend()
plt.tight_layout()
plt.show()

# 7. Histogram of repatriation percentages (large holders)
plt.figure(figsize=(8, 6))
plt.hist(large["Percent_Made_Available"], bins=10)

plt.xlabel("Percent of Remains Made Available for Return")
plt.ylabel("Number of Institutions")
plt.title("Distribution of NAGPRA Repatriation Rates Among Large Holders")

# Optional custom x-limits for histogram
if HIST_X_MIN is not None and HIST_X_MAX is not None:
    plt.xlim(HIST_X_MIN, HIST_X_MAX)

plt.tight_layout()
plt.show()

# 8. Small summary table for the institutions you want to mention in the essay
summary = df[df["Institution"].isin(key_names)].copy()
summary = summary[["Institution", "Remains_Not_Made_Available", "Remains_Made_Available", "Percent_Made_Available"]]
print("\nKey institutions for the essay:")
print(summary.to_string(index=False))
