"""
HR Data Cleaning & Visualization Pipeline
==========================================
Covers:
  - Generating a realistic messy HR dataset
  - Handling missing values, outliers, duplicates
  - Standardising inconsistent categorical data
  - Building a 10-panel analytics dashboard
"""

import random
import warnings
from datetime import datetime, timedelta

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

# ══════════════════════════════════════════════════════════════════
# STEP 1 — GENERATE A REALISTIC (MESSY) HR DATASET
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1 — Generating raw HR dataset")
print("=" * 60)

N = 500

DEPARTMENTS = ["Engineering", "Sales", "HR", "Finance", "Marketing", "Operations"]
JOB_ROLES = {
    "Engineering": ["Software Engineer", "Data Scientist", "DevOps Engineer"],
    "Sales":       ["Sales Rep", "Account Manager", "Sales Manager"],
    "HR":          ["HR Specialist", "Recruiter", "HR Manager"],
    "Finance":     ["Financial Analyst", "Accountant", "Finance Manager"],
    "Marketing":   ["Marketing Specialist", "SEO Analyst", "Marketing Manager"],
    "Operations":  ["Operations Analyst", "Logistics Coordinator", "Ops Manager"],
}
DEPT_BASE_SALARY = {
    "Engineering": 85_000, "Sales": 65_000, "HR": 55_000,
    "Finance": 75_000, "Marketing": 60_000, "Operations": 58_000,
}

# Core columns
dept_col   = np.random.choice(DEPARTMENTS, N)
hire_dates = [datetime(2010, 1, 1) + timedelta(days=random.randint(0, 365 * 12))
              for _ in range(N)]
tenure     = np.round([(datetime(2024, 1, 1) - h).days / 365 for h in hire_dates], 1)
salaries   = np.array([
    np.random.normal(DEPT_BASE_SALARY[d] + 3_000 * y, 8_000)
    for d, y in zip(dept_col, tenure)
]).round(2)

# Inconsistent gender encoding  ← real-world problem
gender_raw  = np.random.choice(["Male", "Female"], N, p=[0.6, 0.4])
GENDER_TYPOS = {"Male": ["Male", "male", "M", "MALE"], "Female": ["Female", "female", "F", "FEMALE"]}
gender_noisy = [random.choice(GENDER_TYPOS[g]) if random.random() < 0.3 else g
                for g in gender_raw]

df = pd.DataFrame({
    "EmployeeID":      range(1001, 1001 + N),
    "Department":      dept_col,
    "JobRole":         [random.choice(JOB_ROLES[d]) for d in dept_col],
    "Gender":          gender_noisy,
    "Age":             np.random.randint(22, 60, N).astype(float),
    "Salary":          salaries,
    "YearsAtCompany":  tenure,
    "MonthlyHours":    np.random.normal(170, 20, N).round(0),
    "PerformanceRating": np.random.choice([1, 2, 3, 4, 5], N, p=[0.05, 0.15, 0.35, 0.30, 0.15]),
    "WorkLifeBalance": np.random.choice([1, 2, 3, 4], N, p=[0.10, 0.25, 0.40, 0.25]),
    "Attrition":       np.random.choice(["Yes", "No"], N, p=[0.20, 0.80]),
    "HireDate":        [d.strftime("%Y-%m-%d") for d in hire_dates],
})

# ── Inject missing values ──────────────────────────────────────────
for col, count in [("Salary", 30), ("Age", 20), ("Department", 15), ("WorkLifeBalance", 10)]:
    df.loc[np.random.choice(df.index, count, replace=False), col] = np.nan

# ── Inject salary & hours outliers ────────────────────────────────
df.loc[np.random.choice(df.index, 6, replace=False), "Salary"] = \
    np.random.choice([250_000, 280_000, 320_000, 360_000, 400_000, 420_000], 6)
df.loc[np.random.choice(df.index, 6, replace=False), "MonthlyHours"] = \
    np.random.choice([350, 380, 400, 420], 6)

# ── Inject duplicate rows ─────────────────────────────────────────
df = pd.concat([df, df.sample(15)], ignore_index=True)
df_raw = df.copy()

print(f"  Rows × Cols  : {df_raw.shape}")
print(f"  Duplicates   : {df_raw.duplicated().sum()}")
print(f"  Missing (per col):\n{df_raw.isnull().sum()[df_raw.isnull().sum() > 0].to_string()}")


# ══════════════════════════════════════════════════════════════════
# STEP 2 — DATA CLEANING
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 2 — Cleaning")
print("=" * 60)

df_clean = df_raw.copy()

# ── 2a. Remove duplicates ──────────────────────────────────────────
before = len(df_clean)
df_clean.drop_duplicates(inplace=True)
df_clean.reset_index(drop=True, inplace=True)
print(f"  [Duplicates]  Removed {before - len(df_clean)} rows → {len(df_clean)} remain")

# ── 2b. Standardise Gender ────────────────────────────────────────
def standardise_gender(val):
    if pd.isna(val):
        return np.nan
    v = str(val).strip().lower()
    if v in ("male", "m"):
        return "Male"
    if v in ("female", "f"):
        return "Female"
    return val.title()

noisy_count = (df_clean["Gender"].str.strip() != df_clean["Gender"].apply(standardise_gender)).sum()
df_clean["Gender"] = df_clean["Gender"].apply(standardise_gender)
print(f"  [Gender]      Standardised {noisy_count} inconsistent values")

# ── 2c. Impute missing values ─────────────────────────────────────
# ORDER MATTERS: fill Department first so the Salary groupby sees no nulls in the key.
# Department → mode
df_clean["Department"] = df_clean["Department"].fillna(df_clean["Department"].mode()[0])
# Salary  → department-level median (preserves pay structure)
df_clean["Salary"] = df_clean.groupby("Department")["Salary"].transform(
    lambda x: x.fillna(x.median())
)
# Age     → global median
df_clean["Age"] = df_clean["Age"].fillna(df_clean["Age"].median())
# WorkLifeBalance → mode
df_clean["WorkLifeBalance"] = df_clean["WorkLifeBalance"].fillna(
    df_clean["WorkLifeBalance"].mode()[0]
)
print(f"  [Missing]     Imputed. Remaining nulls: {df_clean.isnull().sum().sum()}")

# ── 2d. Cap outliers with IQR method ─────────────────────────────
def cap_outliers(series, k=3.0):
    """Winsorise values outside k × IQR from the quartiles."""
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    n_capped = ((series < lo) | (series > hi)).sum()
    return series.clip(lo, hi), lo, hi, n_capped

df_clean["Salary"],        s_lo, s_hi, s_n = cap_outliers(df_clean["Salary"])
df_clean["MonthlyHours"],  h_lo, h_hi, h_n = cap_outliers(df_clean["MonthlyHours"])
print(f"  [Outliers]    Salary → capped {s_n} values to [${s_lo:,.0f} – ${s_hi:,.0f}]")
print(f"  [Outliers]    MonthlyHours → capped {h_n} values to [{h_lo:.0f} – {h_hi:.0f}]")

# ── 2e. Fix data types ────────────────────────────────────────────
df_clean["Age"]              = df_clean["Age"].astype(int)
df_clean["HireDate"]         = pd.to_datetime(df_clean["HireDate"])
df_clean["PerformanceRating"] = df_clean["PerformanceRating"].astype(int)
df_clean["WorkLifeBalance"]  = df_clean["WorkLifeBalance"].astype(int)
print(f"\n  ✔ Final cleaned shape: {df_clean.shape}")

# Save clean CSV
df_clean.to_csv("/mnt/user-data/outputs/hr_clean.csv", index=False)
print("  ✔ hr_clean.csv saved")


# ══════════════════════════════════════════════════════════════════
# STEP 3 — VISUALISATIONS  (10-panel dashboard)
# ══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("STEP 3 — Building dashboard")
print("=" * 60)

DEPT_COLORS = {
    "Engineering": "#4361EE", "Sales": "#F72585", "HR": "#7209B7",
    "Finance":     "#3A0CA3", "Marketing": "#4CC9F0", "Operations": "#F77F00",
}
PERF_COLORS = ["#E63946", "#F4A261", "#FFD166", "#06D6A0", "#118AB2"]

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 12,
})

fig = plt.figure(figsize=(22, 28), facecolor="#F8F9FA")
fig.suptitle(
    "HR Analytics Dashboard  ·  Data Cleaning & Insights Report",
    fontsize=22, fontweight="bold", color="#1A1A2E", y=0.99,
)
gs = gridspec.GridSpec(
    4, 3, figure=fig,
    hspace=0.55, wspace=0.38,
    left=0.06, right=0.96, top=0.96, bottom=0.03,
)

# ── ① Missing-value bar (before) ─────────────────────────────────
ax0 = fig.add_subplot(gs[0, 0])
miss = (df_raw[["Salary", "Age", "Department", "WorkLifeBalance"]]
        .isnull().mean() * 100).sort_values(ascending=True)
ax0.barh(miss.index, miss.values, color=["#F4A261", "#F4A261", "#E63946", "#E63946"])
for i, (col, pct) in enumerate(miss.items()):
    ax0.text(pct + 0.2, i, f"{pct:.1f}%", va="center", fontsize=9)
ax0.set_xlabel("% Missing", fontsize=10)
ax0.set_xlim(0, miss.max() * 1.5)
ax0.set_title("① Missing Values (Before Cleaning)", fontweight="bold", fontsize=11)

# ── ② Data quality scorecard ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 1])
ax1.axis("off")
ax1.set_title("② Data Quality Scorecard", fontweight="bold", fontsize=11)
rows = [
    ("Raw rows",           f"{len(df_raw):,}",             "#E63946"),
    ("After dedup",        f"{len(df_clean):,}",           "#2B9348"),
    ("Duplicates removed", f"{len(df_raw)-len(df_clean)}", "#E63946"),
    ("Nulls imputed",      "75",                           "#2B9348"),
    ("Outliers capped",    f"{s_n + h_n}",                 "#F4A261"),
    ("Gender normalised",  f"~{noisy_count} values",       "#2B9348"),
]
y = 0.87
for label, val, col in rows:
    ax1.text(0.05, y, label + ":", fontsize=10, color="#555", transform=ax1.transAxes)
    ax1.text(0.68, y, val, fontsize=10, fontweight="bold", color=col, transform=ax1.transAxes)
    y -= 0.13
ax1.add_patch(mpatches.FancyBboxPatch(
    (0.02, 0.02), 0.96, 0.96,
    boxstyle="round,pad=0.02", facecolor="white",
    edgecolor="#DEE2E6", transform=ax1.transAxes, zorder=0,
))

# ── ③ Salary distribution before vs after ────────────────────────
ax2 = fig.add_subplot(gs[0, 2])
ax2.hist(df_raw["Salary"].dropna(),   bins=45, alpha=0.50, color="#E63946",
         label="Before (raw)", density=True)
ax2.hist(df_clean["Salary"].dropna(), bins=45, alpha=0.60, color="#4361EE",
         label="After (cleaned)", density=True)
ax2.set_xlabel("Salary ($)", fontsize=10)
ax2.set_ylabel("Density", fontsize=10)
ax2.set_title("③ Salary Distribution: Before vs After", fontweight="bold", fontsize=11)
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

# ── ④ Avg salary by department ───────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
dept_sal = df_clean.groupby("Department")["Salary"].mean().sort_values()
ax3.barh(dept_sal.index, dept_sal.values,
         color=[DEPT_COLORS[d] for d in dept_sal.index])
for i, (d, v) in enumerate(dept_sal.items()):
    ax3.text(v + 400, i, f"${v:,.0f}", va="center", fontsize=9)
ax3.set_xlabel("Average Salary ($)", fontsize=10)
ax3.set_xlim(0, dept_sal.max() * 1.22)
ax3.set_title("④ Avg. Salary by Department", fontweight="bold", fontsize=11)
ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))

# ── ⑤ Attrition rate by department ───────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
attr = (df_clean.groupby("Department")["Attrition"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False))
ax4.bar(attr.index, attr.values, color=[DEPT_COLORS[d] for d in attr.index])
ax4.axhline(attr.mean(), color="#E63946", ls="--", lw=1.5,
            label=f"Avg {attr.mean():.1f}%")
for i, v in enumerate(attr.values):
    ax4.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)
ax4.set_ylabel("Attrition Rate (%)", fontsize=10)
ax4.tick_params(axis="x", rotation=28, labelsize=9)
ax4.legend(fontsize=9)
ax4.set_title("⑤ Attrition Rate by Department", fontweight="bold", fontsize=11)

# ── ⑥ Age distribution ───────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
ax5.hist(df_clean["Age"], bins=22, color="#7209B7", edgecolor="white", linewidth=0.6)
ax5.axvline(df_clean["Age"].mean(),   color="#F72585", lw=2, ls="--",
            label=f"Mean {df_clean['Age'].mean():.1f}")
ax5.axvline(df_clean["Age"].median(), color="#4CC9F0", lw=2, ls=":",
            label=f"Median {df_clean['Age'].median():.0f}")
ax5.set_xlabel("Age", fontsize=10)
ax5.set_ylabel("Count", fontsize=10)
ax5.legend(fontsize=9)
ax5.set_title("⑥ Employee Age Distribution", fontweight="bold", fontsize=11)

# ── ⑦ Salary vs tenure scatter ───────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0:2])
for dept, grp in df_clean.groupby("Department"):
    ax6.scatter(grp["YearsAtCompany"], grp["Salary"],
                alpha=0.45, s=22, color=DEPT_COLORS[dept], label=dept)
z = np.polyfit(df_clean["YearsAtCompany"], df_clean["Salary"], 1)
x_line = np.linspace(df_clean["YearsAtCompany"].min(),
                     df_clean["YearsAtCompany"].max(), 100)
ax6.plot(x_line, np.poly1d(z)(x_line), color="#1A1A2E", lw=2, ls="--", label="Trend")
ax6.set_xlabel("Years at Company", fontsize=10)
ax6.set_ylabel("Salary ($)", fontsize=10)
ax6.legend(fontsize=8, ncol=4, loc="upper left")
ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
ax6.set_title("⑦ Salary vs. Tenure (coloured by Department)", fontweight="bold", fontsize=11)

# ── ⑧ Performance rating pie ─────────────────────────────────────
ax7 = fig.add_subplot(gs[2, 2])
perf_labels = {1: "Poor", 2: "Below Avg", 3: "Average", 4: "Good", 5: "Excellent"}
perf_counts = df_clean["PerformanceRating"].value_counts().sort_index()
wedges, _, autotexts = ax7.pie(
    perf_counts.values,
    labels=[perf_labels[i] for i in perf_counts.index],
    colors=PERF_COLORS,
    autopct="%1.1f%%", startangle=140,
    pctdistance=0.75,
    wedgeprops=dict(edgecolor="white", linewidth=2),
)
for t in autotexts:
    t.set_fontsize(8)
ax7.set_title("⑧ Performance Rating Distribution", fontweight="bold", fontsize=11)

# ── ⑨ Correlation heatmap ────────────────────────────────────────
ax8 = fig.add_subplot(gs[3, 0:2])
num_cols = ["Age", "Salary", "YearsAtCompany", "MonthlyHours",
            "PerformanceRating", "WorkLifeBalance"]
corr = df_clean[num_cols].corr()
sns.heatmap(
    corr, ax=ax8,
    annot=True, fmt=".2f", linewidths=0.5,
    cmap="coolwarm", center=0, vmin=-1, vmax=1,
    annot_kws={"size": 9},
    cbar_kws={"shrink": 0.8},
)
ax8.set_title("⑨ Numeric Feature Correlation Matrix", fontweight="bold", fontsize=11)
ax8.tick_params(axis="x", rotation=30, labelsize=9)
ax8.tick_params(axis="y", rotation=0,  labelsize=9)

# ── ⑩ Attrition by gender (stacked %) ───────────────────────────
ax9 = fig.add_subplot(gs[3, 2])
ga = (df_clean.groupby(["Gender", "Attrition"])
      .size().unstack(fill_value=0))
ga_pct = ga.div(ga.sum(axis=1), axis=0) * 100
ga_pct.plot(kind="bar", ax=ax9,
            color=["#4361EE", "#E63946"],
            edgecolor="white", width=0.55)
ax9.set_ylabel("Percentage (%)", fontsize=10)
ax9.tick_params(axis="x", rotation=0)
ax9.legend(["No Attrition", "Attrition"], fontsize=9)
for container in ax9.containers:
    ax9.bar_label(container, fmt="%.1f%%", fontsize=8, padding=2)
ax9.set_title("⑩ Attrition by Gender", fontweight="bold", fontsize=11)

# ── Footer ────────────────────────────────────────────────────────
fig.text(
    0.5, 0.005,
    f"Generated {datetime.now().strftime('%B %d, %Y')}  ·  "
    "500 employees  ·  Libraries: Pandas · NumPy · Matplotlib · Seaborn",
    ha="center", fontsize=9, color="#999", style="italic",
)

out_png = "/mnt/user-data/outputs/hr_dashboard.png"
plt.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="#F8F9FA")
plt.close()
print(f"  ✔ Dashboard saved → {out_png}")

# Save the script itself too
import shutil
shutil.copy(__file__, "/mnt/user-data/outputs/hr_pipeline.py")
print("  ✔ hr_pipeline.py saved")
print("\nDone! 🎉")
