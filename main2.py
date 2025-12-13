import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1) LOAD FILES
# =========================
bailey_path = "9-8_CSVs/EP_DX_Bailey_11_06_2024_2.xlsx"
eacsf_path  = "9-8_CSVs/EACSF_SumData_AAL_09_2025.csv"

bailey = pd.read_excel(bailey_path)
eacsf  = pd.read_csv(eacsf_path)

print("Bailey columns (n={}):".format(len(bailey.columns)))
print(list(bailey.columns)[:20], "...")  # preview
print("\nEACSF columns (n={}):".format(len(eacsf.columns)))
print(list(eacsf.columns)[:20], "...")

# =========================
# 2) ALIGN IDS (REAL NAMES)
# Bailey key: 'Identifiers'
# EACSF key: 'Case'
# =========================
if 'Identifiers' not in bailey.columns:
    raise KeyError("Expected 'Identifiers' in Bailey file but didn't find it.")

if 'Case' not in eacsf.columns:
    raise KeyError("Expected 'Case' in EACSF file but didn't find it.")

bailey['Identifiers'] = bailey['Identifiers'].astype(str).str.strip()
eacsf['Case']         = eacsf['Case'].astype(str).str.strip()

merged = eacsf.merge(bailey, left_on='Case', right_on='Identifiers', how='inner')
print(f"\nMerged shape: {merged.shape} (inner join on Case ↔ Identifiers)")

# =========================
# 3) FIND/BUILD ASD STATUS
# Scan Bailey columns for diagnosis/risk signals.
# We won't invent names; we search for likely ones and log what we use.
# =========================
bailey_cols_lower = {c: c.lower() for c in bailey.columns}
candidate_diag_cols = [c for c, lc in bailey_cols_lower.items()
                       if ('asd' in lc) or ('diagnosis' in lc) or ('dx' in lc)
                       or ('autism' in lc) or ('risk' in lc) or ('sib' in lc)]

print("\nCandidate diagnosis/risk columns found in Bailey:")
for c in candidate_diag_cols:
    print("  -", c)

# Heuristic combiner:
# - If any candidate col suggests ASD (string contains 'asd'/'autism' or equals 1), mark ASD.
# - Else if suggests high-risk (contains 'risk'/'sib'), mark HR.
# - Else TD.
def infer_status(row):
    # Aggregate relevant text/numeric tokens from candidate columns
    tokens = []
    for c in candidate_diag_cols:
        val = row.get(c, np.nan)
        if pd.isna(val):
            continue
        s = str(val).strip().lower()
        tokens.append(s)
    bag = " | ".join(tokens)

    # priority: ASD > HR > TD
    if any(k in bag for k in ["asd", "autism", "autistic"]) or "1" in tokens:
        return "ASD"
    if any(k in bag for k in ["high risk", "hr", "sibling", "sib", "brother", "sister", "risk"]):
        return "HR"
    return "TD"

merged['ASD_status'] = merged.apply(infer_status, axis=1)

print("\nASD_status value counts (inferred):")
print(merged['ASD_status'].value_counts(dropna=False))

# If you already know the exact column(s), you can replace the inference above with a direct mapping, e.g.:
# merged['ASD_status'] = merged['<YOUR_COLUMN>'].map({...})

# =========================
# 4) DETECT AAL REGION COLUMNS USING REAL HEADERS
# EACSF shows region columns named as integer strings (1..90).
# We'll pick only columns whose names are all-digits and in [1, 90].
# =========================
def is_aal_col(name):
    s = str(name)
    if s.isdigit():
        i = int(s)
        return 1 <= i <= 90
    return False

aal_cols = [c for c in eacsf.columns if is_aal_col(c)]
if not aal_cols:
    raise RuntimeError("No AAL columns detected (1..90 as digit-only column names).")

print("\nDetected AAL region columns (count={}):".format(len(aal_cols)))
print(aal_cols[:20], "...")

# Cast to numeric safely
for c in aal_cols:
    merged[c] = pd.to_numeric(merged[c], errors='coerce')

# =========================
# 5) BASIC SUMMARY METRICS
# =========================
# total CSF across all AAL regions
merged['Total_CSF'] = merged[aal_cols].sum(axis=1, skipna=True)

# quick group means by diagnostic group
group_means = merged.groupby('ASD_status')[aal_cols].mean(numeric_only=True).T
group_means['Diff_ASD_minus_TD'] = group_means.get('ASD', 0) - group_means.get('TD', 0)
group_means_sorted = group_means.sort_values('Diff_ASD_minus_TD', ascending=False)

print("\nTop 10 regions by (ASD - TD) mean CSF difference:")
print(group_means_sorted['Diff_ASD_minus_TD'].head(10))

# =========================
# 6) QUICK PLOTS (pure matplotlib)
# =========================
def boxplot_by_group(df, group_col, value_col, title=None):
    # Build data arrays per group in a consistent order
    order = ['TD', 'HR', 'ASD'] if set(df[group_col].unique()) >= {'TD','ASD'} else sorted(df[group_col].dropna().unique())
    data = [df.loc[df[group_col] == g, value_col].dropna().values for g in order]
    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=order)
    plt.title(title or f"{value_col} by {group_col}")
    plt.ylabel(value_col)
    plt.tight_layout()
    plt.show()

# Example: first detected AAL region
first_region = str(aal_cols[0])
boxplot_by_group(merged, 'ASD_status', first_region, title=f"AAL {first_region}: CSF by ASD_status")

# Bar chart: top 10 ASD–TD diffs
top = group_means_sorted['Diff_ASD_minus_TD'].head(10)
plt.figure(figsize=(8, 5))
plt.barh([str(i) for i in top.index], top.values)
plt.gca().invert_yaxis()
plt.xlabel("Mean CSF difference (ASD - TD)")
plt.title("Top 10 AAL regions by ASD–TD mean CSF difference")
plt.tight_layout()
plt.show()

# =========================
# 7) SIMPLE WHOLE-BRAIN CHECK
# map TD→0, HR→0.5, ASD→1 for a rough ordinal score
status_map = {'TD': 0.0, 'HR': 0.5, 'ASD': 1.0}
merged['ASD_binary'] = merged['ASD_status'].map(status_map)
corr = merged[['ASD_binary', 'Total_CSF']].corr().iloc[0,1]
print(f"\nCorrelation(ASD_binary, Total_CSF) = {corr:.3f}")

# =========================
# 8) OPTIONAL: SAVE OUT SUMMARY TABLES
# =========================
group_means_sorted.to_csv("aal_group_means_and_diffs.csv")
print("\nSaved: aal_group_means_and_diffs.csv")