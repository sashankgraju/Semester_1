import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

bailey = pd.read_excel("9-8_CSVs/EP_DX_Bailey_11_06_2024_2.xlsx")

eacsf = pd.read_csv("9-8_CSVs/EACSF_SumData_AAL_09_2025.csv")

print(bailey.columns)
print(eacsf.columns)


bailey['CandID'] = bailey['CandID'].astype(str).str.strip()
eacsf['Case'] = eacsf['Case'].astype(str).str.strip()

# Merge datasets on matching subject IDs
merged = pd.merge(eacsf, bailey, left_on='Case', right_on='CandID', how='inner')

print(f"Merged dataset shape: {merged.shape}")
print(merged.head())


# Inspect columns that mention ASD diagnosis or risk // FIX THIS TO REFLECT BAILEY
diagnosis_cols = [col for col in merged.columns if 'ASD' in col or 'diagnosis' in col.lower() or 'risk' in col.lower()]
print("Potential diagnosis columns:", diagnosis_cols)

# Example: suppose 'ASD_status' has 'ASD', 'High Risk', or 'TD' values
# Clean up the labels for consistency

# separate DS first before filtering for ASD
# can fuse TD and Control
if 'ASD_status' in merged.columns:
    merged['ASD_status'] = merged['ASD_status'].replace({
        'Autism': 'ASD',
        'HighRisk': 'HR',
        'HR': 'HR',
        'Typically Developing': 'TD',
        'Control': 'TD'
    })


# AAL region columns begin after column E (e.g., index 4)
aal_columns = merged.columns[4:94]  # adjust based on file structure
aal_data = merged[aal_columns]
aal_data = aal_data.apply(pd.to_numeric, errors='coerce')

# -------------------------------------------------------
# STEP 5. CALCULATE REGIONAL STATISTICS
# -------------------------------------------------------

# Mean CSF per region by group
group_means = merged.groupby('ASD_status')[aal_columns].mean().T
group_means.columns.name = "Group"
group_means.head()

# Difference (ASD - TD)
group_means['Diff_ASD-TD'] = group_means['ASD'] - group_means['TD']
group_means_sorted = group_means.sort_values('Diff_ASD-TD', ascending=False)

# -------------------------------------------------------
# STEP 6. VISUALIZE TRENDS
# -------------------------------------------------------

plt.figure(figsize=(12, 6))
sns.boxplot(data=merged, x='ASD_status', y=aal_columns[0])
plt.title(f"CSF volume in {aal_columns[0]} by diagnostic group")
plt.show()

# Top 10 regions with highest ASD–TD difference
plt.figure(figsize=(10, 6))
sns.barplot(
    y=group_means_sorted.index[:10],
    x=group_means_sorted['Diff_ASD-TD'][:10],
    orient='h'
)
plt.title("Top 10 AAL regions with largest CSF difference (ASD – TD)")
plt.xlabel("Mean CSF Volume Difference")
plt.ylabel("AAL Region")
plt.show()

# -------------------------------------------------------
# STEP 7. OPTIONAL: CORRELATION ANALYSIS
# -------------------------------------------------------

# Correlate total CSF volume with ASD status (binary)
merged['ASD_binary'] = merged['ASD_status'].map({'ASD': 1, 'HR': 0.5, 'TD': 0})
merged['Total_CSF'] = aal_data.sum(axis=1)

correlation = merged[['ASD_binary', 'Total_CSF']].corr().iloc[0,1]
print(f"Correlation between total CSF and ASD_binary: {correlation:.3f}")