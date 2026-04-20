# =========================================================
# Exploratory Data Analysis (EDA) on Cancer Dataset
# =========================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.preprocessing import StandardScaler

# =========================================================
# LOAD DATA
# =========================================================

data = pd.read_csv(
    'C:/Users/tta20/OneDrive - University of Virginia/BME 2315 (Comp)/Module 3/Module-3-Fibrosis/Module-4-Cancer/data/TRAINING_SET_GSE62944_subsample_log2TPM.csv',
    index_col=0, header=0
)

metadata_df = pd.read_csv(
    'C:/Users/tta20/OneDrive - University of Virginia/BME 2315 (Comp)/Module 3/Module-3-Fibrosis/Module-4-Cancer/data/TRAINING_SET_GSE62944_metadata.csv',
    index_col=0, header=0
)

print(data.head())

# =========================================================
# LOAD ANGIOGENESIS GENE LIST
# =========================================================

df = pd.read_csv(
    r"C:\Users\tta20\Downloads\Menyhart_JPA_CancerHallmarks_core.txt",
    sep='\t'
)

df.columns = df.columns.astype(str)

angiogenesis_row = df[df.iloc[:, 0] == "SUSTAINED ANGIOGENESIS"]

genes = angiogenesis_row.iloc[0, 1:].dropna()

gene_list = genes.tolist()

# =========================================================
# EXPLORE DATA
# =========================================================

print(data.shape)
print(data.info())
print(data.describe())

print(metadata_df.info())
print(metadata_df.describe())

# =========================================================
# SUBSET GBM
# =========================================================

cancer_type = 'GBM'

cancer_samples = metadata_df[metadata_df['cancer_type'] == cancer_type].index

GBM_data = data[cancer_samples]
GBM_metadata = metadata_df.loc[cancer_samples]

# =========================================================
# CLEANING
# =========================================================

# Remove duplicates
GBM_data = GBM_data.loc[~GBM_data.index.duplicated(keep='first')]
GBM_metadata = GBM_metadata.loc[~GBM_metadata.index.duplicated(keep='first')]

# Missing values (genes)
GBM_data = GBM_data.apply(lambda row: row.fillna(row.mean()), axis=1)
GBM_data = GBM_data.fillna(0)

# Missing values (metadata)
for col in GBM_metadata.columns:
    if GBM_metadata[col].dtype == "object":
        mode_vals = GBM_metadata[col].mode()
        GBM_metadata[col] = GBM_metadata[col].fillna(mode_vals[0] if len(mode_vals) > 0 else "Unknown")
    else:
        GBM_metadata[col] = GBM_metadata[col].fillna(GBM_metadata[col].median())

# Ensure numeric
GBM_data = GBM_data.apply(pd.to_numeric, errors='coerce')

# =========================================================
# OUTLIER HANDLING
# =========================================================

Q1 = GBM_data.quantile(0.25, axis=1)
Q3 = GBM_data.quantile(0.75, axis=1)
IQR = Q3 - Q1

GBM_data = GBM_data.clip(
    lower=Q1 - 1.5 * IQR,
    upper=Q3 + 1.5 * IQR,
    axis=0
)

# =========================================================
# NORMALIZATION
# =========================================================

scaler = StandardScaler()

GBM_data_scaled = pd.DataFrame(
    scaler.fit_transform(GBM_data.T),
    index=GBM_data.columns,
    columns=GBM_data.index
).T

GBM_data = GBM_data_scaled

print("Final shape:", GBM_data.shape)

# =========================================================
# ANGIOGENESIS GENE FILTERING
# =========================================================

desired_gene_list = gene_list

gene_list = [gene for gene in desired_gene_list if gene in GBM_data.index]

for gene in desired_gene_list:
    if gene not in gene_list:
        print(f"Warning: {gene} not found in dataset.")

GBM_gene_data = GBM_data.loc[gene_list]

print(GBM_gene_data.head())

# =========================================================
# BASIC STATS
# =========================================================

print(GBM_gene_data.describe())
print(GBM_gene_data.var(axis=1))
print(GBM_gene_data.mean(axis=1))
print(GBM_gene_data.median(axis=1))

# =========================================================
# METADATA ANALYSIS
# =========================================================

print(metadata_df.groupby('cancer_type')["gender"].value_counts())

metadata_df['age_at_diagnosis'] = pd.to_numeric(
    metadata_df['age_at_diagnosis'],
    errors='coerce'
)

print(metadata_df.groupby('cancer_type')["age_at_diagnosis"].mean())

# =========================================================
# MERGE DATASETS
# =========================================================

GBM_merged = GBM_gene_data.T.merge(
    GBM_metadata,
    left_index=True,
    right_index=True
)

print(GBM_merged.head())

# =========================================================
# PLOTTING
# =========================================================

sns.boxplot(data=GBM_merged, x="gender", y='EGFR')
plt.title("EGFR Expression by Gender in GBM Samples")
plt.show()

GBM_merged[['MYC', 'EGFR']].plot.box()
plt.title("MYC and EGFR Expression in GBM Samples")
plt.show()

# =========================================================
# PCA (ANGIOGENESIS GENE SET)
# =========================================================

X = GBM_gene_data.T  # samples as rows

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    s=100
)

plt.title("PCA of Angiogenesis Genes in GBM")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.show()

# =========================================================
# UMAP
# =========================================================


# Use same feature matrix as PCA (samples x genes)
X = GBM_gene_data.T  # samples as rows

# Standardize
scaled_data = StandardScaler().fit_transform(X)

# UMAP reduction
reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    random_state=42
)

X_umap = reducer.fit_transform(scaled_data)

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=X_umap[:, 0],
    y=X_umap[:, 1],
    s=100
)

plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.title("UMAP Projection of Angiogenesis Genes in GBM")

plt.show()