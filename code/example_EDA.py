# Exploratory data analysis (EDA) on a cancer dataset
# Loading the files and exploring the data with pandas
# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# %%
# Load the data
####################################################
data = pd.read_csv(
    'c:/Users/vkb5cq/Desktop/Spring 2026/BME 2315/Module-4-Cancer/data/TRAINING_SET_GSE62944_subsample_log2TPM.csv', index_col=0, header=0)  # can also use larger dataset with more genes
metadata_df = pd.read_csv(
    'c:/Users/vkb5cq/Desktop/Spring 2026/BME 2315/Module-4-Cancer/data/TRAINING_SET_GSE62944_metadata.csv', index_col=0, header=0)
print(data.head())

df = pd.read_csv(r'C:\Users\vkb5cq\Desktop\Spring 2026\BME 2315\Module-4-Cancer\Menyhart_JPA_CancerHallmarks_core.txt',sep='\t')

angiogenesis_row = df[df[df.columns[0]] == 'SUSTAINED ANGIOGENESIS']

genes = angiogenesis_row.iloc[0, 1:]  # skip the first column (label)
genes = genes.dropna()  # remove empty cells

gene_list = genes.tolist()

# %%
# Explore the data
####################################################
print(data.shape)
print(data.info())
print(data.describe())

# %%
# Explore the metadata
####################################################
print(metadata_df.info())
print(metadata_df.describe())

# %%
# Subset the data for a specific cancer type
####################################################
cancer_type = 'GBM'  # Glioblastoma Multiforme

# From metadata, get the rows where "cancer_type" is equal to the specified cancer type
# Then grab the index of this subset (these are the sample IDs)
cancer_samples = metadata_df[metadata_df['cancer_type'] == cancer_type].index
print(cancer_samples)
# Subset the main data to include only these samples
# When you want a subset of columns, you can pass a list of column names to the data frame in []
GBM_data = data[cancer_samples]

# %%
# Subset by index (genes)
####################################################
desired_gene_list = gbm_genes = gene_list
gene_list = [gene for gene in desired_gene_list if gene in GBM_data.index]
for gene in desired_gene_list:
    if gene not in gene_list:
        print(f"Warning: {gene} not found in the dataset.")

# .loc[] is the method to subset by index labels
# .iloc[] will subset by index position (integer location) instead
GBM_gene_data = GBM_data.loc[gene_list]
print(GBM_gene_data.head())

# %%
# Basic statistics on the subsetted data
####################################################
print(GBM_gene_data.describe())
print(GBM_gene_data.var(axis=1))  # Variance of each gene across samples
# Mean expression of each gene across samples
print(GBM_gene_data.mean(axis=1))
# Median expression of each gene across samples
print(GBM_gene_data.median(axis=1))

# %%
# Explore categorical variables in metadata
####################################################
# groupby allows you to group on a specific column in the dataset,
# and then print out summary stats or counts for other columns within those groups
print(metadata_df.groupby('cancer_type')["gender"].value_counts())

# Explore average age at diagnosis by cancer type
metadata_df['age_at_diagnosis'] = pd.to_numeric(
    metadata_df['age_at_diagnosis'], errors='coerce')
print(metadata_df.groupby(
    'cancer_type')["age_at_diagnosis"].mean())
# %%
# Merging datasets
####################################################
# Merge the subsetted expression data with metadata for GBM samples,
# so rows are samples and columns include gene expression for EGFR and MYC and metadata
GBM_metadata = metadata_df.loc[cancer_samples]
GBM_merged = GBM_gene_data.T.merge(GBM_metadata, left_index=True, right_index=True)
print(GBM_merged.head())

# %%
# Plotting
####################################################
# Boxplot of EGFR expression in GBM samples using SEABORN
# Works really well with pandas dataframes, because most methods allow you to pass in a dataframe directly
sns.boxplot(data=GBM_merged, x="gender", y='EGFR')
plt.title("EGFR Expression by Gender in GBM Samples")
plt.show()

# Boxplot of MYC and EGFR expression in GBM samples using PANDAS directly
GBM_merged[['MYC', 'EGFR']].plot.box()
plt.title("MYC and EGFR Expression in GBM Samples")
plt.show()

# %%

X = GBM_gene_data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = scaler.fit_transform(X)
X_pca = PCA(n_components=4).fit_transform(X)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0],
                y=X_pca[:, 1],
                s=100)
plt.title("PCA of Genes (GBM)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()
