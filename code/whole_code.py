import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import umap

# -----------------------------
# Load data
# -----------------------------
data = pd.read_csv(
    r'C:\Users\vkb5cq\Desktop\Spring 2026\BME 2315\Module-4-Cancer\data\TRAINING_SET_GSE62944_subsample_log2TPM.csv',
    index_col=0
)

metadata_df = pd.read_csv(
    r'C:\Users\vkb5cq\Desktop\Spring 2026\BME 2315\Module-4-Cancer\data\TRAINING_SET_GSE62944_metadata.csv',
    index_col=0
)

print(data.head())
print(data.shape)
print(data.info())

print(metadata_df.head())
print(metadata_df.info())

# -----------------------------
# Load angiogenesis gene list
# -----------------------------
df = pd.read_csv(
    r"C:\Users\vkb5cq\Desktop\Spring 2026\BME 2315\Module-4-Cancer\Menyhart_JPA_CancerHallmarks_core.txt",
    sep='\t'
)

angiogenesis_row = df[df.iloc[:, 0] == "SUSTAINED ANGIOGENESIS"]
genes = angiogenesis_row.iloc[0, 1:].dropna()

desired_gene_list = genes.tolist()

# -----------------------------
# Filter for GBM samples
# -----------------------------
cancer_type = 'GBM'
cancer_samples = metadata_df[metadata_df['cancer_type'] == cancer_type].index

GBM_data = data[cancer_samples]
GBM_metadata = metadata_df.loc[cancer_samples]

# Keep only genes that exist
gene_list = [gene for gene in desired_gene_list if gene in GBM_data.index]
GBM_gene_data = GBM_data.loc[gene_list]

# -----------------------------
# Clean data
# -----------------------------
GBM_data = GBM_data.apply(pd.to_numeric, errors='coerce')
GBM_data = GBM_data.fillna(GBM_data.mean(axis=1), axis=0)

X = GBM_gene_data.T

# -----------------------------
# Create AGE labels (IMPORTANT)
# -----------------------------
age = pd.to_numeric(
    GBM_metadata.loc[X.index, "age_at_diagnosis"],
    errors="coerce"
)

age_cutoff = 60
age_group = (age > age_cutoff).astype(int)

# Align indices
age_group = age_group.loc[GBM_gene_data.columns]

print("\nAge group counts (0=younger, 1=older):")
print(age_group.value_counts())

# -----------------------------
# Scale gene data
# -----------------------------
X_scaled = StandardScaler().fit_transform(X)

# -----------------------------
# PCA
# -----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=age_group,
    palette=["blue", "red"],
    s=100
)

plt.title("PCA of Angiogenesis Genes in GBM (Colored by Age)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Age Group (0=Young, 1=Old)")

plt.show()

# -----------------------------
# UMAP
# -----------------------------
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=X_umap[:, 0],
    y=X_umap[:, 1],
    hue=age_group,
    palette=["blue", "red"],
    s=100
)

plt.title("UMAP of Angiogenesis Genes in GBM (Colored by Age)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")

plt.show()

# -----------------------------
# KMeans clustering
# -----------------------------
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=clusters,
    style=age_group,
    palette="Set2",
    s=100
)

plt.title("KMeans Clusters vs Age Groups")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans

# import umap

# data = pd.read_csv(r'C:\Users\vkb5cq\Desktop\Spring 2026\BME 2315\Module-4-Cancer\data\TRAINING_SET_GSE62944_subsample_log2TPM.csv',
#     index_col=0
# )

# metadata_df = pd.read_csv(r'C:\Users\vkb5cq\Desktop\Spring 2026\BME 2315\Module-4-Cancer\data\TRAINING_SET_GSE62944_metadata.csv',
#     index_col=0
# )

# print(data.head())
# print(data.shape)
# print(data.info())

# print(metadata_df.head())
# print(metadata_df.info())

# df = pd.read_csv(r"C:\Users\vkb5cq\Desktop\Spring 2026\BME 2315\Module-4-Cancer\Menyhart_JPA_CancerHallmarks_core.txt",
#     sep='\t'
# )

# angiogenesis_row = df[df.iloc[:, 0] == "SUSTAINED ANGIOGENESIS"]
# genes = angiogenesis_row.iloc[0, 1:].dropna()

# desired_gene_list = genes.tolist()

# cancer_type = 'GBM'
# cancer_samples = metadata_df[metadata_df['cancer_type'] == cancer_type].index

# GBM_data = data[cancer_samples]
# GBM_metadata = metadata_df.loc[cancer_samples]

# gene_list = [gene for gene in desired_gene_list if gene in GBM_data.index]
# GBM_gene_data = GBM_data.loc[gene_list]

# GBM_data = GBM_data.apply(pd.to_numeric, errors='coerce')
# GBM_data = GBM_data.fillna(GBM_data.mean(axis=1), axis=0)

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

# GBM_data_scaled = pd.DataFrame(
#     scaler.fit_transform(GBM_data.T),
#     index=GBM_data.columns,
#     columns=GBM_data.index
# ).T

# GBM_gene_data.var(axis=1).sort_values(ascending=False).head(10)

# X = GBM_gene_data
# X_scaled = StandardScaler().fit_transform(X)

# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X_scaled)

# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], s=100)

# plt.title("PCA of Angiogenesis Genes in GBM")
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.show()

# reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
# X_umap = reducer.fit_transform(X_scaled)

# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], s=100)

# plt.title("UMAP of Angiogenesis Genes in GBM")
# plt.xlabel("UMAP Dimension 1")
# plt.ylabel("UMAP Dimension 2")
# plt.show()

# kmeans = KMeans(n_clusters=3, random_state=42)
# clusters = kmeans.fit_predict(X_pca)

# plt.figure(figsize=(8, 6))
# sns.scatterplot(
#     x=X_pca[:, 0],
#     y=X_pca[:, 1],
#     hue=clusters,
#     palette="Set2",
#     s=100
# )

# plt.title("KMeans Clustering of Angiogenesis Genes")
# plt.xlabel("PC1")
# plt.ylabel("PC2")
# plt.show()

# -----------------------------
# Prepare data for ML (transpose)
# -----------------------------
X = GBM_gene_data.T  # patients as rows

# Clean and convert age
age = GBM_metadata.loc[X.index, "age_at_diagnosis"]
age = pd.to_numeric(age, errors='coerce')

# Drop missing ages
valid_idx = age.dropna().index
X = X.loc[valid_idx]
age = age.loc[valid_idx]

# Binary classification: older vs younger
age_cutoff = 60
y = (age > age_cutoff).astype(int)

print("\nML dataset shape:", X.shape)
print("\nClass balance:\n", y.value_counts())

# -----------------------------
# 80/10/10 split
# -----------------------------
from sklearn.model_selection import train_test_split

train_idx, temp_idx = train_test_split(
    X.index,
    test_size=0.2,
    random_state=42,
    stratify=y
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.5,
    random_state=42,
    stratify=y.loc[temp_idx]
)

X_train = X.loc[train_idx]
X_val = X.loc[val_idx]

y_train = y.loc[train_idx]
y_val = y.loc[val_idx]

print("\nTrain size:", X_train.shape)
print("Validation size:", X_val.shape)

# -----------------------------
# 🔥 FEATURE SELECTION (IMPORTANT FIX)
# -----------------------------
# Select top 30 most variable genes from training set ONLY
gene_variance = X_train.var(axis=0)
top_genes = gene_variance.sort_values(ascending=False).head(30).index

print("\nUsing top genes:", list(top_genes))

X_train = X_train[top_genes]
X_val = X_val[top_genes]

# -----------------------------
# Scale features
# -----------------------------
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# -----------------------------
# ✅ FIXED Logistic Regression
# -----------------------------
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    penalty='l2',              # FIX: stable regularization
    C=0.1,                     # FIX: not too aggressive
    solver='liblinear',
    class_weight='balanced',   # FIX: handles imbalance
    max_iter=1000,
    random_state=42
)

model.fit(X_train_scaled, y_train)

# -----------------------------
# Evaluate on validation set
# -----------------------------
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_val_scaled)

print("\nACCURACY:", accuracy_score(y_val, y_pred))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_val, y_pred))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_val, y_pred))

# -----------------------------
# Gene importance (NOW FIXED)
# -----------------------------
coefficients = pd.DataFrame({
    "Gene": top_genes,
    "Coefficient": model.coef_[0]
})

coefficients["Interpretation"] = coefficients["Coefficient"].apply(
    lambda x: "OLDER" if x > 0 else "YOUNGER"
)

print("\nTop genes (OLDER):")
print(coefficients.sort_values(by="Coefficient", ascending=False).head(10))

print("\nTop genes (YOUNGER):")
print(coefficients.sort_values(by="Coefficient").head(10))

# -----------------------------
# Confusion Matrix Plot
# -----------------------------
from sklearn.metrics import ConfusionMatrixDisplay

cm = confusion_matrix(y_val, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Younger (<=60)", "Older (>60)"]
)

disp.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix for GBM Age Prediction")
plt.show()

# -----------------------------
# Error + Overfitting Check
# -----------------------------
accuracy = accuracy_score(y_val, y_pred)
error_rate = 1 - accuracy

print(f"\nModel Accuracy: {accuracy:.4f}")
print(f"Model Error Rate: {error_rate:.4f}")

# Training performance
y_train_pred = model.predict(X_train_scaled)

train_accuracy = accuracy_score(y_train, y_train_pred)

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

gap = train_accuracy - accuracy
print(f"Overfitting Gap: {gap:.4f}")
# # -----------------------------
# # Prepare data for ML (transpose)
# # -----------------------------
# X = GBM_gene_data.T  # patients as rows

# # Convert continuous age into binary classification
# # You can adjust cutoff if needed (60 is common in cancer studies)

# age = GBM_metadata.loc[X.index, "age_at_diagnosis"]

# # Drop missing ages
# valid_idx = age.dropna().index
# X = X.loc[valid_idx]
# age = age.loc[valid_idx]

# # Create binary label: 1 = older, 0 = younger
# age_cutoff = 60
# age = pd.to_numeric(
#     GBM_metadata.loc[X.index, "age_at_diagnosis"],
#     errors="coerce"  # turns bad values into NaN
# )
# y = (age > age_cutoff).astype(int)

# print("\nML dataset shape:", X.shape)
# print("Age summary:\n", age.describe())
# print("\nClass balance (0 = younger, 1 = older):\n", y.value_counts())

# # -----------------------------
# # 80/10/10 split
# # -----------------------------
# from sklearn.model_selection import train_test_split

# train_idx, temp_idx = train_test_split(
#     X.index,
#     test_size=0.2,
#     random_state=42,
#     stratify=y
# )

# val_idx, test_idx = train_test_split(
#     temp_idx,
#     test_size=0.5,
#     random_state=42,
#     stratify=y.loc[temp_idx]
# )

# X_train = X.loc[train_idx]
# X_val = X.loc[val_idx]

# y_train = y.loc[train_idx]
# y_val = y.loc[val_idx]

# print("\nTrain size:", X_train.shape)
# print("Validation size:", X_val.shape)

# # -----------------------------
# # Scale features
# # -----------------------------
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_val_scaled = scaler.transform(X_val)

# # -----------------------------
# # Train Logistic Regression
# # -----------------------------
# from sklearn.linear_model import LogisticRegression

# # -----------------------------
# # Train Regularized Logistic Regression
# # -----------------------------
# from sklearn.linear_model import LogisticRegression

# # L2 Regularization (Ridge)
# model = LogisticRegression(
#     penalty='l2',     # type of regularization
#     C=0.1,            # smaller C = stronger regularization
#     solver='liblinear',
#     max_iter=1000,
#     random_state=42
# )

# model.fit(X_train_scaled, y_train)

# # -----------------------------
# # Evaluate on validation set
# # -----------------------------
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# y_pred = model.predict(X_val_scaled)

# print("\nACCURACY:", accuracy_score(y_val, y_pred))

# print("\nCONFUSION MATRIX:")
# print(confusion_matrix(y_val, y_pred))

# print("\nCLASSIFICATION REPORT:")
# print(classification_report(y_val, y_pred))

# # -----------------------------
# # Gene importance
# # -----------------------------
# coefficients = pd.DataFrame({
#     "Gene": X.columns,
#     "Coefficient": model.coef_[0]
# })

# # Add interpretation column (optional but nice)
# coefficients["Interpretation"] = coefficients["Coefficient"].apply(
#     lambda x: "OLDER" if x > 0 else "YOUNGER"
# )

# print("\nTop genes associated with OLDER patients (>60):")
# print(coefficients.sort_values(by="Coefficient", ascending=False).head(10))

# print("\nTop genes associated with YOUNGER patients (<=60):")
# print(coefficients.sort_values(by="Coefficient").head(10))

# # -----------------------------
# # Visual Confusion Matrix + Error Rate
# # -----------------------------
# from sklearn.metrics import ConfusionMatrixDisplay

# # Create confusion matrix
# cm = confusion_matrix(y_val, y_pred)

# # Plot confusion matrix
# plt.figure(figsize=(6, 5))

# disp = ConfusionMatrixDisplay(
#     confusion_matrix=cm,
#     display_labels=["Younger (<=60)", "Older (>60)"]
# )

# disp.plot(cmap="Blues", values_format='d')

# plt.title("Confusion Matrix for GBM Age Prediction")
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")

# plt.show()

# # -----------------------------
# # Calculate model error
# # -----------------------------
# accuracy = accuracy_score(y_val, y_pred)
# error_rate = 1 - accuracy

# print(f"\nModel Accuracy: {accuracy:.4f}")
# print(f"Model Error Rate: {error_rate:.4f}")

# # Optional: Misclassified samples
# misclassified = np.sum(y_val != y_pred)

# print(f"Misclassified Patients: {misclassified}")
# print(f"Total Validation Patients: {len(y_val)}")

# # Training performance
# y_train_pred = model.predict(X_train_scaled)

# train_accuracy = accuracy_score(y_train, y_train_pred)

# print(f"Training Accuracy: {train_accuracy:.4f}")
# print(f"Validation Accuracy: {accuracy:.4f}")

# # Compare training vs validation accuracy
# y_train_pred = model.predict(X_train_scaled)
# y_val_pred = model.predict(X_val_scaled)

# train_accuracy = accuracy_score(y_train, y_train_pred)
# val_accuracy = accuracy_score(y_val, y_val_pred)

# print(f"\nTraining Accuracy: {train_accuracy:.4f}")
# print(f"Validation Accuracy: {val_accuracy:.4f}")

# # Overfitting gap
# gap = train_accuracy - val_accuracy
# print(f"Overfitting Gap: {gap:.4f}")