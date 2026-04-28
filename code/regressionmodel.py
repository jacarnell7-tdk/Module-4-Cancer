# -----------------------------
# Import libraries
# -----------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# Load data (UPDATE PATHS)
# -----------------------------
data = pd.read_csv(r"C:\Users\vkb5cq\Desktop\Spring 2026\BME 2315\Module-4-Cancer\data\TRAINING_SET_GSE62944_subsample_log2TPM.csv",
    index_col=0
)

metadata_df = pd.read_csv(r"C:\Users\vkb5cq\Desktop\Spring 2026\BME 2315\Module-4-Cancer\data\TRAINING_SET_GSE62944_metadata.csv",
    index_col=0
)

hallmark_genes = pd.read_csv(r"C:\Users\vkb5cq\Desktop\Spring 2026\BME 2315\Module-4-Cancer\Menyhart_JPA_CancerHallmarks_core.txt",
    sep="\t"
)

# -----------------------------
# Format expression data
# -----------------------------
data = data.T  # patients as rows

print("Initial data shape:", data.shape)
print("Metadata shape:", metadata_df.shape)

# -----------------------------
# Align samples
# -----------------------------
common_samples = data.index.intersection(metadata_df.index)
data = data.loc[common_samples]
metadata_df = metadata_df.loc[common_samples]

print("Aligned data shape:", data.shape)

# -----------------------------
# OPTIONAL: Filter for GBM
# -----------------------------
cancer_type = "GBM"

gbm_idx = metadata_df[metadata_df["cancer_type"] == cancer_type].index

data = data.loc[gbm_idx]
metadata_df = metadata_df.loc[gbm_idx]

print("GBM-only data shape:", data.shape)

# -----------------------------
# Extract angiogenesis genes
# -----------------------------
angiogenesis_row = hallmark_genes[
    hallmark_genes.iloc[:, 0] == "SUSTAINED ANGIOGENESIS"
]

genes = angiogenesis_row.iloc[0, 1:].dropna().tolist()

valid_genes = [g for g in genes if g in data.columns]

print("Using genes:", valid_genes)

# -----------------------------
# Define target
# -----------------------------
y = metadata_df["gender"].map({"MALE": 1, "FEMALE": 0})

# Drop missing labels
valid_idx = y.dropna().index
data = data.loc[valid_idx]
y = y.loc[valid_idx]

X = data[valid_genes]

print("Final dataset shape:", X.shape)
print("Class balance:\n", y.value_counts())

# -----------------------------
# Create 80/10/10 split
# -----------------------------
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

# Build datasets
X_train = X.loc[train_idx]
X_val = X.loc[val_idx]

y_train = y.loc[train_idx]
y_val = y.loc[val_idx]

print("\nSplit sizes:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)

print("\nValidation class balance:")
print(y_val.value_counts())

# -----------------------------
# Scale data
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# -----------------------------
# Train model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# -----------------------------
# Evaluate on validation set
# -----------------------------
y_pred = model.predict(X_val_scaled)

print("\nACCURACY:", accuracy_score(y_val, y_pred))

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_val, y_pred))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_val, y_pred))

# -----------------------------
# Gene importance
# -----------------------------
gene_importance = pd.DataFrame({
    "Gene": valid_genes,
    "Coefficient": model.coef_[0]
})

print("\nTop positive (predict MALE):")
print(gene_importance.sort_values(by="Coefficient", ascending=False).head(10))

print("\nTop negative (predict FEMALE):")
print(gene_importance.sort_values(by="Coefficient").head(10))