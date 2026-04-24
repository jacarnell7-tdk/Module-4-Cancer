# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# 1. Load gene expression data
# -----------------------------
data = pd.read_csv(
    'c:/Users/vkb5cq/Desktop/Spring 2026/BME 2315/Module-4-Cancer/data/TRAINING_SET_GSE62944_subsample_log2TPM.csv',
    index_col=0
)

# First column = gene names → set as index, then transpose
data = data.T

print("Expression data shape:", data.shape)

# -----------------------------
# 2. Load angiogenesis gene list
# -----------------------------
hallmark_df = pd.read_csv(
    r"c:/Users/vkb5cq/Desktop/Spring 2026/BME 2315/Module-4-Cancer/Menyhart_JPA_CancerHallmarks_core.txt",
    sep='\t'
)

angiogenesis_row = hallmark_df[hallmark_df.iloc[:, 0] == "SUSTAINED ANGIOGENESIS"]
genes = angiogenesis_row.iloc[0, 1:].dropna()
desired_gene_list = genes.tolist()

# -----------------------------
# 3. Load metadata
# -----------------------------
metadata_df = pd.read_csv(
    'c:/Users/vkb5cq/Desktop/Spring 2026/BME 2315/Module-4-Cancer/data/TRAINING_SET_GSE62944_metadata.csv',
    index_col=0
)

print("Metadata shape:", metadata_df.shape)

print("Example expression genes:", list(data.columns[:20]))
print("Example hallmark genes:", desired_gene_list[:20])

# -----------------------------
# 4. Align patients between datasets
# -----------------------------
common_patients = data.index.intersection(metadata_df.index)

data = data.loc[common_patients]
metadata_df = metadata_df.loc[common_patients]

print("Aligned data shape:", data.shape)
print("Aligned metadata shape:", metadata_df.shape)

# -----------------------------
# 5. Define features (X) and target (y)
# -----------------------------
valid_genes = [g for g in desired_gene_list if g in data.columns]
print("Using genes:", valid_genes)

X = data[valid_genes]

# Encode tumor status
y = metadata_df["tumor_status"].map({"WITH TUMOR": 1, "WITHOUT TUMOR": 0})

# Drop missing labels if any
valid_idx = y.dropna().index
X = X.loc[valid_idx]
y = y.loc[valid_idx]

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)

print(y.value_counts())

# -----------------------------
# 6. Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 7. Scale features
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 8. Train logistic regression
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# -----------------------------
# 9. Evaluate model
# -----------------------------
y_pred = model.predict(X_test_scaled)

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 10. Gene importance
# -----------------------------
coefficients = pd.DataFrame({
    "Gene": valid_genes,
    "Coefficient": model.coef_[0]
})

print("\nGene Importance:")
print(coefficients.sort_values(by="Coefficient", ascending=False))