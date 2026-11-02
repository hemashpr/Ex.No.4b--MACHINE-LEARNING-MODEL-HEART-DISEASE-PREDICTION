import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# 1. Generate a reproducible synthetic heart-disease dataset.
# This is demonstration data, not real patient records or a clinical model.
feature_names = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
    "fasting_blood_sugar", "resting_ecg", "max_heart_rate",
    "exercise_angina", "oldpeak", "slope", "major_vessels", "thal"
]

X_raw, y = make_classification(
    n_samples=1025,
    n_features=13,
    n_informative=9,
    n_redundant=2,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=2,
    weights=[0.55, 0.45],
    class_sep=0.9,
    flip_y=0.025,
    random_state=42
)

# Map generated features to plausible demonstration ranges.
feature_ranges = [
    (29, 78), (0, 1), (0, 3), (90, 190), (120, 360),
    (0, 1), (0, 2), (85, 205), (0, 1), (0, 6),
    (0, 2), (0, 3), (0, 2)
]

X = np.zeros_like(X_raw)
for column, (lower, upper) in enumerate(feature_ranges):
    values = X_raw[:, column]
    normalized = (values - values.min()) / (values.max() - values.min())
    X[:, column] = np.round(lower + normalized * (upper - lower))

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

print("FIRST FIVE RECORDS")
print(df.head().to_string(index=False))
print("\nDATASET SHAPE:", df.shape)
print("\nMISSING VALUES")
print(df.isnull().sum().to_string())

# 2. Separate features and target; split with class proportions preserved.
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\nTRAINING SET:", X_train.shape)
print("TESTING SET :", X_test.shape)

# 3. Define six classification models.
# Scaling is applied inside pipelines for models sensitive to feature scale.
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42))
    ]),
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=7))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=4, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    ),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", C=1.5, gamma="scale"))
    ]),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=120, learning_rate=0.05, max_depth=2,
        random_state=42
    )
}

# 4. Train models and calculate evaluation metrics.
predictions = {}
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-score": f1_score(y_test, y_pred, zero_division=0)
    })

results_df = pd.DataFrame(results).sort_values(
    by="F1-score", ascending=False
).reset_index(drop=True)

print("\nMODEL PERFORMANCE")
print(results_df.to_string(
    index=False,
    formatters={
        "Accuracy": "{:.4f}".format,
        "Precision": "{:.4f}".format,
        "Recall": "{:.4f}".format,
        "F1-score": "{:.4f}".format
    }
))

best_model = results_df.iloc[0]["Model"]
print("\nBEST MODEL BY F1-SCORE:", best_model)

# 5. Generate model-comparison graph.
plot_data = results_df.sort_values("F1-score", ascending=True)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(plot_data["Model"], plot_data["F1-score"])
ax.set_xlim(0, 1)
ax.set_xlabel("F1-score")
ax.set_title("Heart Disease Classification - Model Comparison")
ax.grid(axis="x", alpha=0.25)
fig.tight_layout()
fig.savefig("model_comparison.png", dpi=180, bbox_inches="tight")
fig.savefig("model_comparison.svg", bbox_inches="tight")
plt.close(fig)

# 6. Generate confusion matrix for the Random Forest classifier.
cm = confusion_matrix(y_test, predictions["Random Forest"])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"], ax=ax
)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - Random Forest")
fig.tight_layout()
fig.savefig("confusion_matrix.png", dpi=180, bbox_inches="tight")
fig.savefig("confusion_matrix.svg", bbox_inches="tight")
plt.close(fig)

print("\nRANDOM FOREST CONFUSION MATRIX")
print(cm)
print("\nOUTPUT IMAGES SAVED:")
print("model_comparison.png")
print("confusion_matrix.png")
print("\nRESULT: Six classification models trained and evaluated successfully.")
