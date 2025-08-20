import numpy as np
import pandas as pd
import joblib
import os
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1) Load dataset
iris = datasets.load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names

# 2) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 3) Define candidate models + grids
models = {
    "LogisticRegression": (
        LogisticRegression(max_iter=1000),
        {"clf__C": [0.1, 1.0, 10]}
    ),
    "RandomForest": (
        RandomForestClassifier(),
        {"clf__n_estimators": [50, 100],
         "clf__max_depth": [None, 5, 10]}
    )
}

best_model, best_score, best_name = None, 0, ""

# 4) Cross-validation to pick best model
for name, (model, param_grid) in models.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", model)])
    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(X_train, y_train)
    print(f"{name}: CV best score={grid.best_score_:.4f}, best params={grid.best_params_}")

    if grid.best_score_ > best_score:
        best_model = grid.best_estimator_
        best_score = grid.best_score_
        best_name = name

print("\n===== Best Model =====")
print(best_name)
print(best_model)

# 5) Evaluate on test data
y_pred = best_model.predict(X_test)
print("\n===== Test Accuracy =====")
print(accuracy_score(y_test, y_pred))

print("\n===== Classification Report =====")
print(classification_report(y_test, y_pred, target_names=target_names))

print("\n===== Confusion Matrix =====")
print(confusion_matrix(y_test, y_pred))

# 6) Save model locally
os.makedirs("models", exist_ok=True)
model_path = "models/iris_best_model.joblib"
joblib.dump(best_model, model_path)
print(f"\n✅ Model saved at: {model_path}")

# 7) Simple prediction function
def predict_iris(features):
    """
    Predict Iris species from a feature vector [sepal_len, sepal_wid, petal_len, petal_wid].
    Example: predict_iris([5.1, 3.5, 1.4, 0.2])
    """
    model = joblib.load(model_path)
    prediction = model.predict([features])[0]
    probas = model.predict_proba([features])[0]
    return {
        "species": target_names[prediction],
        "probabilities": dict(zip(target_names, probas.round(3)))
    }

# Example run
example = [5.1, 3.5, 1.4, 0.2]
print("\nExample prediction:", predict_iris(example))

