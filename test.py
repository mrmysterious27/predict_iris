import argparse
import joblib
import numpy as np

# Path to the trained model
MODEL_PATH = "models/iris_best_model.joblib"
TARGET_NAMES = ["setosa", "versicolor", "virginica"]

def main():
    # Command-line arguments
    p = argparse.ArgumentParser(description="Predict Iris species")
    p.add_argument("sepal_length", type=float)
    p.add_argument("sepal_width", type=float)
    p.add_argument("petal_length", type=float)
    p.add_argument("petal_width", type=float)
    args = p.parse_args()

    # Load the trained model
    model = joblib.load(MODEL_PATH)

    # Prepare input for prediction
    x = np.array([[args.sepal_length, args.sepal_width, args.petal_length, args.petal_width]])

    # Predict class
    pred = model.predict(x)[0]

    # Try probability prediction (if supported by model)
    try:
        proba = model.predict_proba(x)[0]
    except Exception:
        proba = None

    # Output results
    label = TARGET_NAMES[int(pred)]
    print(f"Predicted: {label} (class {int(pred)})")

    if proba is not None:
        print("Probabilities:")
        for name, p in zip(TARGET_NAMES, proba):
            print(f"  {name}: {p:.3f}")

if __name__ == "__main__":
    main()