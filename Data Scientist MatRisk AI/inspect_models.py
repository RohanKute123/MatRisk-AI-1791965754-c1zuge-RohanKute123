
from pathlib import Path
import joblib

model_dir = Path("models")

files = [
    "material_stability_model.joblib",
    "commodity_model.joblib",
    "bridge_deficiency_model.joblib",
]

for filename in files:
    path = model_dir / filename

    print("\n" + "=" * 55)
    print(filename)

    if not path.exists():
        print("Model file not found")
        continue

    try:
        saved = joblib.load(path)

        if isinstance(saved, dict):
            print("Saved keys:", list(saved.keys()))

            model = saved.get("model")

            if "feature_columns" in saved:
                print("Feature columns:")
                print(saved["feature_columns"])

            if "target_column" in saved:
                print("Target column:", saved["target_column"])

            if model is not None:
                print("Model type:", type(model).__name__)

                if hasattr(model, "classes_"):
                    print("Classes:", model.classes_)

                if hasattr(model, "n_features_in_"):
                    print("Expected feature count:", model.n_features_in_)
        else:
            print("Saved object type:", type(saved).__name__)

            if hasattr(saved, "feature_names_in_"):
                print("Feature columns:", saved.feature_names_in_)

            if hasattr(saved, "n_features_in_"):
                print("Expected feature count:", saved.n_features_in_)

    except Exception as exc:
        print("Error:", exc)
