
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, jsonify, request

from src.dashboard_charts import create_charts

app = Flask(__name__)

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

DATASETS = {
    "DS1": "DS1_material_properties_5500.csv",
    "DS2": "DS2_commodity_prices_10yr.csv",
    "DS3": "DS3_infrastructure_bridges_5000.csv",
    "DS4": "DS4_crossdomain_features_daily.csv",
    "DS5": "DS5_element_prices_monthly.csv",
    "DS6": "DS6_historical_failures_2000.csv",
}

MODELS = {
    "Material Stability": "material_stability_model.joblib",
    "Commodity Price": "commodity_model.joblib",
    "Bridge Deficiency": "bridge_deficiency_model.joblib",
}


def get_dataset_status():
    results = []

    for dataset_id, filename in DATASETS.items():
        path = RAW_DIR / filename

        item = {
            "id": dataset_id,
            "filename": filename,
            "exists": path.exists(),
            "rows": 0,
            "columns": 0,
            "size_kb": 0,
        }

        if path.exists():
            try:
                df = pd.read_csv(path)

                item["rows"] = int(len(df))
                item["columns"] = int(len(df.columns))
                item["size_kb"] = round(
                    path.stat().st_size / 1024, 2
                )
            except Exception:
                app.logger.exception(
                    "Could not read dataset: %s", filename
                )
                item["error"] = "Unable to read this CSV file."

        results.append(item)

    return results


def get_model_status():
    results = []

    for name, filename in MODELS.items():
        path = MODEL_DIR / filename

        results.append({
            "name": name,
            "filename": filename,
            "exists": path.exists(),
            "size_kb": (
                round(path.stat().st_size / 1024, 2)
                if path.exists()
                else 0
            ),
        })

    return results


def get_report_files():
    if not REPORT_DIR.exists():
        return []

    return sorted(
        file.name
        for file in REPORT_DIR.iterdir()
        if file.is_file()
    )


def get_dashboard_statistics():
    datasets = get_dataset_status()
    models = get_model_status()

    return {
        "datasets": datasets,
        "models": models,
        "total_datasets": len(datasets),
        "available_datasets": sum(
            1 for item in datasets if item["exists"]
        ),
        "available_models": sum(
            1 for item in models if item["exists"]
        ),
        "total_rows": sum(
            item["rows"] for item in datasets if item["exists"]
        ),
        "total_columns": sum(
            item["columns"] for item in datasets if item["exists"]
        ),
        "total_reports": len(get_report_files()),
    }


def load_saved_model(filename):
    path = MODEL_DIR / filename

    if not path.exists():
        return None

    return joblib.load(path)


def inspect_saved_model(saved):
    if isinstance(saved, dict):
        model = saved.get("model")

        features = saved.get("feature_columns")
        if features is None:
            features = saved.get("features")

        target = saved.get("target_column")

        if model is None and hasattr(saved, "predict"):
            model = saved

        return model, features, target

    model = saved
    features = getattr(saved, "feature_names_in_", None)

    return model, features, None


def json_safe(value):
    if hasattr(value, "item"):
        return value.item()

    return value


def predict_from_json(filename):
    if not request.is_json:
        return jsonify({
            "status": "error",
            "message": "Send the request body as JSON.",
        }), 400

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict) or not payload:
        return jsonify({
            "status": "error",
            "message": "Provide a non-empty JSON object of features.",
        }), 400

    try:
        saved = load_saved_model(filename)

    except Exception:
        app.logger.exception("Could not load model: %s", filename)

        return jsonify({
            "status": "error",
            "message": "The saved model could not be loaded.",
        }), 500

    if saved is None:
        return jsonify({
            "status": "error",
            "message": f"Model file not found: {filename}",
        }), 404

    saved_model, features, target = inspect_saved_model(saved)

    if saved_model is None or not hasattr(saved_model, "predict"):
        return jsonify({
            "status": "error",
            "message": "The saved file does not contain a usable model.",
        }), 500

    if features is None:
        return jsonify({
            "status": "needs_model_inputs",
            "message": (
                "The saved model does not provide its feature list. "
                "Inspect the model training script."
            ),
        }), 422

    feature_columns = list(features)

    missing = [
        column for column in feature_columns
        if column not in payload
    ]

    extra = [
        column for column in payload
        if column not in feature_columns
    ]

    if missing:
        return jsonify({
            "status": "invalid_input",
            "message": "Required feature values are missing.",
            "missing_features": missing,
            "expected_features": feature_columns,
        }), 400

    if extra:
        return jsonify({
            "status": "invalid_input",
            "message": "Unexpected feature names were supplied.",
            "unexpected_features": extra,
            "expected_features": feature_columns,
        }), 400

    try:
        row = pd.DataFrame(
            [[payload[column] for column in feature_columns]],
            columns=feature_columns,
        )

        if row.isna().any().any():
            return jsonify({
                "status": "invalid_input",
                "message": "Feature values cannot be empty or null.",
            }), 400

        prediction = saved_model.predict(row)[0]

        result = {
            "status": "success",
            "prediction": json_safe(prediction),
        }

        if target:
            result["target"] = target

        if hasattr(saved_model, "predict_proba"):
            probabilities = saved_model.predict_proba(row)[0]

            result["probabilities"] = [
                float(value) for value in probabilities
            ]

            if hasattr(saved_model, "classes_"):
                result["classes"] = [
                    json_safe(value)
                    for value in saved_model.classes_
                ]

        return jsonify(result)

    except (TypeError, ValueError, KeyError) as exc:
        return jsonify({
            "status": "invalid_input",
            "message": str(exc),
        }), 400

    except Exception:
        app.logger.exception("Prediction failed for %s", filename)

        return jsonify({
            "status": "error",
            "message": (
                "Prediction failed. Check the input feature types "
                "and model preprocessing."
            ),
        }), 500


@app.route("/")
def index():
    return render_template(
        "index.html",
        **get_dashboard_statistics()
    )


@app.route("/datasets")
def datasets_page():
    return render_template(
        "datasets.html",
        datasets=get_dataset_status()
    )


@app.route("/models")
def models_page():
    return render_template(
        "models.html",
        models=get_model_status()
    )


@app.route("/predictions", methods=["GET", "POST"])
def predictions_page():
    return render_template(
        "predictions.html",
        models=get_model_status()
    )


@app.route("/reports")
def reports_page():
    return render_template(
        "reports.html",
        reports=get_report_files()
    )


@app.route("/analytics")
def analytics():
    try:
        return render_template(
            "analytics.html",
            charts=create_charts()
        )

    except Exception:
        app.logger.exception("Analytics page failed")

        return render_template(
            "analytics.html",
            charts=[],
            error="Unable to create charts. Check your CSV files."
        ), 500


@app.route("/api/health")
def health():
    stats = get_dashboard_statistics()

    return jsonify({
        "status": "healthy",
        "application": "MatRisk AI",
        "datasets_available": stats["available_datasets"],
        "total_datasets": stats["total_datasets"],
        "models_available": stats["available_models"],
    })


@app.route("/api/datasets")
def api_datasets():
    return jsonify(get_dataset_status())


@app.route("/api/models")
def api_models():
    return jsonify(get_model_status())


@app.route("/api/reports")
def api_reports():
    return jsonify(get_report_files())


@app.route("/api/analytics")
def api_analytics():
    try:
        charts = create_charts()

        return jsonify({
            "status": "success",
            "chart_count": len(charts),
            "charts": [
                {"title": chart["title"]}
                for chart in charts
            ],
        })

    except Exception:
        app.logger.exception("Analytics API failed")

        return jsonify({
            "status": "error",
            "message": "Unable to generate analytics.",
        }), 500


@app.route("/api/predict/material", methods=["POST"])
def predict_material():
    return predict_from_json(
        "material_stability_model.joblib"
    )


@app.route("/api/predict/commodity", methods=["POST"])
def predict_commodity():
    return predict_from_json(
        "commodity_model.joblib"
    )


@app.route("/api/predict/bridge", methods=["POST"])
def predict_bridge():
    return predict_from_json(
        "bridge_deficiency_model.joblib"
    )


@app.errorhandler(404)
def page_not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({
            "status": "error",
            "message": "API endpoint not found.",
        }), 404

    return (
        "Page not found. Return to the MatRisk AI dashboard.",
        404,
    )


@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "status": "error",
        "message": "An internal server error occurred.",
    }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
