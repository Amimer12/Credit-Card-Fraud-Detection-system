from flask import Flask, request, render_template, jsonify
import numpy as np
import joblib
import pandas as pd

# Load models and medians
rf_model = joblib.load("models/rf_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
feature_medians = joblib.load("models/feature_medians.pkl")  # Pandas Series

app = Flask(__name__)

def ensemble_predict(X_input, threshold=0.35, w_xgb=0.6, w_rf=0.4):
    proba_rf = rf_model.predict_proba(X_input)[:, 1]
    proba_xgb = xgb_model.predict_proba(X_input)[:, 1]
    avg_proba = (w_xgb * proba_xgb + w_rf * proba_rf) / (w_xgb + w_rf)
    prediction = (avg_proba >= threshold).astype(int)
    return prediction[0], avg_proba[0]

@app.route("/", methods=["GET", "POST"])
def index():
    fraud_prob = None
    pred_label = None

    if request.method == "POST":
        # Start with median values for all features
        input_data = feature_medians.to_dict()

        # Replace with user inputs
        amount = float(request.form.get("Amount", 0))
        time = float(request.form.get("Time", 0))
        input_data["Amount"] = amount
        input_data["Time"] = time

        # Convert to DataFrame row
        X_input = pd.DataFrame([input_data])

        # Align feature order with training
        X_input = X_input[rf_model.feature_names_in_]

        # Make prediction
        pred_label, fraud_prob = ensemble_predict(X_input)

    return render_template("index.html", fraud_prob=fraud_prob, pred_label=pred_label)

# 🔹 Route to generate random V1–V28 values
@app.route("/random_features")
def random_features():
    # Pick only V1–V28 columns
    v_features = feature_medians.filter(like="V")

    # Generate random values (normal distribution around median)
    random_vals = {
        col: np.round(np.random.normal(loc=v_features[col], scale=1), 4)
        for col in v_features.index
    }

    # Sort by feature number (V1 → V28)
    ordered_vals = dict(sorted(random_vals.items(), key=lambda x: int(x[0][1:])))

    return jsonify(ordered_vals)


if __name__ == "__main__":
    app.run(debug=True)
