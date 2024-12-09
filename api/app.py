from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load models and files
logistic_model = load("models/logistic_model.joblib")
random_forest = load("models/random_forest_model.joblib")
naive_bayes_model = load("models/naive_bayes_model.joblib")
kmeans = load("models/kmeans_model.joblib")
scaler = load("models/scaler.joblib")
vectorizer = load("models/naive_bayes_vectorizer.joblib")
metrics = pd.read_csv("models/model_metrics.csv").to_dict(orient="index")

# Load feature importance
try:
    rf_importance = pd.read_csv("models/random_forest_feature_importance.csv")
    rf_importance_data = rf_importance.to_dict(orient="records")
except FileNotFoundError:
    rf_importance_data = None

@app.route('/predict_purchase', methods=['POST'])
def predict_purchase():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        input_df_scaled = scaler.transform(input_df)

        logistic_prediction = logistic_model.predict(input_df_scaled)[0]
        random_forest_prediction = random_forest.predict(input_df)[0]
        naive_bayes_prediction = 1  # Placeholder
        kmeans_cluster = kmeans.predict(input_df_scaled)[0]

        response = {
            "logistic_prediction": int(logistic_prediction),
            "random_forest_prediction": int(random_forest_prediction),
            "naive_bayes_prediction": int(naive_bayes_prediction),
            "kmeans_cluster": int(kmeans_cluster),
            "metrics": metrics,
            "random_forest_importance": rf_importance_data
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_model_scores', methods=['GET'])
def get_model_scores():
    if metrics is not None:
        return jsonify(metrics)
    else:
        return jsonify({"error": "Model metrics file not found."}), 404

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
