from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return "Titanic Cancellation Prediction API"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    print("Received JSON data:", data)  # <--- prints to Flask console
    input_data = np.array(data["features"]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)