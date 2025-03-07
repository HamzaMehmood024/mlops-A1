from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Home route (renders HTML form)
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input from form
        features = [float(x) for x in request.form.values()]
        features_array = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features_array)
        result = f"Predicted Class: {prediction[0]}"
        
        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
