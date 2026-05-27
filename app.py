from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# --- Load Model and Scaler ---
# Load the best model saved from the notebook
with open("crop_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the scaler saved from the notebook
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# --- Fertilizer Recommendation Logic (Rule-Based) ---
# This dictionary maps crops to fertilizer recommendations.
# These are simplified recommendations for demonstration purposes.
fertilizer_recommendations = {
    'rice': 'High Nitrogen (Urea), Phosphorus, and Potassium.',
    'maize': 'High Nitrogen (Urea/Ammonium Nitrate), Phosphorus, and Potassium.',
    'chickpea': 'Low Nitrogen, High Phosphorus (DAP), and Potassium.',
    'kidneybeans': 'Low Nitrogen, High Phosphorus, and Potassium.',
    'pigeonpeas': 'Low Nitrogen, High Phosphorus, and Potassium.',
    'mothbeans': 'Low Nitrogen, High Phosphorus, and Potassium.',
    'mungbean': 'Low Nitrogen, High Phosphorus, and Potassium.',
    'blackgram': 'Low Nitrogen, High Phosphorus, and Potassium.',
    'lentil': 'Low Nitrogen, High Phosphorus, and Potassium.',
    'pomegranate': 'Balanced NPK, with emphasis on Potassium during fruit development.',
    'banana': 'Very High Potassium, High Nitrogen.',
    'mango': 'Balanced NPK, reduce Nitrogen before flowering.',
    'grapes': 'Balanced NPK, with extra Potassium.',
    'watermelon': 'High Nitrogen and Potassium.',
    'muskmelon': 'High Nitrogen and Potassium.',
    'apple': 'Balanced NPK, with Boron.',
    'orange': 'High Nitrogen and Potassium, plus micronutrients like Zinc and Iron.',
    'papaya': 'High Nitrogen and Phosphorus.',
    'coconut': 'High Potassium and Chlorine.',
    'cotton': 'High Nitrogen and Potassium.',
    'jute': 'High Nitrogen.',
    'coffee': 'High Nitrogen, balanced with Phosphorus and Potassium.'
}


@app.route("/", methods=["GET"])
def index():
    """Render the main page."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle the prediction request.
    Processes form data, makes predictions, and returns results as JSON.
    """
    # --- 1. Get and Process Input Data ---
    try:
        # Get data from the form
        form_data = [
            float(request.form["N"]),
            float(request.form["P"]),
            float(request.form["K"]),
            float(request.form["temperature"]),
            float(request.form["humidity"]),
            float(request.form["ph"]),
            float(request.form["rainfall"])
        ]
        
        # Convert to a NumPy array and scale it
        sample = np.array([form_data])
        sample_scaled = scaler.transform(sample)

    except (KeyError, ValueError):
        return jsonify({"error": "Invalid input. Please fill all fields with numeric values."}), 400

    # --- 2. Make Top-3 Predictions ---
    # Use predict_proba to get confidence scores for all classes
    probabilities = model.predict_proba(sample_scaled)[0]
    
    # Get the top 3 predictions
    top3_indices = np.argsort(probabilities)[-3:][::-1]
    top3_crops = model.classes_[top3_indices]
    top3_confidences = probabilities[top3_indices]

    top_recommendations = [
        {"crop": crop, "confidence": round(conf * 100, 2)}
        for crop, conf in zip(top3_crops, top3_confidences)
    ]

    # --- 3. Soil Health Analysis ---
    n, p, k, ph = form_data[0], form_data[1], form_data[2], form_data[5]
    
    soil_health = {
        "N": "Low" if n < 50 else "Optimal" if 50 <= n <= 100 else "High",
        "P": "Low" if p < 30 else "Optimal" if 30 <= p <= 60 else "High",
        "K": "Low" if k < 30 else "Optimal" if 30 <= k <= 60 else "High",
        "pH": "Acidic" if ph < 6.5 else "Optimal" if 6.5 <= ph <= 7.5 else "Alkaline"
    }

    # --- 4. Fertilizer Recommendation ---
    # Get fertilizer recommendation for the top predicted crop
    top_crop_name = top_recommendations[0]['crop'].lower()
    fertilizer_info = fertilizer_recommendations.get(top_crop_name, "No specific recommendation available.")

    # --- 5. Return Results as JSON ---
    return jsonify({
        "top_recommendations": top_recommendations,
        "soil_health": soil_health,
        "fertilizer_recommendation": {
            "crop": top_recommendations[0]['crop'],
            "recommendation": fertilizer_info
        }
    })

if __name__ == "__main__":
    app.run(debug=True)