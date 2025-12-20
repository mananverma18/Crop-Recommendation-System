from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("crop_model.pkl", "rb") as f:
    model = pickle.load(f)

# OPTIONAL: load scaler if it exists
try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None

# Crop mapping (MATCHES TRAINING)
crop_map = {
    0: "Apple",
    1: "Banana",
    2: "Blackgram",
    3: "Chickpea",
    4: "Coconut",
    5: "Coffee",
    6: "Cotton",
    7: "Grapes",
    8: "Jute",
    9: "Maize",
    10: "Mango",
    11: "Mothbeans",
    12: "Mungbean",
    13: "Muskmelon",
    14: "Orange",
    15: "Papaya",
    16: "Pigeonpeas",
    17: "Pomegranate",
    18: "Rice",
    19: "Watermelon"
}


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_name = None
    prediction_no = None

    if request.method == "POST":
        sample = np.array([[
            float(request.form["N"]),
            float(request.form["P"]),
            float(request.form["K"]),
            float(request.form["temperature"]),
            float(request.form["humidity"]),
            float(request.form["ph"]),
            float(request.form["rainfall"])
        ]])

        if scaler is not None:
            sample = scaler.transform(sample)

        prediction_no = int(model.predict(sample)[0])
        prediction_name = model.classes_[prediction_no]  # 🔥 FIX

    return render_template(
    "index.html",
    prediction_name=prediction_name,
    prediction_no=prediction_no,
    crop_map=crop_map   # ✅ THIS FIXES EVERYTHING
)

if __name__ == "__main__":
    app.run(debug=True)
