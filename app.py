from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)


# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "artifacts", "model", "xgboost_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "artifacts", "processed", "scaler.pkl")

# Load the model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Raw features expected from the form
RAW_FEATURES = [
    'Vibration_Hz',
    'Packet_Loss_Perc',
    'Quality_Control_Defect_Rate_Perc',
    'Production_Speed_units_per_hr',
    'Error_Rate_Perc',
    'Operation_Mode',
    'Predictive_Maintenance_Score'
]

# Processed features expected by the model
SELECTED_FEATURES = [
    'Vibration_Hz', 'Packet_Loss_Perc', 'Quality_Control_Defect_Rate_Perc',
    'Production_Speed_units_per_hr', 'Error_Rate_Perc', 'Mode_Active',
    'Mode_Idle', 'PM_Bin_Low', 'PM_Bin_Medium', 'PM_Bin_High'
]

# Map of prediction class labels
LABELS = {0: "High", 1: "Low", 2: "Medium"}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # Collect form input
            input_data = {}
            for feature in RAW_FEATURES:
                value = request.form.get(feature)
                if feature == 'Operation_Mode':
                    input_data[feature] = value
                else:
                    input_data[feature] = float(value)

            # Create DataFrame
            df = pd.DataFrame([input_data])

            # One-hot encode Operation_Mode
            df['Mode_Active'] = (df['Operation_Mode'] == 'Active').astype(int)
            df['Mode_Idle'] = (df['Operation_Mode'] == 'Idle').astype(int)
            df.drop(columns=['Operation_Mode'], inplace=True)

            # Bin Predictive_Maintenance_Score
            bins = [0, 33.33, 66.66, 100]
            labels = ['Low', 'Medium', 'High']
            df['Predictive_Maintenance_Binned'] = pd.cut(
                df['Predictive_Maintenance_Score'], bins=bins, labels=labels, include_lowest=True
            )
            df = pd.get_dummies(df, columns=['Predictive_Maintenance_Binned'], prefix='PM_Bin')
            df.drop(columns=['Predictive_Maintenance_Score'], inplace=True)

            # Ensure all expected features exist
            for feature in SELECTED_FEATURES:
                if feature not in df.columns:
                    df[feature] = 0

            # Reorder to match model
            df = df[SELECTED_FEATURES]

            # Scale and predict
            scaled = scaler.transform(df)
            pred = model.predict(scaled)[0]
            prediction = LABELS.get(pred, "Unknown")

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", features=RAW_FEATURES, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
