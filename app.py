from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Paths to the model and scaler
MODEL_PATH = r"C:\MLOps Projects\Smart-Manufacturing-Machines-Efficiency-Prediction\artifacts\model\xgboost_model.pkl"
SCALER_PATH = r"C:\MLOps Projects\Smart-Manufacturing-Machines-Efficiency-Prediction\artifacts\processed\scaler.pkl"

# Load the model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Raw features the user will input
RAW_FEATURES = [
    'Vibration_Hz', 'Packet_Loss_Perc', 'Quality_Control_Defect_Rate_Perc',
    'Production_Speed_units_per_hr', 'Error_Rate_Perc', 'Operation_Mode',
    'Predictive_Maintenance_Score'
]

# Features expected by the model (10 features)
SELECTED_FEATURES = [
    'Vibration_Hz', 'Packet_Loss_Perc', 'Quality_Control_Defect_Rate_Perc',
    'Production_Speed_units_per_hr', 'Error_Rate_Perc', 'Mode_Active',
    'Mode_Idle', 'PM_Bin_Low', 'PM_Bin_Medium', 'PM_Bin_High'
]

# Labels mapping
LABELS = {0: "High", 1: "Low", 2: "Medium"}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # Collect input data from the form
            input_data = {}
            for feature in RAW_FEATURES:
                value = request.form.get(feature)
                if feature == 'Operation_Mode':
                    input_data[feature] = value
                else:
                    input_data[feature] = float(value)

            # Create a DataFrame for preprocessing
            df = pd.DataFrame([input_data])

            # Preprocess the input data
            # 1. One-hot encode Operation_Mode
            df['Mode_Active'] = (df['Operation_Mode'] == 'Active').astype(int)
            df['Mode_Idle'] = (df['Operation_Mode'] == 'Idle').astype(int)
            df = df.drop(columns=['Operation_Mode'])

            # 2. Bin Predictive_Maintenance_Score
            bins = [0, 33.33, 66.66, 100]  # Same bins as in training
            labels = ['Low', 'Medium', 'High']
            df['Predictive_Maintenance_Binned'] = pd.cut(
                df['Predictive_Maintenance_Score'], bins=bins, labels=labels, include_lowest=True
            )
            df = pd.get_dummies(df, columns=['Predictive_Maintenance_Binned'], prefix='PM_Bin')
            df = df.drop(columns=['Predictive_Maintenance_Score'])

            # 3. Ensure all selected features are present
            for feature in SELECTED_FEATURES:
                if feature not in df.columns:
                    df[feature] = 0

            # 4. Reorder columns to match SELECTED_FEATURES
            df = df[SELECTED_FEATURES]

            # 5. Scale the selected features using the loaded scaler
            input_array = df.values
            scaled_array = scaler.transform(input_array)

            # 6. Make prediction
            pred = model.predict(scaled_array)[0]
            prediction = LABELS.get(pred, "Unknown")

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction, features=RAW_FEATURES)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)