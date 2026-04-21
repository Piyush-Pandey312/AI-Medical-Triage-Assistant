# ── ml_module.py: Reusable ML Prediction Module ─────────────────

import joblib
import pandas as pd

# 🔹 Load saved model, scaler, encoders
model = joblib.load('ml_model.pkl')
scaler = joblib.load('ml_scaler.pkl')
encoders = joblib.load('encoder.pkl')


def predict_risk(Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS,
                 RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope):
    """
    Predict heart disease risk
    Returns: dict {label, score}
    """

    # Create input dataframe
    input_data = pd.DataFrame([{
        'Age': Age,
        'Sex': Sex,
        'ChestPainType': ChestPainType,
        'RestingBP': RestingBP,
        'Cholesterol': Cholesterol,
        'FastingBS': FastingBS,
        'RestingECG': RestingECG,
        'MaxHR': MaxHR,
        'ExerciseAngina': ExerciseAngina,
        'Oldpeak': Oldpeak,
        'ST_Slope': ST_Slope
    }])

    # Encode categorical features
    for col in input_data.columns:
        if col in encoders:
            input_data[col] = encoders[col].transform(input_data[col])

    # Scale features
    input_scaled = scaler.transform(input_data)

    # Predict probability
    score = float(model.predict_proba(input_scaled)[0][1])

    # Assign label
    if score > 0.6:
        label = 'HIGH'
    elif score > 0.35:
        label = 'MODERATE'
    else:
        label = 'LOW'

    return {
        'label': label,
        'score': round(score, 2)
    }


# 🔥 Optional: Quick test (runs if file executed directly)
if __name__ == "__main__":
    result = predict_risk(
        Age=55,
        Sex='M',
        ChestPainType='ATA',
        RestingBP=160,
        Cholesterol=280,
        FastingBS=1,
        RestingECG='Normal',
        MaxHR=95,
        ExerciseAngina='Y',
        Oldpeak=1.5,
        ST_Slope='Flat'
    )

    print("Test Output:", result)