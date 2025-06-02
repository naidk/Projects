import numpy as np

def make_prediction(input_data: list, model, scaler):
    """
    Make a prediction using the trained model and scaler.

    Args:
        input_data (list): List of 8 input features.
        model: Trained classification model (with predict_proba).
        scaler: Fitted scaler (e.g., StandardScaler).

    Returns:
        tuple: (predicted_label, probability_dict)
    """
    try:
        # Convert to NumPy array and reshape for prediction
        input_array = np.array(input_data).reshape(1, -1)

        # Scale the input
        input_scaled = scaler.transform(input_array)

        # Predict probabilities
        probabilities = model.predict_proba(input_scaled)[0]  # [non-diabetic, diabetic]

        # Get the predicted class
        prediction = np.argmax(probabilities)

        # Class labels (adjust if your model was trained with different label order)
        class_labels = ['Non-Diabetic', 'Diabetic']

        return class_labels[prediction], {
            "Non-Diabetic": round(probabilities[0], 4),
            "Diabetic": round(probabilities[1], 4)
        }

    except Exception as e:
        return "Error", {"message": str(e)}
