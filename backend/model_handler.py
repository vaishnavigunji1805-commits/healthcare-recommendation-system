import numpy as np
import tensorflow as tf
from lstm_model import build_healthcare_lstm

class HealthcareModelHandler:
    def __init__(self, model_weights_path="health_lstm_model.h5"):
        # Initialize architecture and try to load pre-trained weights
        self.model = build_healthcare_lstm(time_steps=5, features=4)
        try:
            self.model.load_weights(model_weights_path)
            print("Successfully loaded pre-trained LSTM weights.")
        except Exception:
            print("Weights file not found or mismatched. Operating in un-trained test mode.")

    def run_prediction(self, sequence_data: list):
        """
        Accepts a nested list of shape (5, 4) -> 5 time steps, 4 features each.
        Returns a float probability risk score.
        """
        # Convert incoming list to a NumPy array with batch size 1 -> Shape: (1, 5, 4)
        input_array = np.array([sequence_data], dtype=np.float32)
        
        # Get raw prediction probability
        prediction_prob = self.model.predict(input_array)[0][0]
        
        return float(prediction_prob)
