import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_healthcare_lstm(time_steps=5, features=4):
    """
    Builds a professional sequence-to-classification LSTM network
    for processing sequential wearable sensor data windows.
    """
    model = Sequential([
        # First LSTM Layer: processes time-series frames and passes sequences forward
        LSTM(64, input_shape=(time_steps, features), return_sequences=True),
        Dropout(0.2), # Prevents overfitting on training datasets
        
        # Second LSTM Layer: collapses the sequence down to global features
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        
        # Dense layers for final classification logic
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid') # Outputs a probability score between 0 and 1
    ])
    
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    # Let's verify the architecture looks sound
    network = build_healthcare_lstm()
    network.summary()
