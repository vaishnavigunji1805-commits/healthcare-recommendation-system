import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_healthcare_lstm(time_steps=5, features=4):
    """
    Builds an industry-standard sequential LSTM Deep Learning model 
    for tracking temporal trends in wearable medical sensor logs.
    """
    model = Sequential([
        # Layer 1: Processes the 5 time-steps sequence window
        LSTM(64, input_shape=(time_steps, features), return_sequences=True),
        Dropout(0.2), # Prevents the neural network from overfitting
        
        # Layer 2: Aggregates sequence features into a single state vector
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        
        # Layer 3: Fully connected decision dense layer
        Dense(16, activation='relu'),
        
        # Output Layer: Sigmoid maps prediction to a probability percentage (0.0 to 1.0)
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    # Let's verify the model compiles correctly
    my_model = build_healthcare_lstm()
    my_model.summary()
