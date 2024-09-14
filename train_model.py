import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Simulated data generation
np.random.seed(0)
n_samples = 1000
time_steps = 10
data = {
    'time': np.arange(n_samples),
    'arrival_delay': np.sin(np.linspace(0, 20, n_samples)) + np.random.normal(0, 0.5, n_samples)
}
df = pd.DataFrame(data)

# Preprocessing
def preprocess_data(df, time_steps):
    X, y = [], []
    for i in range(len(df) - time_steps):
        X.append(df['arrival_delay'].values[i:i+time_steps])
        y.append(df['arrival_delay'].values[i+time_steps])
    return np.array(X), np.array(y)

X, y = preprocess_data(df, time_steps)
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)
y = scaler.transform(y.reshape(-1, 1)).ravel()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Model definition
model = Sequential([
    LSTM(50, input_shape=(time_steps, 1), return_sequences=True),
    LSTM(50),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Reshape data for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('lstm_model.h5')
