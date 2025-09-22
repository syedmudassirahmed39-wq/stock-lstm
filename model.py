import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ----------------------------
# 1. Download Stock Data
# ----------------------------
ticker = 'AAPL'  # Change this to 'MSFT', 'DELL', 'HPQ', etc.
data = yf.download(ticker, start='2010-01-01', end='2025-09-22')
data = data[['Close']]

# ----------------------------
# 2. Preprocess Data
# ----------------------------
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(data)

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 60
X, y = create_sequences(scaled_data, sequence_length)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# ----------------------------
# 3. Define LSTM Model
# ----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

model = LSTMModel()
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----------------------------
# 4. Training Loop
# ----------------------------
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    y_train_pred = model(X_train)
    loss = loss_function(y_train_pred.squeeze(), y_train)
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# ----------------------------
# 5. Evaluation
# ----------------------------
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test)

# Inverse scaling
y_test_pred = scaler.inverse_transform(y_test_pred.numpy())
y_test_actual = scaler.inverse_transform(y_test.numpy())

# Plot results
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label='Actual Prices')
plt.plot(y_test_pred, label='Predicted Prices')
plt.title(f'{ticker} Stock Price Prediction')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()

# ----------------------------
# 6. Predict Next Day Price
# ----------------------------
with torch.no_grad():
    last_sequence = torch.tensor(scaled_data[-sequence_length:], dtype=torch.float32).unsqueeze(0)
    future_price = model(last_sequence)
    future_price = scaler.inverse_transform(future_price.numpy())
    print(f'Predicted next price for {ticker}: {future_price[0][0]:.2f}')
