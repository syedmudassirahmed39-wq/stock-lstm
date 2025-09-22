from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

app = FastAPI()

# ----------------------------
# LSTM Model Definition
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

model_path = "model.pth"
sequence_length = 60

# ----------------------------
# Function to train and save model
# ----------------------------
def train_and_save_model():
    print("Training model...")
    ticker = 'AAPL'  # Default for training
    data = yf.download(ticker, start='2010-01-01', end='2025-09-22')[['Close']]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length])
    X = np.array(X)
    y = np.array(y)

    # Convert to tensors
    X_train = torch.tensor(X, dtype=torch.float32)
    y_train = torch.tensor(y, dtype=torch.float32)

    # Initialize model
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Save model
    torch.save(model.state_dict(), model_path)
    print("Model trained and saved.")
    return model

# ----------------------------
# Load or Train Model
# ----------------------------
model = LSTMModel()
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded from disk.")
else:
    model = train_and_save_model()
    model.eval()

# ----------------------------
# FastAPI Routes
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    html_content = """
    <html>
        <head>
            <title>Stock Prediction LSTM</title>
        </head>
        <body>
            <h1>Stock Price Prediction API</h1>
            <p>Use the endpoint <code>/predict/&lt;ticker&gt;</code> to get predictions.</p>
            <p>Example: <a href="/predict/AAPL">/predict/AAPL</a></p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/predict/{ticker}")
def predict_stock(ticker: str):
    # Download data
    data = yf.download(ticker, start='2010-01-01', end='2025-09-22')[['Close']]
    if len(data) < sequence_length:
        return {"error": "Not enough data for this ticker."}

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(data)

    last_sequence = torch.tensor(scaled_data[-sequence_length:], dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        future_price = model(last_sequence)
        future_price = scaler.inverse_transform(future_price.numpy())

    return {"ticker": ticker, "predicted_next_price": float(future_price[0][0])}

