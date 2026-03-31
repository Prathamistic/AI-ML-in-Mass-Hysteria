import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from .config import LSTM_WINDOW_SIZE, CHECKPOINTS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden_dim: int):
        super(LSTMAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=n_features, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=0.1
        )
        # Latent/Bottleneck state representation
        self.encoder_state = None 
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, # Can be larger depending on reconstruction complexity
            num_layers=2, 
            batch_first=True,
            dropout=0.1
        )
        
        self.reconstructor = nn.Linear(hidden_dim, n_features)
        
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim

    def forward(self, x):
        batch_size = x.shape[0]
        
        # Encode
        encoded, (hidden, cell) = self.encoder(x)
        self.encoder_state = hidden[-1]  # Save bottleneck state for clustering later
        
        # Repeat the last hidden state for the sequence length
        repeated_state = self.encoder_state.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # Decode
        decoded, _ = self.decoder(repeated_state)
        
        # Reconstruct exactly the features
        reconstruction = self.reconstructor(decoded)
        
        return reconstruction

def create_sequences(df: pd.DataFrame, window_size: int = LSTM_WINDOW_SIZE):
    """
    Transforms dataframe into 3D historical windows (samples, time_steps, features).
    Imputes NaN forwards via pad/ffill. No leakage back filling.
    """
    df = df.copy()
    feature_cols = ['volume', 'mean_sentiment', 'pct_negative', 
                    'mean_distress', 'rumor_velocity', 'retweet_ratio', 'crisis_kw_density']
    
    # Forward fill missing days chronologically starting from min date
    df[feature_cols] = df[feature_cols].ffill()
    
    # Drop rows entirely if still NaN (i.e. very first days of collection)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    
    feature_data = df[feature_cols].values
    
    X = []
    # e.g. for row 14, window is [0:14], so we capture 14 exact steps
    for i in range(len(feature_data) - window_size + 1):
        X.append(feature_data[i: i + window_size])
        
    return np.array(X)

def train_lstm_autoencoder(X: np.ndarray, split_ratio=0.7, epochs=50, lr=1e-3, hidden_dim=32):
    """
    Main training loop. Ensures chronological holdout sequence. Fits on Train, evals Validation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Training on {device}...")
    
    # 1. Strict time-aware split
    train_size = int(len(X) * split_ratio)
    X_train = X[:train_size]
    X_val = X[train_size:]
    
    # 2. Fit Scalers explicitly on train dataset (no leakage!)
    # X shape: (samples, time_steps, features)
    n_features = X.shape[2]
    
    # Reshape train to 2D for scaling, then shape back
    X_train_reshaped = X_train.reshape(-1, n_features)
    scaler = StandardScaler()
    scaler.fit(X_train_reshaped)
    
    X_train_scaled = scaler.transform(X_train_reshaped).reshape(X_train.shape)
    
    if len(X_val) > 0:
         X_val_reshaped = X_val.reshape(-1, n_features)
         X_val_scaled = scaler.transform(X_val_reshaped).reshape(X_val.shape)
    else:
         X_val_scaled = np.array([])
    
    # 3. Model setup
    model = LSTMAutoencoder(seq_len=LSTM_WINDOW_SIZE, n_features=n_features, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    train_dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False) # Important: Keep time series sequential! Alternatively, shuffling autoencoder windows IS permissible if using stateless LSTM, but let's be safe. Let's shuffle since individual sliding windows are structurally independent.
    # Actually, shuffling is okay for independent sliding windows:
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) 
    
    if len(X_val_scaled) > 0:
        val_dataset = TensorDataset(torch.tensor(X_val_scaled, dtype=torch.float32))
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    best_val_loss = float('inf')
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINTS_DIR, 'best_lstm_ae.pt')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed = model(x_batch)
            loss = criterion(reconstructed, x_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation Loop
        val_loss = 0
        if len(X_val_scaled) > 0:
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    x_batch = batch[0].to(device)
                    reconstructed = model(x_batch)
                    loss = criterion(reconstructed, x_batch)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), checkpoint_path)
                
        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
    logging.info(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    return model, scaler

def extract_hidden_states(model, X_scaled):
    """
    Feed whole scaled sequence chronologically to get 1D feature representation for changepoints.
    """
    device = next(model.parameters()).device
    model.eval()
    
    tensor_data = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        _ = model(tensor_data)
        hidden_states = model.encoder_state.cpu().numpy()
        
    return hidden_states
