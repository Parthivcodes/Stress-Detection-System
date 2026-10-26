"""
Model Architectures for Stress Level Detection:
- Logistic Regression (Simple baseline)
- Random Forest (Non-linear classical ML baseline)
- XGBoost (Strong tree-based ML model)
- 1D CNN (Learns local patterns from physiological feature vectors)
- LSTM (Learns sequential/temporal representations)
- CNN-LSTM (Combines local pattern extraction and temporal modeling)
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. PyTorch Dataset Definition
# ==========================================
class PhysiologicalDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================================
# 2. Deep Learning Architectures
# ==========================================
class Conv1DNet(nn.Module):
    """
    1D Convolutional Neural Network:
    Applies 1D convolutions across physiological feature dimensions to extract
    local invariant feature patterns and high-order interactions.
    """
    def __init__(self, input_dim: int, num_classes: int = 3):
        super().__init__()
        # Input shape: (Batch, Channels=1, Length=input_dim)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(64 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (Batch, input_dim) -> (Batch, 1, input_dim)
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.flatten(1)
        x = self.dropout(self.relu(self.fc1(x)))
        logits = self.fc2(x)
        return logits


class LSTMNet(nn.Module):
    """
    Optimized Recurrent Neural Network (LSTM) with Attention Pooling & MLP Skip Connection:
    - Compact 1-layer Bidirectional LSTM (hidden_dim=32) prevents over-fitting on tabular cohorts.
    - Self-Attention Pooling allows dynamic feature weighting regardless of arbitrary column order.
    - MLP Skip connection provides direct gradient flow from raw normalized features.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1, num_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.skip = nn.Linear(input_dim, 32)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: (Batch, input_dim)
        raw_skip = torch.relu(self.skip(x))
        x_seq = x.unsqueeze(-1)  # (Batch, input_dim, 1)
        out, _ = self.lstm(x_seq) # (Batch, input_dim, hidden_dim * 2)
        
        # Dynamic Attention Pooling over sequence steps
        attn_weights = torch.softmax(self.attn(out), dim=1)
        pooled = torch.sum(out * attn_weights, dim=1) # (Batch, hidden_dim * 2)
        
        # Combine recurrent attention features with raw skip projection
        combined = torch.cat([pooled, raw_skip], dim=1) # (Batch, hidden_dim * 2 + 32)
        logits = self.fc(combined)
        return logits


class CNNLSTMNet(nn.Module):
    """
    Hybrid CNN-LSTM Architecture:
    First extracts localized hierarchical patterns from physiological signals via 1D CNN,
    then processes the transformed latent sequence through an LSTM to capture sequential dynamics.
    """
    def __init__(self, input_dim: int, num_classes: int = 3):
        super().__init__()
        # 1D CNN Feature Extractor
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # Sequence length after pool: ceil(input_dim / 2)
        seq_len_post_pool = input_dim // 2

        # LSTM Layer processing sequential feature slices
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: (Batch, input_dim) -> (Batch, 1, input_dim)
        x = x.unsqueeze(1)
        feat = self.conv(x)  # (Batch, 32, seq_len)
        feat = feat.permute(0, 2, 1)  # (Batch, seq_len, 32)
        lstm_out, (hn, cn) = self.lstm(feat)
        last_step = lstm_out[:, -1, :]  # Take last recurrent step
        logits = self.fc(last_step)
        return logits

# ==========================================
# 3. Classical Machine Learning Builders
# ==========================================
def build_logistic_regression():
    return LogisticRegression(
        max_iter=1000,
        C=1.0,
        random_state=42,
        class_weight='balanced'
    )

def build_random_forest():
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        min_samples_split=4,
        random_state=42,
        class_weight='balanced'
    )

def build_xgboost():
    return XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric='mlogloss',
        random_state=42
    )

# ==========================================
# 4. Deep Learning Training Loop with Early Stopping
# ==========================================
def train_deep_model(model: nn.Module, 
                     train_loader: DataLoader, 
                     val_loader: DataLoader, 
                     epochs: int = 70, 
                     lr: float = 0.001, 
                     patience: int = 12):
    """
    Standard PyTorch training loop with learning rate scheduling,
    validation loss tracking, and early stopping to guarantee best generalization.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_loss = float('inf')
    best_weights = None
    no_improve_count = 0

    for epoch in range(epochs):
        # Training Phase
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            outputs = model(X_b)
            loss = criterion(outputs, y_b)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(y_b)
            preds = outputs.argmax(dim=1)
            correct += (preds == y_b).sum().item()
            total += len(y_b)

        train_loss = total_loss / total
        train_acc = correct / total

        # Validation Phase
        model.eval()
        val_total_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                outputs = model(X_b)
                loss = criterion(outputs, y_b)

                val_total_loss += loss.item() * len(y_b)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == y_b).sum().item()
                val_total += len(y_b)

        val_loss = val_total_loss / val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # Early Stopping Check
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                # Early stop reached
                break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, history

def predict_deep_model(model: nn.Module, X: np.ndarray):
    """
    Generate predicted classes and class probability distributions for test samples.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
    return preds, probs
