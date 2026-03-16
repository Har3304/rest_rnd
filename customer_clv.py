import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




df = pd.read_csv("customer_clv.csv")




df['first_visit'] = pd.to_datetime(df['first_visit'])
df['last_visit'] = pd.to_datetime(df['last_visit'])

df['customer_age_days'] = (df['last_visit'] - df['first_visit']).dt.days
df['days_since_last_visit'] = (pd.Timestamp.today() - df['last_visit']).dt.days

df.drop(['first_visit', 'last_visit'], axis=1, inplace=True)

y = df['estimated_clv_inr']
X = df.drop(['estimated_clv_inr', 'customer_id'], axis=1)

encoder = LabelEncoder()
X['is_loyalty_member'] = encoder.fit_transform(X['is_loyalty_member'])

X = pd.get_dummies(X, columns=['gender', 'area', 'customer_segment'], dtype=int)


X_train, X_test, y_train, y_test = train_test_split(
    X, y.values, test_size=0.2, random_state=42
)
scaler = StandardScaler()

X_train_np = scaler.fit_transform(X_train)
X_test_np = scaler.transform(X_test)




X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=64,
    shuffle=False
)

class RevenueModel(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

model = RevenueModel(X_train.shape[1]).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
epochs = 100
patience = 10

best_loss = float("inf")
early_stop_counter = 0


for epoch in tqdm(range(epochs)):

    model.train()
    train_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        preds = model(batch_X)
        loss = criterion(preds, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    
    
    
    model.eval()
    test_loss = 0

    with torch.no_grad():

        for batch_X, batch_y in test_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            preds = model(batch_X)
            loss = criterion(preds, batch_y)

            test_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_test_loss = test_loss / len(test_loader)

    with torch.no_grad():
        y_pred = model(X_test.to(device)).cpu().numpy()
        y_true = y_test.numpy()

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Test Loss: {avg_test_loss:.4f} | "
        f"MAE: {mae:.2f} | "
        f"RMSE: {rmse:.2f} | "
        f"R²: {r2:.4f}"
    )

    if avg_test_loss < best_loss:
        best_loss = avg_test_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "best_clv_model.pt")
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping triggered")
            break

print("Training Complete")
model.load_state_dict(torch.load("best_clv_model.pt"))
model.eval()

with torch.no_grad():

    predictions = model(X_test.to(device)).cpu().numpy()

print("\nSample Predictions:")
print(predictions[:10])
print(y_test[:10])
