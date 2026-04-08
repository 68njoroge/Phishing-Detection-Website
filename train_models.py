import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
import joblib

from features import extract_all_features, url_to_sequence

# -------------------------
# Device config
# -------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# -------------------------
# LSTM Model  (imported by app.py)
# -------------------------
class LSTMModel(nn.Module):
    def __init__(self, vocab_size=50, embed_dim=64, hidden_dim=128):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=0.3
        )

        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat((h[-2, :, :], h[-1, :, :]), dim=1)
        return self.fc(h)


# -------------------------
# Training  (only when run directly)
# -------------------------
if __name__ == "__main__":

    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('phishing_site_urls.csv')

    print(f"Columns found: {df.columns.tolist()}")
    print(f"Raw shape: {df.shape}")

    # Flexible column detection
    url_col = None
    label_col = None

    for col in df.columns:
        if col.lower() in ('url', 'urls', 'domain', 'address'):
            url_col = col
        if col.lower() in ('label', 'labels', 'result', 'target', 'class', 'type'):
            label_col = col

    if url_col is None or label_col is None:
        print(f"Could not auto-detect columns. Available: {df.columns.tolist()}")
        raise ValueError("Please check your CSV column names.")

    print(f"Using URL column: '{url_col}' | Label column: '{label_col}'")

    df = df[[url_col, label_col]].copy()
    df.columns = ['url', 'label']

    # Label normalization
    print(f"Unique label values: {df['label'].unique()}")

    label_map = {
        'good': 0, 'bad': 1,
        'legitimate': 0, 'phishing': 1,
        'benign': 0, 'malicious': 1,
        'safe': 0, 'unsafe': 1,
        '0': 0, '1': 1, '-1': 1,
    }

    if df['label'].dtype == object:
        df['label'] = df['label'].str.lower().str.strip().map(label_map)
    else:
        df['label'] = df['label'].map({1: 0, -1: 1}).fillna(df['label'])
        df['label'] = df['label'].astype(int)

    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    # Clean URLs
    df = df.drop_duplicates(subset='url')
    df['url'] = (
        df['url']
        .str.lower()
        .str.strip()
        .str.rstrip('/')
        .str.split('?').str[0]
    )
    df = df.dropna(subset=['url'])
    df = df[df['url'].str.len() > 3]

    print(f"Clean dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    # XGBoost features
    print("Extracting features... (this may take a few minutes)")
    feature_list = []
    for i, url in enumerate(df['url']):
        if i % 10000 == 0:
            print(f"  Processing {i}/{len(df)}...")
        try:
            features = extract_all_features(url)
            feature_list.append(list(features.values()))
        except Exception:
            feature_list.append([0] * 29)

    X = np.array(feature_list)
    y = df['label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        objective='binary:logistic',
        eval_metric='logloss',
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    joblib.dump(xgb_model, 'xgb_model.pkl')
    print("XGBoost model saved.")

    y_pred_xgb = xgb_model.predict(X_test)
    print(
        f"XGBoost — Acc: {accuracy_score(y_test, y_pred_xgb):.4f}  "
        f"Prec: {precision_score(y_test, y_pred_xgb):.4f}  "
        f"Recall: {recall_score(y_test, y_pred_xgb):.4f}  "
        f"F1: {f1_score(y_test, y_pred_xgb):.4f}"
    )
    print("XGBoost Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

    # Character map
    print("Building character map...")
    all_chars  = set(''.join(df['url']))
    char_map   = {c: i + 1 for i, c in enumerate(sorted(all_chars))}
    vocab_size = len(char_map) + 1
    joblib.dump(char_map, 'char_map.pkl')
    print(f"Vocab size: {vocab_size}")

    # LSTM sequences
    print("Building LSTM sequences... (this may take a few minutes)")
    X_seq = np.stack([
        url_to_sequence(url, max_len=200, char_map=char_map)
        for url in df['url']
    ])

    X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(
        X_seq, y, test_size=0.2, random_state=42
    )

    model     = LSTMModel(vocab_size=vocab_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    scaler    = GradScaler()

    train_dataset = TensorDataset(
        torch.tensor(X_seq_train, dtype=torch.long),
        torch.tensor(y_seq_train, dtype=torch.float).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    # Train LSTM
    print("Training LSTM...")
    for epoch in range(20):
        model.train()
        total_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            with autocast():
                out  = model(batch_x)
                loss = criterion(out, batch_y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1:>2}  Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), 'lstm_model.pth')
    print("LSTM model saved.")

    # Evaluate LSTM
    print("Evaluating LSTM...")
    model.eval()
    all_preds = []

    test_dataset = TensorDataset(
        torch.tensor(X_seq_test, dtype=torch.long),
        torch.tensor(y_seq_test, dtype=torch.float).unsqueeze(1)
    )
    test_loader = DataLoader(test_dataset, batch_size=512)

    with torch.no_grad():
        for batch_x_test, _ in test_loader:
            batch_x_test = batch_x_test.to(device)
            outputs = torch.sigmoid(model(batch_x_test))
            all_preds.append(outputs.cpu().numpy())

    y_pred_lstm = (np.concatenate(all_preds) > 0.5).astype(float).squeeze()
    print(
        f"LSTM — Acc: {accuracy_score(y_seq_test, y_pred_lstm):.4f}  "
        f"Prec: {precision_score(y_seq_test, y_pred_lstm):.4f}  "
        f"Recall: {recall_score(y_seq_test, y_pred_lstm):.4f}  "
        f"F1: {f1_score(y_seq_test, y_pred_lstm):.4f}"
    )
    print("LSTM Confusion Matrix:\n", confusion_matrix(y_seq_test, y_pred_lstm))
    print("\n✅ All models trained and saved successfully!")
    print("You can now run: python app.py")