import pickle
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from flwr_datasets.partitioner import DirichletPartitioner
from datasets import Dataset
from sklearn.metrics import roc_auc_score, average_precision_score

import random
import os

# ----------------------------
# (0) Reproducibility (same)
# ----------------------------
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_all(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# ----------------------------
# (1) Load data (same)
# ----------------------------
with open('./data/output/'+'X_train_fold_0.pkl', 'rb') as fp:
    X_train = pickle.load(fp)
with open('./data/output/'+'Y_train_fold_0.pkl', 'rb') as fp:
    Y_train = pickle.load(fp)
with open('./data/output/'+'X_test_fold_0.pkl', 'rb') as fp:
    X_test = pickle.load(fp)
with open('./data/output/'+'Y_test_fold_0.pkl', 'rb') as fp:
    Y_test = pickle.load(fp)

X_all = pd.concat([X_train, X_test], axis=0, ignore_index=True)
y_array = np.concatenate([Y_train.to_numpy(), Y_test.to_numpy()]).astype(np.int64)

# ============================================================
# (2) CHANGED: replace pd.get_dummies(...) + dense huge matrix
#     with integer-encoding categoricals + numeric float32.
#     This avoids the massive one-hot expansion.
# ============================================================

# Identify categorical vs numeric columns
cat_cols = X_all.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X_all.columns if c not in cat_cols]

# Encode categoricals to int64 codes
cat_cardinalities = []
X_cat_np = np.zeros((len(X_all), len(cat_cols)), dtype=np.int64)

for j, c in enumerate(cat_cols):
    codes, uniques = pd.factorize(X_all[c].astype("string").fillna("__MISSING__"))
    X_cat_np[:, j] = codes.astype(np.int64)  # 0..K-1
    cat_cardinalities.append(len(uniques))

# Numeric to float32 (clean inf/nan)
X_num_np = (
    X_all[num_cols]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
    .to_numpy(dtype=np.float32)
)

print("X_cat_shape", X_cat_np.shape, "X_num_shape", X_num_np.shape, "y_len", len(y_array))
print("Num categorical cols:", len(cat_cols), "Num numeric cols:", len(num_cols))


fds = Dataset.from_dict({
    "x_cat": X_cat_np,        # int64
    "x_num": X_num_np,        # float32
    "label": y_array,         # int64
})
fds.set_format(type="torch", columns=["x_cat", "x_num", "label"])


class ResBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)

class TabEncoder(nn.Module):
    """
    Embeds each categorical column and concatenates with numeric features.
    """
    def __init__(self, cat_cardinalities, n_num, emb_dim=16):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(card, emb_dim) for card in cat_cardinalities])
        self.out_dim = emb_dim * len(cat_cardinalities) + n_num

    def forward(self, x_cat, x_num):
        if len(self.embs) > 0:
            e = torch.cat([emb(x_cat[:, i]) for i, emb in enumerate(self.embs)], dim=1)
            return torch.cat([e, x_num], dim=1)
        return x_num

class FederatedResNet(nn.Module):
    def __init__(self, cat_cardinalities, n_num, hidden_dim=128, n_blocks=3, dropout=0.5, emb_dim=16):
        super().__init__()

        # --- INPUT ENCODER (NEW) ---
        self.encoder = TabEncoder(cat_cardinalities, n_num, emb_dim=emb_dim)

        # --- SHARED BODY (same idea) ---
        layers = [nn.Linear(self.encoder.out_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_blocks):
            layers.append(ResBlock(hidden_dim, dropout))
        self.body = nn.Sequential(*layers)

        # --- LOCAL HEAD (same) ---
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x_cat, x_num):
        x = self.encoder(x_cat, x_num)
        features = self.body(x)
        return self.head(features)

# ----------------------------
# (5) Train/Test (CHANGED): read x_cat + x_num instead of features
# ----------------------------
def train(net, trainloader, epochs, lr, momentum, weight_decay, device, reg_params=None, lamda=0.0):
    net.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)

    net.train()
    total_loss = 0.0

    for epoch in range(epochs):
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for batch in pbar:
            x_cat = batch["x_cat"].to(device)
            x_num = batch["x_num"].to(device)
            y = batch["label"].to(device).float().unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            logits = net(x_cat, x_num)
            loss = criterion(logits, y)

            if reg_params is not None and lamda > 0:
                proximal_term = 0.0
                for local_p, global_p in zip(net.parameters(), reg_params):
                    proximal_term += (local_p - global_p).pow(2).sum()
                loss = loss + (lamda / 2) * proximal_term

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return total_loss / (len(trainloader) * epochs)

def test(net, testloader, device):
    net.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")

    total_loss = 0.0
    all_y_true = []
    all_y_probs = []

    net.eval()
    with torch.no_grad():
        for batch in testloader:
            x_cat = batch["x_cat"].to(device)
            x_num = batch["x_num"].to(device)
            y = batch["label"].to(device).float().unsqueeze(1)

            logits = net(x_cat, x_num)
            total_loss += criterion(logits, y).item()

            probs = torch.sigmoid(logits)
            all_y_true.append(y.cpu().numpy())
            all_y_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_y_true)
    y_probs = np.concatenate(all_y_probs)

    loss = total_loss / len(testloader.dataset)
    auroc = roc_auc_score(y_true, y_probs)
    auprc = average_precision_score(y_true, y_probs)

    preds = (y_probs > 0.5).astype(int)
    accuracy = (preds == y_true).mean()

    return loss, accuracy, auroc, auprc

# ----------------------------
# (6) Data partitioning (same logic; uses fds with new columns)
# ----------------------------
def load_data(partition_id: int, num_partitions: int, batch_size: int, dataset_split_arg, seed: int):
    partitioner = DirichletPartitioner(
        num_partitions=num_partitions,
        partition_by="label",
        alpha=dataset_split_arg,
        min_partition_size=100,
        self_balancing=True,
        seed=seed
    )

    partitioner.dataset = fds
    client_dataset = partitioner.load_partition(partition_id)

    partition_train_val = client_dataset.train_test_split(test_size=0.2, seed=seed)
    train_ds = partition_train_val["train"]
    val_ds = partition_train_val["test"]

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, valloader

trainl, vall = load_data(0, 6, 32, 1000, seed=42)


y_tr = trainl.dataset["label"]
n_tr = len(y_tr)
n1_tr = (y_tr == 1).sum().item()
n0_tr = (y_tr == 0).sum().item()

y_va = vall.dataset["label"]
n_va = len(y_va)
n1_va = (y_va == 1).sum().item()
n0_va = (y_va == 0).sum().item()

print("Samples:", f"Train: {n_tr}", f"Validation: {n_va}")
print(f"Train Readmissions (1): {n1_tr} ({100*n1_tr/n_tr:.2f}%)", f"Validation Readmissions (1): {n1_va} ({100*n1_va/n_va:.2f}%)")
print(f"Train No Readmission (0): {n0_tr} ({100*n0_tr/n_tr:.2f}%)", f"Validation No Readmission (0): {n0_va} ({100*n0_va/n_va:.2f}%)")

# ----------------------------
# (8) Model init (CHANGED: input_dim replaced by (cat_cardinalities, n_num))
# ----------------------------
model = FederatedResNet(
    cat_cardinalities=cat_cardinalities,
    n_num=X_num_np.shape[1],
    hidden_dim=128,
    n_blocks=3,
    dropout=0.5,
    emb_dim=16
)

epochs = 50
print(f"\n Training Resnet for {epochs} epochs")
train(model, trainl, epochs, 0.001, 0.9, 1e-4, device)
print(" Results loss, accuracy, auroc, auprc", test(model, vall, device))
