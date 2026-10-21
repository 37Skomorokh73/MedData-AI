import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# --- ПУТИ ---
PROJECT_ROOT = Path(r"C:\Users\andre\OneDrive\Рабочий стол\Исследовательский проект")
TENSORS_DIR = PROJECT_ROOT / "MedData_AI" / "processed_tensors"
MASTER_CSV_PATH = PROJECT_ROOT / "MedData_AI" / "ptbxl_clinical_and_voltage.csv"
WEIGHTS_OUT_PATH = PROJECT_ROOT / "MedData_AI" / "echonext_mini_v2_weights.pth"
EMBEDDINGS_OUT_CSV = PROJECT_ROOT / "MedData_AI" / "echonext_v2_embeddings.csv"

TARGET_COLS = ["is_NORM", "is_MI", "is_STTC", "is_CD", "is_HYP"]


# --- 1. ДАТАСЕТ PYTORCH ---
class PTBXLDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tensors_dir: Path):
        self.ecg_ids = df["ecg_id"].values.astype(int)
        self.labels = df[TARGET_COLS].values.astype(np.float32)
        self.tensors_dir = tensors_dir

    def __len__(self):
        return len(self.ecg_ids)

    def __getitem__(self, idx):
        ecg_id = self.ecg_ids[idx]
        tensor_path = self.tensors_dir / f"{ecg_id:05d}.npy"
        # Читаем (2500, 12) и транспонируем в (12, 2500) для Conv1d
        ecg_np = np.load(tensor_path).T
        return torch.from_numpy(ecg_np).float(), torch.from_numpy(self.labels[idx]), ecg_id


# --- 2. УЛУЧШЕННАЯ АРХИТЕКТУРА ECHONEXT-MINI V2 (Без линейного вырождения) ---
class EchoNextMiniV2(nn.Module):
    def __init__(self, in_channels=12, embed_dim=128, num_classes=5):
        super().__init__()
        # Блок 1: локальная морфология зубцов Q, R, S
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(32)

        # Блок 2: сегмент ST и зубец T (с дилатацией для расширения обзора)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=6, dilation=2)
        self.bn2 = nn.BatchNorm1d(64)

        # Блок 3: межканальные паттерны сердечного цикла (dilation=4)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=8, dilation=4)
        self.bn3 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Комбинация среднего и максимального пулинга (128 + 128 = 256)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Нелинейный проекционный слой полного ранга (256 -> 128)
        self.projection = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # Классификационная голова (5 макроклассов)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x, return_embeddings=False):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        avg_x = torch.flatten(self.avg_pool(x), 1)
        max_x = torch.flatten(self.max_pool(x), 1)
        pooled = torch.cat([avg_x, max_x], dim=1)  # (Batch, 256)

        embeddings = self.projection(pooled)  # (Batch, 128) - полный ранг!
        logits = self.classifier(embeddings)  # (Batch, 5)

        if return_embeddings:
            return logits, embeddings
        return logits


def evaluate_auc(model, loader, device):
    model.eval()
    all_targets, all_probs = [], []
    with torch.no_grad():
        for bx, by, _ in loader:
            bx = bx.to(device)
            probs = torch.sigmoid(model(bx)).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(by.numpy())
    y_true = np.vstack(all_targets)
    y_prob = np.vstack(all_probs)
    return roc_auc_score(y_true, y_prob, average="macro")


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[СТАРТ] Устройство вычислений: {device}")

    df = pd.read_csv(MASTER_CSV_PATH)
    if "strat_fold" not in df.columns:
        raise ValueError("Сначала запустите patch_clinical_table.py для добавления strat_fold!")

    # СТРОГОЕ РАЗДЕЛЕНИЕ ПО СТАНДАРТУ PTB-XL:
    # Фолды 1..8 — обучение CNN, Фолд 9 — валидация, Фолд 10 — неприкосновенный тест
    df_train = df[df["strat_fold"].isin([1, 2, 3, 4, 5, 6, 7, 8])].reset_index(drop=True)
    df_val = df[df["strat_fold"] == 9].reset_index(drop=True)

    print(f"Train выборка (фолды 1-8): {len(df_train)} ЭКГ | Val выборка (фолд 9): {len(df_val)} ЭКГ")

    train_loader = DataLoader(PTBXLDataset(df_train, TENSORS_DIR), batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(PTBXLDataset(df_val, TENSORS_DIR), batch_size=256, shuffle=False, num_workers=0)

    model = EchoNextMiniV2(in_channels=12, embed_dim=128, num_classes=5).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    EPOCHS = 4
    best_val_auc = 0.0

    print("\n[1/2] Обучение EchoNextMiniV2 строго на фолдах 1-8...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Эпоха {epoch + 1}/{EPOCHS}")
        for bx, by, _ in pbar:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        val_auc = evaluate_auc(model, val_loader, device)
        print(
            f" --> Эпоха {epoch + 1} | Train Loss: {running_loss / len(train_loader):.4f} | Val Macro ROC-AUC (Fold 9): {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), WEIGHTS_OUT_PATH)

    print(f"\nЛучшие веса (Val AUC = {best_val_auc:.4f}) сохранены в: {WEIGHTS_OUT_PATH}")

    # --- 3. БЫСТРОЕ ПАКЕТНОЕ ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ ДЛЯ ВСЕХ 21 837 ПАЦИЕНТОВ ---
    print("\n[2/2] Пакетное извлечение 128 эмбеддингов и 5 вероятностей для всей базы...")
    model.load_state_dict(torch.load(WEIGHTS_OUT_PATH, map_location=device, weights_only=True))
    model.eval()

    full_loader = DataLoader(PTBXLDataset(df, TENSORS_DIR), batch_size=256, shuffle=False, num_workers=0)

    all_ids, all_probs, all_embs = [], [], []
    with torch.no_grad():
        for bx, _, becg_id in tqdm(full_loader, desc="Пакетный инференс"):
            bx = bx.to(device)
            logits, embs = model(bx, return_embeddings=True)
            probs = torch.sigmoid(logits).cpu().numpy()

            all_ids.append(becg_id.numpy())
            all_probs.append(probs)
            all_embs.append(embs.cpu().numpy())

    ids_arr = np.concatenate(all_ids).reshape(-1, 1)
    probs_arr = np.vstack(all_probs)
    embs_arr = np.vstack(all_embs)

    prob_cols = [f"cnn_prob_{c.replace('is_', '')}" for c in TARGET_COLS]
    emb_cols = [f"emb_{i}" for i in range(128)]

    df_out = pd.DataFrame(
        np.hstack([ids_arr, np.round(probs_arr, 6), np.round(embs_arr, 6)]),
        columns=["ecg_id"] + prob_cols + emb_cols
    )
    df_out["ecg_id"] = df_out["ecg_id"].astype(int)
    df_out.to_csv(EMBEDDINGS_OUT_CSV, index=False)

    print(f"\n[УСПЕХ] Матрица нейросетевых признаков сохранена: {EMBEDDINGS_OUT_CSV}")
    print(f"Размерность: {df_out.shape} (1 ecg_id + 5 вероятностей + 128 независимых эмбеддингов)")


if __name__ == "__main__":
    main()