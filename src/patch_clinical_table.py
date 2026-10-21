import numpy as np
import pandas as pd
from pathlib import Path

# --- ПУТИ ---
PROJECT_ROOT = Path(r"C:\Users\andre\OneDrive\Рабочий стол\Исследовательский проект")
PTBXL_DB_CSV = PROJECT_ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3" / "ptbxl_database.csv"
MASTER_CSV_PATH = PROJECT_ROOT / "MedData_AI" / "ptbxl_clinical_and_voltage.csv"


def main():
    print("[1/3] Загрузка существующей таблицы вольтажа и исходной базы PTB-XL...")
    if not MASTER_CSV_PATH.exists():
        raise FileNotFoundError(f"Файл не найден: {MASTER_CSV_PATH}")

    df_master = pd.read_csv(MASTER_CSV_PATH)
    df_ptbxl = pd.read_csv(PTBXL_DB_CSV, usecols=["ecg_id", "patient_id", "strat_fold"])

    # Удаляем старые колонки strat_fold/patient_id, если скрипт запускается повторно
    for col in ["patient_id", "strat_fold", "bmi"]:
        if col in df_master.columns:
            df_master = df_master.drop(columns=[col])

    print("[2/3] Присоединение strat_fold и patient_id (защита от Data Leakage)...")
    df_merged = pd.merge(df_ptbxl, df_master, on="ecg_id", how="inner")

    print("[3/3] Очистка клинических аномалий (HIPAA age=300) и расчет ИМТ (BMI)...")
    # 1. В PTB-XL возраст > 89 закодирован числом 300. Приводим к 90 годам:
    df_merged["age"] = df_merged["age"].clip(upper=90.0)
    # Заполняем единичные пропуски возраста медианой
    df_merged["age"] = df_merged["age"].fillna(df_merged["age"].median())

    # 2. Очистка нереалистичных выбросов роста и веса и расчет BMI
    valid_h = df_merged["height"].where((df_merged["height"] >= 120) & (df_merged["height"] <= 220))
    valid_w = df_merged["weight"].where((df_merged["weight"] >= 30) & (df_merged["weight"] <= 250))
    df_merged["bmi"] = np.round(valid_w / ((valid_h / 100.0) ** 2), 2)

    # Упорядочиваем колонки: ключи -> клиника -> таргеты -> вольтаж
    lead_cols = [c for c in df_merged.columns if c.startswith("lead_")]
    target_cols = ["is_NORM", "is_MI", "is_STTC", "is_CD", "is_HYP"]
    clinical_cols = ["age", "sex", "height", "weight", "bmi"]
    key_cols = ["ecg_id", "patient_id", "strat_fold"]

    ordered_cols = key_cols + clinical_cols + target_cols + lead_cols
    df_merged = df_merged[ordered_cols].sort_values("ecg_id").reset_index(drop=True)

    df_merged.to_csv(MASTER_CSV_PATH, index=False)

    print("\n[УСПЕХ] Мастер-таблица обновлена!")
    print(f"Итоговая размерность: {df_merged.shape}")
    print("Распределение записей по официальным фолдам (1-8 Train, 9 Val, 10 Test):")
    print(df_merged["strat_fold"].value_counts().sort_index().to_dict())
    print(f"Максимальный возраст после очистки: {df_merged['age'].max()} лет")


if __name__ == "__main__":
    main()