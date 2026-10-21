import os
import ast
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from tqdm import tqdm
from scipy.signal import resample_poly, butter, sosfiltfilt, iirnotch, tf2sos

# --- ПУТИ (ИЗМЕНИТЕ НА СВОИ, ГДЕ РАСПАКОВАН АРХИВ) ---
PROJECT_ROOT = Path(r"C:\Users\andre\OneDrive\Рабочий стол\Исследовательский проект")
# Путь к распакованной папке PTB-XL
PTBXL_DIR = PROJECT_ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
OUT_RECORDS_DIR = PROJECT_ROOT / "MedData_AI" / "processed_tensors"
OUT_CSV_PATH = PROJECT_ROOT / "MedData_AI" / "ptbxl_clinical_and_voltage.csv"


# --- ЭТАЛОННАЯ ФИЛЬТРАЦИЯ (Из прошлого модуля) ---
def apply_bandpass_filter(signal: np.ndarray, fs: float) -> np.ndarray:
    filtered = signal.copy()
    nyquist = 0.5 * fs
    if fs > 100.0:
        b_notch, a_notch = iirnotch(w0=50.0, Q=30.0, fs=fs)
        sos_notch = tf2sos(b_notch, a_notch)
        filtered = sosfiltfilt(sos_notch, filtered, axis=0)

    sos_band = butter(4, [0.5 / nyquist, min(40.0 / nyquist, 0.99)], btype="bandpass", output="sos")
    return sosfiltfilt(sos_band, filtered, axis=0)


def apply_polyphase_resampling(signal: np.ndarray, orig_fs: int, target_fs: int = 250) -> np.ndarray:
    if int(orig_fs) == int(target_fs):
        return signal
    gcd = np.gcd(int(orig_fs), int(target_fs))
    return resample_poly(signal, up=int(target_fs // gcd), down=int(orig_fs // gcd), axis=0)


def apply_z_normalization_with_voltage(signal: np.ndarray) -> tuple[np.ndarray, dict]:
    means = np.mean(signal, axis=0, keepdims=True)
    stds = np.std(signal, axis=0, keepdims=True)
    ptps = np.ptp(signal, axis=0)

    voltage_stats = {}
    for i in range(12):
        voltage_stats[f"lead_{i + 1}_std"] = round(float(stds[0, i]), 6)
        voltage_stats[f"lead_{i + 1}_ptp"] = round(float(ptps[i]), 6)

    safe_stds = np.where(stds > 1e-8, stds, 1.0)
    normalized = ((signal - means) / safe_stds).astype(np.float32)
    return normalized, voltage_stats


# --- ИЗВЛЕЧЕНИЕ ДИАГНОЗОВ (TARGETS) ---
def load_and_map_targets():
    print("[1/3] Загрузка и маппинг клинических диагнозов PTB-XL...")

    db_path = PTBXL_DIR / 'ptbxl_database.csv'
    scp_path = PTBXL_DIR / 'scp_statements.csv'

    if not db_path.exists() or not scp_path.exists():
        raise FileNotFoundError(f"Не найдены CSV файлы базы данных в {PTBXL_DIR}")

    # Загружаем базу пациентов
    Y = pd.read_csv(db_path, index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))  # Парсим строки словарей

    # Загружаем справочник диагнозов
    agg_df = pd.read_csv(scp_path, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]  # Берем только диагностические метки

    # Функция сопоставления диагнозов с суперклассами (NORM, MI, STTC, CD, HYP)
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    # Так как у пациента может быть несколько диагнозов, создадим Multi-Hot Encoding
    # Для бинарной классификации можно будет потом выбрать один класс (например, MI или NORM)
    Y['is_NORM'] = Y['diagnostic_superclass'].apply(lambda x: 1 if 'NORM' in x else 0)
    Y['is_MI'] = Y['diagnostic_superclass'].apply(lambda x: 1 if 'MI' in x else 0)
    Y['is_STTC'] = Y['diagnostic_superclass'].apply(lambda x: 1 if 'STTC' in x else 0)
    Y['is_CD'] = Y['diagnostic_superclass'].apply(lambda x: 1 if 'CD' in x else 0)
    Y['is_HYP'] = Y['diagnostic_superclass'].apply(lambda x: 1 if 'HYP' in x else 0)

    return Y


# --- ГЛАВНЫЙ ПАЙПЛАЙН ---
def main():
    OUT_RECORDS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Получаем базу пациентов с диагнозами
    df_clinical = load_and_map_targets()

    print(f"[2/3] Начинаем цифровую обработку {len(df_clinical)} записей ЭКГ...")

    final_metadata = []

    # Итерация с прогресс-баром
    for ecg_id, row in tqdm(df_clinical.iterrows(), total=len(df_clinical), desc="Обработка сигналов"):
        # Путь к сырому файлу .dat / .hea (в ptbxl_database.csv они указаны в колонке filename_hr)
        # Мы используем filename_hr (500 Гц) для максимального качества
        record_rel_path = row['filename_hr']
        record_full_path = str(PTBXL_DIR / record_rel_path)

        try:
            # Чтение сигнала
            raw_signal, fields = wfdb.rdsamp(record_full_path)
            orig_fs = int(fields["fs"])

            # Применение ЦОС (DSP)
            if raw_signal.shape[0] == 12 and raw_signal.shape[1] != 12:
                raw_signal = raw_signal.T

            clean_signal = apply_bandpass_filter(raw_signal, fs=orig_fs)
            resampled_signal = apply_polyphase_resampling(clean_signal, orig_fs=orig_fs, target_fs=250)
            final_tensor, voltage_stats = apply_z_normalization_with_voltage(resampled_signal)

            # Сохранение тензора
            out_npy = OUT_RECORDS_DIR / f"{ecg_id:05d}.npy"
            np.save(out_npy, final_tensor)

            # Сбор всех признаков (Клинические + Диагнозы + Вольтаж)
            record_meta = {
                "ecg_id": ecg_id,
                "age": row.get('age', np.nan),
                "sex": row.get('sex', np.nan),
                "height": row.get('height', np.nan),
                "weight": row.get('weight', np.nan),
                "is_NORM": row['is_NORM'],
                "is_MI": row['is_MI'],
                "is_STTC": row['is_STTC'],
                "is_CD": row['is_CD'],
                "is_HYP": row['is_HYP']
            }
            record_meta.update(voltage_stats)
            final_metadata.append(record_meta)

        except Exception as e:
            # Игнорируем битые файлы, если такие попадутся в архиве
            print(f"\nОшибка чтения ecg_id={ecg_id}: {e}")
            continue

    # 3. Сохранение финальной базы
    print("\n[3/3] Сохранение мастер-таблицы признаков...")
    df_final = pd.DataFrame(final_metadata)
    df_final.to_csv(OUT_CSV_PATH, index=False)

    print(f"\n[УСПЕХ] Обработка завершена!")
    print(f"Сохранено тензоров: {len(df_final)}")
    print(f"Мастер-таблица сохранена в: {OUT_CSV_PATH}")


if __name__ == "__main__":
    main()