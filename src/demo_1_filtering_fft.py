import zipfile
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.fft import rfft, rfftfreq
from preprocess_dataset import apply_bandpass_filter

PROJECT_ROOT = Path(r"C:\Users\andre\OneDrive\Рабочий стол\Исследовательский проект")
PTBXL_ZIP = PROJECT_ROOT / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
MUSE_XML = PROJECT_ROOT / "IntroECG-master" / "7-EchoNext Minimodel" / "xml_input" / "MUSE_example.xml"
NPY_FILE = PROJECT_ROOT / "Demo_Data" / "ptbxl_processed" / "records" / "ecg_1.npy"
OUT_DIR = PROJECT_ROOT / "echonext" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_real_ecg(record_index: int = 0):
    """Загружает реальную 12-канальную запись 500 Гц из архива PTB-XL."""
    if PTBXL_ZIP.exists():
        import wfdb
        with zipfile.ZipFile(PTBXL_ZIP, 'r') as z:
            hr_hea = sorted([f for f in z.namelist() if "records500/" in f and f.endswith(".hea")])
            target_hea = hr_hea[record_index]
            target_dat = target_hea[:-4] + ".dat"
            with tempfile.TemporaryDirectory() as tmpdir:
                z.extract(target_hea, tmpdir)
                z.extract(target_dat, tmpdir)
                rec_path = Path(tmpdir) / target_hea[:-4]
                signals, fields = wfdb.rdsamp(str(rec_path))
                lead_names = fields.get("sig_name", [f"Lead_{i+1}" for i in range(12)])
                return np.array(signals, dtype=np.float32), float(fields["fs"]), f"PTB-XL ({Path(target_hea).stem})", lead_names

    if NPY_FILE.exists():
        data = np.load(NPY_FILE)
        if data.shape[0] == 12 and data.shape[1] != 12:
            data = data.T
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        return data, len(data) / 10.0, "ecg_1.npy", lead_names

    raise FileNotFoundError(f"Не найдены данные ЭКГ в: {PROJECT_ROOT}")


def main():
    data, fs, source_label, lead_names = load_real_ecg(record_index=0)
    lead_idx = 7  # Отведение V2
    clean_signal = data[:, lead_idx]
    t = np.arange(len(clean_signal)) / fs

    # Моделирование клинических артефактов: дыхательный дрейф (0.25 Гц) + наводка сети (50 Гц)
    baseline_wander = 0.45 * np.sin(2 * np.pi * 0.25 * t)
    powerline = 0.15 * np.sin(2 * np.pi * 50.0 * t) if fs > 100 else 0.0
    noisy_signal = clean_signal + baseline_wander + powerline

    # Применение нашего эталонного фильтра из боевого модуля preprocess_dataset.py
    filtered_2d = apply_bandpass_filter(noisy_signal[:, None], fs=fs, lowcut=0.5, highcut=40.0, order=4)
    filtered_signal = filtered_2d[:, 0]

    # Расчет одностороннего амплитудного спектра Фурье (rFFT)
    N = len(t)
    xf = rfftfreq(N, d=1.0 / fs)
    yf_raw = np.abs(rfft(noisy_signal - np.mean(noisy_signal))) / N
    yf_filtered = np.abs(rfft(filtered_signal - np.mean(filtered_signal))) / N

    fig, axs = plt.subplots(2, 2, figsize=(15, 8.5))
    fig.suptitle(
        f"Метод 1: Полосовая фильтрация Баттерворта 0.5–40 Гц + Notch 50 Гц ({source_label}, Отведение {lead_names[lead_idx]})",
        fontsize=13, fontweight='bold', y=0.98
    )

    axs[0, 0].plot(t, noisy_signal, color='salmon', linewidth=1.0)
    axs[0, 0].set_title("А) Сигнал ЭКГ до фильтрации (Дрейф изолинии 0.25 Гц + помеха 50 Гц)", fontsize=11)
    axs[0, 0].set_xlabel("Время (секунды)")
    axs[0, 0].set_ylabel("Амплитуда (мВ)")
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)

    axs[0, 1].plot(t, filtered_signal, color='steelblue', linewidth=1.1)
    axs[0, 1].set_title("Б) После нуль-фазовой фильтрации sosfiltfilt (Стабильная изолиния, сдвиг 0 рад)", fontsize=11)
    axs[0, 1].set_xlabel("Время (секунды)")
    axs[0, 1].set_ylabel("Амплитуда (мВ)")
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)

    axs[1, 0].plot(xf, yf_raw, color='salmon', linewidth=1.2)
    axs[1, 0].set_title("В) Амплитудный спектр Фурье (FFT) зашумленного сигнала", fontsize=11)
    axs[1, 0].set_xlabel("Частота (Гц)")
    axs[1, 0].set_ylabel("Нормированная амплитуда")
    axs[1, 0].set_xlim(0, 60)
    axs[1, 0].axvline(50, color='crimson', linestyle='--', alpha=0.7, label='Пик сетевой помехи (50 Гц)')
    axs[1, 0].axvline(0.25, color='darkorange', linestyle=':', alpha=0.9, label='Дрейф дыхания (<0.5 Гц)')
    axs[1, 0].legend(loc='upper right')
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)

    axs[1, 1].plot(xf, yf_filtered, color='steelblue', linewidth=1.2)
    axs[1, 1].set_title("Г) Спектр Фурье после фильтрации (Пики 0.25 Гц и 50 Гц полностью подавлены)", fontsize=11)
    axs[1, 1].set_xlabel("Частота (Гц)")
    axs[1, 1].set_ylabel("Нормированная амплитуда")
    axs[1, 1].set_xlim(0, 60)
    axs[1, 1].axvspan(0.5, 40.0, color='mediumseagreen', alpha=0.15, label='Полезное окно ЭКГ (0.5–40 Гц)')
    axs[1, 1].legend(loc='upper right')
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = OUT_DIR / "fig1_butterworth_filtering.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[УСПЕХ] Обновленный График 1 сохранен: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()