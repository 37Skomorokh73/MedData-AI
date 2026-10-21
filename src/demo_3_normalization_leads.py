import numpy as np
import matplotlib.pyplot as plt
from demo_1_filtering_fft import load_real_ecg, OUT_DIR
from preprocess_dataset import apply_bandpass_filter


def main():
    # Берем запись №2 (00002_hr), где естественный размах в V2 составляет 4.01 мВ против 0.51 мВ в aVL
    data, fs, source_label, lead_names = load_real_ecg(record_index=1)
    clean_data = apply_bandpass_filter(data, fs=fs)
    t = np.arange(len(clean_data)) / fs

    # Сравниваем реальные отведения без искусственных множителей:
    # Отведение aVL (индекс 4, низковольтное от конечностей) и V2 (индекс 7, высоковольтное грудное)
    lead_limb = clean_data[:, 4]
    lead_chest = clean_data[:, 7]

    ptp_limb = np.ptp(lead_limb)
    ptp_chest = np.ptp(lead_chest)

    # Поотведенческая Z-нормализация
    norm_limb = (lead_limb - np.mean(lead_limb)) / (np.std(lead_limb) + 1e-8)
    norm_chest = (lead_chest - np.mean(lead_chest)) / (np.std(lead_chest) + 1e-8)

    fig, axs = plt.subplots(2, 2, figsize=(15, 8.5))
    fig.suptitle(
        f"Метод 3: Поотведенческая Z-нормализация и выравнивание масштаба ({source_label})",
        fontsize=13, fontweight='bold', y=0.98
    )

    axs[0, 0].plot(t, lead_chest, color='darkorange', label=f'Отведение V2 (Размах: {ptp_chest:.2f} мВ)', alpha=0.85)
    axs[0, 0].plot(t, lead_limb, color='teal', label=f'Отведение aVL (Размах: {ptp_limb:.2f} мВ)', alpha=0.9)
    axs[0, 0].set_title("А) Реальные сигналы ДО нормализации (7.8-кратный дисбаланс амплитуд)", fontsize=11)
    axs[0, 0].set_xlabel("Время (секунды)")
    axs[0, 0].set_ylabel("Амплитуда (мВ)", labelpad=8)
    axs[0, 0].legend(loc='upper right')
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)

    axs[0, 1].plot(t, norm_chest, color='darkorange', label=f'V2 Z-score (μ={norm_chest.mean():.2f}, σ={norm_chest.std():.2f})', alpha=0.85)
    axs[0, 1].plot(t, norm_limb, color='teal', label=f'aVL Z-score (μ={norm_limb.mean():.2f}, σ={norm_limb.std():.2f})', alpha=0.75)
    axs[0, 1].set_title("Б) После Z-нормализации (Единый безразмерный масштаб признаков)", fontsize=11)
    axs[0, 1].set_xlabel("Время (секунды)")
    axs[0, 1].set_ylabel("Стандартизированная амплитуда (Z-score)", labelpad=8)
    axs[0, 1].legend(loc='upper right')
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)

    axs[1, 0].hist(lead_chest, bins=60, color='darkorange', alpha=0.55, density=True, label=f'V2 (σ = {np.std(lead_chest):.3f} мВ)')
    axs[1, 0].hist(lead_limb, bins=60, color='teal', alpha=0.55, density=True, label=f'aVL (σ = {np.std(lead_limb):.3f} мВ)')
    axs[1, 0].set_title("В) Плотность распределения амплитуд ДО нормализации", fontsize=11)
    axs[1, 0].set_xlabel("Амплитуда (мВ)")
    axs[1, 0].set_ylabel("Плотность вероятности")
    axs[1, 0].legend(loc='upper right')
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)

    axs[1, 1].hist(norm_chest, bins=60, color='darkorange', alpha=0.55, density=True, label='V2 ~ N(0, 1)')
    axs[1, 1].hist(norm_limb, bins=60, color='teal', alpha=0.55, density=True, label='aVL ~ N(0, 1)')
    axs[1, 1].set_title("Г) Плотность распределения ПОСЛЕ (Приведение к N(0, 1))", fontsize=11)
    axs[1, 1].set_xlabel("Значение Z-score")
    axs[1, 1].set_ylabel("Плотность вероятности")
    axs[1, 1].legend(loc='upper right')
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = OUT_DIR / "fig3_z_normalization.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[УСПЕХ] Обновленный График 3 сохранен: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()