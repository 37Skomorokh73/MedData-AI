import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from demo_1_filtering_fft import load_real_ecg, OUT_DIR
from preprocess_dataset import apply_bandpass_filter, apply_polyphase_resampling


def main():
    data, fs_original, source_label, lead_names = load_real_ecg(record_index=0)
    lead_idx = 1  # Стандартное отведение II
    clean_signal = apply_bandpass_filter(data[:, [lead_idx]], fs=fs_original)[:, 0]
    t_original = np.arange(len(clean_signal)) / fs_original

    fs_target = 250
    resampled_signal = apply_polyphase_resampling(clean_signal[:, None], orig_fs=int(fs_original), target_fs=fs_target)[:, 0]
    t_resampled = np.arange(len(resampled_signal)) / fs_target

    # Детализация 1 секунды вокруг комплекса QRS (1.5 - 2.5 с)
    mask_orig = (t_original >= 1.5) & (t_original <= 2.5)
    mask_resamp = (t_resampled >= 1.5) & (t_resampled <= 2.5)

    peak_orig = np.max(clean_signal[mask_orig])
    peak_resamp = np.max(resampled_signal[mask_resamp])
    peak_diff = abs(peak_orig - peak_resamp)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_title(
        f"Метод 2: Полифазный ресэмплинг комплекса QRS ({source_label}, Отведение {lead_names[lead_idx]})",
        fontsize=13, fontweight='bold', pad=12
    )

    ax.plot(
        t_original[mask_orig], clean_signal[mask_orig],
        label=f'Исходный сигнал ({int(fs_original)} Гц, 500 отсчетов/с)',
        color='silver', linewidth=3.8
    )
    ax.plot(
        t_resampled[mask_resamp], resampled_signal[mask_resamp],
        marker='o', linestyle='-', color='crimson', markersize=4.5, linewidth=1.3,
        label=f'Полифазный ресэмплинг ({int(fs_target)} Гц, 250 отсчетов/с)'
    )

    ax.set_xlabel("Время (секунды)", fontsize=11)
    ax.set_ylabel("Амплитуда (мВ)", fontsize=11)
    ax.legend(loc='upper right', fontsize=10.5)
    ax.grid(True, linestyle='--', alpha=0.6)

    info_text = (
        f"• Сжатие размерности: {int(fs_original)} Гц → {int(fs_target)} Гц (в 2 раза)\n"
        f"• Пик R-зубца (500 Гц): {peak_orig:.4f} мВ\n"
        f"• Пик R-зубца (250 Гц): {peak_resamp:.4f} мВ (Δ = {peak_diff:.4f} мВ)\n"
        f"• Морфология P-QRS-T и крутизна фронтов сохранены"
    )
    ax.text(
        0.03, 0.93, info_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.92, edgecolor='gray')
    )

    plt.tight_layout()
    save_path = OUT_DIR / "fig2_polyphase_resampling.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[УСПЕХ] Обновленный График 2 сохранен: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()