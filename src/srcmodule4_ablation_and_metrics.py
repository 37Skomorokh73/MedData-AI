import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    f1_score
)

warnings.filterwarnings("ignore")

# 1. Настройка путей (как в module3)
ROOT_DIR = Path(__file__).resolve().parent.parent
CLINICAL_CSV = ROOT_DIR / "ptbxl_clinical_and_voltage.csv"
EMBED_CSV = ROOT_DIR / "echonext_v2_embeddings.csv"
NPZ_PATH = ROOT_DIR / "data_samples" / "late_fusion_splits.npz"
FIGURES_DIR = ROOT_DIR / "figures"
MODELS_DIR = ROOT_DIR / "models"

FIGURES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

TARGETS = ["is_NORM", "is_STTC", "is_CD", "is_MI", "is_HYP"]
TARGET_TO_CNN_PROB = {
    "is_NORM": "cnn_prob_NORM",
    "is_STTC": "cnn_prob_STTC",
    "is_CD":   "cnn_prob_CD",
    "is_MI":   "cnn_prob_MI",
    "is_HYP":  "cnn_prob_HYP",
}


# 2. Загрузка данных (из .npz или напрямую из двух CSV)
def load_splits():
    if NPZ_PATH.exists():
        print(f"[1/4] Загрузка сплитов из архива: {NPZ_PATH.name}...")
        data = np.load(NPZ_PATH, allow_pickle=True)
        feat_cols = list(data["feature_names"])
        targ_cols = list(data["target_names"])
        X_train = pd.DataFrame(data["X_train"], columns=feat_cols)
        y_train = pd.DataFrame(data["y_train"], columns=targ_cols)
        X_val = pd.DataFrame(data["X_val"], columns=feat_cols)
        y_val = pd.DataFrame(data["y_val"], columns=targ_cols)
        X_test = pd.DataFrame(data["X_test"], columns=feat_cols)
        y_test = pd.DataFrame(data["y_test"], columns=targ_cols)
    else:
        print("[1/4] Загрузка и слияние мастер-таблиц CSV...")
        df_clin = pd.read_csv(CLINICAL_CSV)
        df_emb = pd.read_csv(EMBED_CSV)
        df = pd.merge(df_clin, df_emb, on="ecg_id", how="inner")

        clinical_cols = ["age", "sex", "bmi"]
        voltage_cols = [c for c in df_clin.columns if c.startswith("lead_") and (c.endswith("_ptp") or c.endswith("_std"))]
        cnn_prob_cols = [c for c in df_emb.columns if c.startswith("cnn_prob_")]
        emb_cols = [c for c in df_emb.columns if c.startswith("emb_")]
        feat_cols = clinical_cols + voltage_cols + cnn_prob_cols + emb_cols

        train_mask = df["strat_fold"].isin(range(1, 9))
        val_mask = df["strat_fold"] == 9
        test_mask = df["strat_fold"] == 10

        X_train, y_train = df.loc[train_mask, feat_cols], df.loc[train_mask, TARGETS]
        X_val, y_val = df.loc[val_mask, feat_cols], df.loc[val_mask, TARGETS]
        X_test, y_test = df.loc[test_mask, feat_cols], df.loc[test_mask, TARGETS]

    clinical_cols = [c for c in X_train.columns if c in ("age", "sex", "bmi")]
    voltage_cols = [c for c in X_train.columns if c.startswith("lead_")]
    cnn_cols = [c for c in X_train.columns if c.startswith("cnn_prob_") or c.startswith("emb_")]

    groups = {
        "clinical": clinical_cols,
        "voltage": voltage_cols,
        "cnn": cnn_cols,
        "all": list(X_train.columns)
    }
    return X_train, y_train, X_val, y_val, X_test, y_test, groups


# 3. Функции для поиска порогов и расчета клинических метрик (Задача 1)
def find_optimal_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    # Критерий 1: Максимум F1 на валидации (Fold 9)
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_idx = int(np.argmax(f1_scores[:-1]))
    thr_f1 = float(pr_thresholds[best_f1_idx])

    # Критерий 2: Индекс Юдена (Sensitivity + Specificity - 1) на валидации (Fold 9)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_j_idx = int(np.argmax(j_scores))
    thr_youden = float(roc_thresholds[best_j_idx])

    return {
        "thr_f1": round(thr_f1, 4),
        "thr_youden": round(thr_youden, 4),
        "val_max_f1": round(float(f1_scores[best_f1_idx]), 4),
        "val_max_youden": round(float(j_scores[best_j_idx]), 4),
    }


def compute_clinical_row(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    ppv = tp / (tp + fp + 1e-8)
    npv = tn / (tn + fn + 1e-8)
    f1_opt = 2 * (ppv * sens) / (ppv + sens + 1e-8)
    f1_def = f1_score(y_true, (y_prob >= 0.5).astype(int))

    return {
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "Threshold": thr,
        "Sensitivity": sens,
        "Specificity": spec,
        "PPV (Precision)": ppv,
        "NPV": npv,
        "F1 (Opt)": f1_opt,
        "F1 (@0.5)": f1_def,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
    }


# 4. Обучение конфигурации XGBoost для Ablation Study (Задача 2)
def evaluate_subset_xgb(X_train, y_train, X_val, y_val, X_test, cols):
    val_probs, test_probs = {}, {}

    for target in TARGETS:
        yt_tr = y_train[target].values
        yt_va = y_val[target].values
        pos_weight = (len(yt_tr) - yt_tr.sum()) / max(yt_tr.sum(), 1)

        model = XGBClassifier(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=1.0,
            reg_lambda=2.0,
            scale_pos_weight=pos_weight,
            eval_metric="auc",
            early_stopping_rounds=40,
            random_state=42,
            n_jobs=-1,
            tree_method="hist"
        )
        model.fit(
            X_train[cols], yt_tr,
            eval_set=[(X_val[cols], yt_va)],
            verbose=False
        )
        val_probs[target] = model.predict_proba(X_val[cols])[:, 1]
        test_probs[target] = model.predict_proba(X_test[cols])[:, 1]

    return val_probs, test_probs


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test, groups = load_splits()

    # -----------------------------------------------------------------
    # ЗАДАЧА 2: ABLATION STUDY (Вклад каждой группы признаков)
    # -----------------------------------------------------------------
    print("\n[2/4] Запуск Ablation Study (6 конфигураций на тесте Fold 10)...")
    ablation_configs = [
        ("1. Baseline: EchoNext-Mini V2 (Pure CNN)", None),
        ("2. Tabular Only (Clinical + Voltage)", groups["clinical"] + groups["voltage"]),
        ("3. XGB: CNN Only (Probs + Embs)", groups["cnn"]),
        ("4. XGB: CNN + Clinical (Age, Sex, BMI)", groups["cnn"] + groups["clinical"]),
        ("5. XGB: CNN + Voltage (Amplitude)", groups["cnn"] + groups["voltage"]),
        ("6. XGB: Full Late Fusion (All 160)", groups["all"]),
    ]

    ablation_rows = []
    full_val_probs, full_test_probs = {}, {}

    for name, cols in ablation_configs:
        print(f"      -> Расчет: {name}")
        if cols is None:
            val_p = {t: X_val[TARGET_TO_CNN_PROB[t]].values for t in TARGETS}
            test_p = {t: X_test[TARGET_TO_CNN_PROB[t]].values for t in TARGETS}
            n_feats = 5
        else:
            val_p, test_p = evaluate_subset_xgb(X_train, y_train, X_val, y_val, X_test, cols)
            n_feats = len(cols)

        if name.startswith("6."):
            full_val_probs, full_test_probs = val_p, test_p

        row = {"Configuration": name, "Features": n_feats}
        aucs, pr_aucs, f1s = [], [], []
        for t in TARGETS:
            yt = y_test[t].values
            pt = test_p[t]
            yv = y_val[t].values
            pv = val_p[t]

            auc_v = roc_auc_score(yt, pt)
            pr_v = average_precision_score(yt, pt)
            thr = find_optimal_thresholds(yv, pv)["thr_f1"]
            f1_v = f1_score(yt, (pt >= thr).astype(int))

            row[t] = auc_v
            aucs.append(auc_v)
            pr_aucs.append(pr_v)
            f1s.append(f1_v)

        row["Macro ROC-AUC"] = np.mean(aucs)
        row["Macro PR-AUC"] = np.mean(pr_aucs)
        row["Macro F1"] = np.mean(f1s)
        ablation_rows.append(row)

    df_ablation = pd.DataFrame(ablation_rows).set_index("Configuration")
    base_macro = df_ablation.loc["1. Baseline: EchoNext-Mini V2 (Pure CNN)", "Macro ROC-AUC"]
    df_ablation["Δ Macro AUC"] = df_ablation["Macro ROC-AUC"] - base_macro

    print("\n" + "=" * 100)
    print("ТАБЛИЦА 1: ABLATION STUDY НА ИЗОЛИРОВАННОМ ТЕСТЕ (FOLD 10, N = 2 198)")
    print("=" * 100)
    print(df_ablation.round(4).to_string())
    df_ablation.round(4).to_csv(FIGURES_DIR / "ablation_study_fold10.csv")

    # -----------------------------------------------------------------
    # ЗАДАЧА 1: КЛИНИЧЕСКИЕ МЕТРИКИ И КАЛИБРОВКА ПОРОГОВ
    # -----------------------------------------------------------------
    print("\n[3/4] Расчет клинических метрик и оптимальных порогов (Fold 9 -> Fold 10)...")
    thresholds_dict = {}
    rows_f1, rows_youden = [], []

    for t in TARGETS:
        yv = y_val[t].values
        yt = y_test[t].values
        pv = full_val_probs[t]
        pt = full_test_probs[t]

        opt = find_optimal_thresholds(yv, pv)
        thresholds_dict[t] = opt

        m_f1 = compute_clinical_row(yt, pt, opt["thr_f1"])
        m_f1["Class"] = t
        rows_f1.append(m_f1)

        m_j = compute_clinical_row(yt, pt, opt["thr_youden"])
        m_j["Class"] = t
        rows_youden.append(m_j)

    df_clin_f1 = pd.DataFrame(rows_f1).set_index("Class")
    df_clin_youden = pd.DataFrame(rows_youden).set_index("Class")

    avg_cols = ["ROC-AUC", "PR-AUC", "Sensitivity", "Specificity", "PPV (Precision)", "NPV", "F1 (Opt)", "F1 (@0.5)"]
    for df_c in (df_clin_f1, df_clin_youden):
        df_c.loc["Macro AVG", avg_cols] = df_c[avg_cols].mean()

    print("\n" + "=" * 100)
    print("ТАБЛИЦА 2: КЛИНИЧЕСКИЕ МЕТРИКИ LATE FUSION (ПОРОГ ПО MAX F1 НА ВАЛИДАЦИИ)")
    print("=" * 100)
    print(df_clin_f1.round(4).to_string())
    df_clin_f1.round(4).to_csv(FIGURES_DIR / "clinical_metrics_f1_fold10.csv")

    print("\n" + "=" * 100)
    print("ТАБЛИЦА 3: КЛИНИЧЕСКИЕ МЕТРИКИ LATE FUSION (ПОРОГ ПО ИНДЕКСУ ЮДЕНА НА ВАЛИДАЦИИ)")
    print("=" * 100)
    print(df_clin_youden.round(4).to_string())
    df_clin_youden.round(4).to_csv(FIGURES_DIR / "clinical_metrics_youden_fold10.csv")

    thr_path = MODELS_DIR / "optimal_thresholds.json"
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(thresholds_dict, f, indent=4, ensure_ascii=False)
    print(f"\n      Оптимальные пороги сохранены в: {thr_path}")

    # -----------------------------------------------------------------
    # 4. ГЕНЕРАЦИЯ ГРАФИКОВ fig5 И fig6
    # -----------------------------------------------------------------
    print("\n[4/4] Построение графиков fig5_ablation_study.png и fig6_pr_curves_fold10.png...")
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    short_names = [
        "1. Pure CNN\n(Baseline)",
        "2. Tabular Only\n(Clin + Volt)",
        "3. XGB: CNN\nOnly",
        "4. XGB: CNN\n+ Clinical",
        "5. XGB: CNN\n+ Voltage",
        "6. Full Late\nFusion"
    ]
    x = np.arange(len(short_names))
    w = 0.36

    m_vals = df_ablation["Macro ROC-AUC"].values
    h_vals = df_ablation["is_HYP"].values

    b1 = ax.bar(x - w/2, m_vals, w, label="Macro ROC-AUC (Все 5 классов)", color="#5b9bd5")
    b2 = ax.bar(x + w/2, h_vals, w, label="is_HYP ROC-AUC (Гипертрофия)", color="#ff9933")

    ax.set_ylabel("ROC-AUC (Тест, Фолд 10)", fontsize=11)
    ax.set_title("Ablation Study: Вклад модальностей в ансамбле MedData AI (Фолд 10, N = 2 198)", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9.5)
    ax.set_ylim(min(min(m_vals), min(h_vals)) - 0.04, 0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10)

    for rect in b1:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for rect in b2:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_ablation_study.png", dpi=300)
    plt.close()

    # График PR-кривых
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)
    for t in TARGETS:
        yt = y_test[t].values
        pt = full_test_probs[t]
        prec, rec, _ = precision_recall_curve(yt, pt)
        ap = average_precision_score(yt, pt)
        ax.plot(rec, prec, lw=2, label=f"{t} (PR-AUC = {ap:.4f})")

    ax.set_xlabel("Чувствительность (Recall / Sensitivity)", fontsize=11)
    ax.set_ylabel("Прогностическая ценность (Precision / PPV)", fontsize=11)
    ax.set_title("Precision-Recall кривые ансамбля MedData AI (Тест, Фолд 10)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_pr_curves_fold10.png", dpi=300)
    plt.close()
    print(f"      Графики сохранены в папку: {FIGURES_DIR}")