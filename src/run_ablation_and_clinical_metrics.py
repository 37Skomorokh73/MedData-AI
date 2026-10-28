import json
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    f1_score
)

# =====================================================================
# 1. КОНФИГУРАЦИЯ И НАЗВАНИЯ КЛАССОВ
# =====================================================================
TARGETS = ["is_NORM", "is_MI", "is_STTC", "is_CD", "is_HYP"]
CNN_PROB_COLS = ["cnn_prob_NORM", "cnn_prob_MI", "cnn_prob_STTC", "cnn_prob_CD", "cnn_prob_HYP"]

MODELS_DIR = Path("C:\\Users\\andre\\OneDrive\Рабочий стол\Исследовательский проект\для github\\models")       # Папка, где лежат xgb_late_fusion_*.json
OUTPUT_DIR = Path("C:\\Users\\andre\\OneDrive\Рабочий стол\Исследовательский проект\для github\\figures")      # Папка для таблиц, графиков и порогов
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# 2. ФУНКЦИИ ДЛЯ КЛИНИЧЕСКИХ МЕТРИК И ПОИСКА ПОРОГОВ (ЗАДАЧА 1)
# =====================================================================
def find_optimal_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    """Находит оптимальные пороги на валидационной выборке (Fold 9)."""
    # 1. Порог по максимуму F1-меры
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-8)
    best_f1_idx = np.argmax(f1_scores[:-1])
    thr_f1 = float(pr_thresholds[best_f1_idx])

    # 2. Порог по индексу Юдена (Sensitivity + Specificity - 1)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_j_idx = np.argmax(j_scores)
    thr_youden = float(roc_thresholds[best_j_idx])

    return {
        "thr_f1": round(thr_f1, 4),
        "thr_youden": round(thr_youden, 4),
        "val_max_f1": round(float(f1_scores[best_f1_idx]), 4),
        "val_max_youden": round(float(j_scores[best_j_idx]), 4),
    }


def compute_binary_clinical_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    """Считает полный набор клинических метрик при фиксированном пороге."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    sensitivity = tp / (tp + fn + 1e-8)  # Recall / TPR
    specificity = tn / (tn + fp + 1e-8)  # TNR
    ppv = tp / (tp + fp + 1e-8)          # Precision
    npv = tn / (tn + fn + 1e-8)          # Negative Predictive Value
    f1 = 2 * (ppv * sensitivity) / (ppv + sensitivity + 1e-8)

    return {
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Threshold": threshold,
        "Sensitivity (Recall)": sensitivity,
        "Specificity": specificity,
        "PPV (Precision)": ppv,
        "NPV": npv,
        "F1-score": f1,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
    }


def evaluate_full_clinical_table(
    y_val: pd.DataFrame, val_probs: dict,
    y_test: pd.DataFrame, test_probs: dict,
    mode: str = "thr_f1"
) -> tuple[pd.DataFrame, dict]:
    """Строит итоговую таблицу клинических метрик на Fold 10 с порогами из Fold 9."""
    thresholds_dict = {}
    rows = []

    for target in TARGETS:
        yv = y_val[target].values
        yt = y_test[target].values
        pv = val_probs[target]
        pt = test_probs[target]

        opt_thrs = find_optimal_thresholds(yv, pv)
        thresholds_dict[target] = opt_thrs
        chosen_thr = opt_thrs[mode]

        metrics = compute_binary_clinical_metrics(yt, pt, threshold=chosen_thr)
        metrics["Class"] = target
        # Дополнительно считаем F1 при дефолтном пороге 0.5 для сравнения
        f1_default = f1_score(yt, (pt >= 0.5).astype(int))
        metrics["F1 (@0.5)"] = f1_default
        rows.append(metrics)

    df_metrics = pd.DataFrame(rows).set_index("Class")

    # Добавляем строку Macro Average
    macro_cols = ["ROC-AUC", "PR-AUC", "Sensitivity (Recall)", "Specificity", "PPV (Precision)", "NPV", "F1-score", "F1 (@0.5)"]
    macro_row = df_metrics[macro_cols].mean()
    df_metrics.loc["Macro AVG", macro_cols] = macro_row
    df_metrics.loc["Macro AVG", "Threshold"] = np.nan

    return df_metrics, thresholds_dict


# =====================================================================
# 3. АВТОМАТИЧЕСКОЕ РАЗБИЕНИЕ ПРИЗНАКОВ И ABLATION STUDY (ЗАДАЧА 2)
# =====================================================================
def split_feature_groups(feature_columns: list[str]) -> dict[str, list[str]]:
    """Автоматически делит признаки на модальности по префиксам."""
    cnn_probs = [c for c in feature_columns if c.startswith("cnn_prob_")]
    cnn_embs = [c for c in feature_columns if c.startswith("emb_")]
    voltage = [c for c in feature_columns if c.startswith("lead_")]
    clinical = [c for c in feature_columns if c not in set(cnn_probs + cnn_embs + voltage)]

    print(f"[Features] CNN Probs: {len(cnn_probs)} | Embeddings: {len(cnn_embs)} | "
          f"Voltage: {len(voltage)} | Clinical: {len(clinical)} ({clinical})")

    return {
        "cnn_all": cnn_probs + cnn_embs,
        "voltage": voltage,
        "clinical": clinical,
        "all": feature_columns
    }


def train_or_eval_xgb_subset(
    X_train: pd.DataFrame, y_train: pd.DataFrame,
    X_val: pd.DataFrame, y_val: pd.DataFrame,
    X_test: pd.DataFrame, y_test: pd.DataFrame,
    cols: list[str],
    load_saved_dir: Path | None = None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Обучает 5 моделей XGBoost на выбранном подмножестве колонок (или загружает готовые JSON)."""
    val_preds, test_preds = {}, {}

    for target in TARGETS:
        saved_model_path = load_saved_dir / f"xgb_late_fusion_{target}.json" if load_saved_dir else None

        if saved_model_path and saved_model_path.exists():
            model = xgb.XGBClassifier()
            model.load_model(saved_model_path)
        else:
            pos_count = y_train[target].sum()
            neg_count = len(y_train) - pos_count
            scale_pos_weight = float(neg_count / max(pos_count, 1))

            model = xgb.XGBClassifier(
                n_estimators=600,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                reg_alpha=0.5,
                reg_lambda=2.0,
                scale_pos_weight=scale_pos_weight,
                eval_metric="auc",
                early_stopping_rounds=40,
                random_state=42,
                n_jobs=-1
            )
            model.fit(
                X_train[cols], y_train[target],
                eval_set=[(X_val[cols], y_val[target])],
                verbose=False
            )

        val_preds[target] = model.predict_proba(X_val[cols])[:, 1]
        test_preds[target] = model.predict_proba(X_test[cols])[:, 1]

    return val_preds, test_preds


def run_ablation_experiments(
    X_train: pd.DataFrame, y_train: pd.DataFrame,
    X_val: pd.DataFrame, y_val: pd.DataFrame,
    X_test: pd.DataFrame, y_test: pd.DataFrame
) -> tuple[pd.DataFrame, dict, dict]:
    """Прогоняет все конфигурации Ablation Study и возвращает сводную таблицу + предсказания Full модели."""
    groups = split_feature_groups(list(X_train.columns))

    configs = [
        ("1. Baseline: EchoNext-Mini V2 (Pure CNN)", None, False),
        ("2. Tabular Only (Clinical + Voltage)", groups["clinical"] + groups["voltage"], False),
        ("3. XGB: CNN Only (Probs + Embs)", groups["cnn_all"], False),
        ("4. XGB: CNN + Clinical (Age, Sex)", groups["cnn_all"] + groups["clinical"], False),
        ("5. XGB: CNN + Voltage (Amplitude)", groups["cnn_all"] + groups["voltage"], False),
        ("6. XGB: Full Late Fusion (All)", groups["all"], True),
    ]

    ablation_rows = []
    full_val_probs, full_test_probs = {}, {}

    for name, cols, use_saved in configs:
        print(f"Running {name}...")
        if cols is None:
            # Чистая нейросеть: берём вероятности напрямую из колонок cnn_prob_*
            val_p = {t: X_val[prob_col].values for t, prob_col in zip(TARGETS, CNN_PROB_COLS)}
            test_p = {t: X_test[prob_col].values for t, prob_col in zip(TARGETS, CNN_PROB_COLS)}
            n_feats = 5
        else:
            val_p, test_p = train_or_eval_xgb_subset(
                X_train, y_train, X_val, y_val, X_test, y_test,
                cols=cols,
                load_saved_dir=MODELS_DIR if use_saved else None
            )
            n_feats = len(cols)

        if use_saved:
            full_val_probs, full_test_probs = val_p, test_p

        row = {"Configuration": name, "Features": n_feats}
        aucs, pr_aucs, f1s = [], [], []

        for t in TARGETS:
            yt = y_test[t].values
            pt = test_p[t]
            yv = y_val[t].values
            pv = val_p[t]

            auc_val = roc_auc_score(yt, pt)
            pr_val = average_precision_score(yt, pt)
            thr = find_optimal_thresholds(yv, pv)["thr_f1"]
            f1_val = f1_score(yt, (pt >= thr).astype(int))

            row[t] = auc_val
            aucs.append(auc_val)
            pr_aucs.append(pr_val)
            f1s.append(f1_val)

        row["Macro ROC-AUC"] = np.mean(aucs)
        row["Macro PR-AUC"] = np.mean(pr_aucs)
        row["Macro F1"] = np.mean(f1s)
        ablation_rows.append(row)

    df_ablation = pd.DataFrame(ablation_rows).set_index("Configuration")
    baseline_auc = df_ablation.loc["1. Baseline: EchoNext-Mini V2 (Pure CNN)", "Macro ROC-AUC"]
    df_ablation["Δ Macro AUC"] = df_ablation["Macro ROC-AUC"] - baseline_auc

    return df_ablation, full_val_probs, full_test_probs


# =====================================================================
# 4. ВИЗУАЛИЗАЦИЯ (ГРАФИКИ ДЛЯ ДИПЛОМА / РЕПОЗИТОРИЯ)
# =====================================================================
def plot_ablation_and_pr(df_ablation: pd.DataFrame, y_test: pd.DataFrame, full_test_probs: dict):
    # График 1: Ablation Study (Macro AUC vs is_HYP AUC)
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    short_names = [
        "1. Pure CNN\n(Baseline)",
        "2. Tabular Only\n(Clin + Volt)",
        "3. XGB: CNN\nOnly",
        "4. XGB: CNN\n+ Clinical",
        "5. XGB: CNN\n+ Voltage",
        "6. Full Late\nFusion"
    ]
    x = np.arange(len(short_names))
    width = 0.36

    macro_vals = df_ablation["Macro ROC-AUC"].values
    hyp_vals = df_ablation["is_HYP"].values

    rects1 = ax.bar(x - width/2, macro_vals, width, label="Macro ROC-AUC (All 5 Classes)", color="#2b5c8f")
    rects2 = ax.bar(x + width/2, hyp_vals, width, label="is_HYP ROC-AUC (Hypertrophy)", color="#d95f02")

    ax.set_ylabel("ROC-AUC (Test Fold 10)", fontsize=11, fontweight="bold")
    ax.set_title("Ablation Study: Вклад модальностей в качество классификации (PTB-XL Fold 10)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=9.5)
    ax.set_ylim(0.70, 0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=10)

    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.4f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig5_ablation_study.png")
    plt.close()

    # График 2: Precision-Recall кривые на Fold 10 для Full Late Fusion
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    for t in TARGETS:
        yt = y_test[t].values
        pt = full_test_probs[t]
        prec, rec, _ = precision_recall_curve(yt, pt)
        ap = average_precision_score(yt, pt)
        ax.plot(rec, prec, lw=2, label=f"{t} (PR-AUC = {ap:.4f})")

    ax.set_xlabel("Recall (Sensitivity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Precision (PPV)", fontsize=11, fontweight="bold")
    ax.set_title("Precision-Recall Curves — Hybrid Late Fusion (Fold 10)", fontsize=12, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig6_pr_curves_fold10.png")
    plt.close()