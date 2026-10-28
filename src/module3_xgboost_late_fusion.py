import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")

# 1. Настройка путей
ROOT_DIR = Path(__file__).resolve().parent.parent
CLINICAL_CSV = ROOT_DIR / "ptbxl_clinical_and_voltage.csv"
EMBED_CSV = ROOT_DIR / "echonext_v2_embeddings.csv"
DATA_SAMPLES_DIR = ROOT_DIR / "data_samples"
FIGURES_DIR = ROOT_DIR / "figures"
MODELS_DIR = ROOT_DIR / "models"

DATA_SAMPLES_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# 2. Загрузка и слияние таблиц по ecg_id
print("[1/5] Загрузка и объединение мастер-таблиц...")
df_clin = pd.read_csv(CLINICAL_CSV)
df_emb = pd.read_csv(EMBED_CSV)
df = pd.merge(df_clin, df_emb, on="ecg_id", how="inner")
print(f"      Успешно объединено записей: {len(df)} (ожидалось: 21 799)")

# 3. Базовые списки колонок и маски официальных стратифицированных фолдов PTB-XL
TARGETS = ["is_NORM", "is_STTC", "is_CD", "is_MI", "is_HYP"]

CLINICAL_COLS = ["age", "sex", "bmi"]
VOLTAGE_COLS = [
    c for c in df_clin.columns
    if c.startswith("lead_") and (c.endswith("_ptp") or c.endswith("_std"))
]
CNN_PROB_COLS = [c for c in df_emb.columns if c.startswith("cnn_prob_")]
EMB_COLS = [c for c in df_emb.columns if c.startswith("emb_")]

train_mask = df["strat_fold"].isin(range(1, 9)).values
val_mask = (df["strat_fold"] == 9).values
test_mask = (df["strat_fold"] == 10).values


# 4. Доменный кардиологический Feature Engineering и сжатие латентного пространства
print("[2/5] Расчет кардиологических индексов, калибровка логитов и сжатие эмбеддингов...")

eps = 1e-5

# 4.1. Электрокардиографические вольтажно-клинические индексы
# Отведения в PTB-XL: 1=I, 2=II, 3=III, 4=aVR, 5=aVL, 6=aVF, 7=V1, 8=V2, 9=V3, 10=V4, 11=V5, 12=V6
v1_ptp = df["lead_7_ptp"].values
v3_ptp = df["lead_9_ptp"].values
v5_ptp = df["lead_11_ptp"].values
v6_ptp = df["lead_12_ptp"].values
avl_ptp = df["lead_5_ptp"].values

limb_ptp_cols = [f"lead_{i}_ptp" for i in range(1, 7)]
prec_ptp_cols = [f"lead_{i}_ptp" for i in range(7, 13)]
all_ptp_cols = [f"lead_{i}_ptp" for i in range(1, 13)]
all_std_cols = [f"lead_{i}_std" for i in range(1, 13)]

limb_mean_ptp = df[limb_ptp_cols].mean(axis=1).values
prec_mean_ptp = df[prec_ptp_cols].mean(axis=1).values
all_mean_ptp = df[all_ptp_cols].mean(axis=1).values
all_mean_std = df[all_std_cols].mean(axis=1).values

age_val = np.nan_to_num(df["age"].values, nan=60.0)
sex_val = np.nan_to_num(df["sex"].values, nan=0.5)
bmi_val = np.nan_to_num(df["bmi"].values, nan=26.0)

sokolow_lyon = v1_ptp + np.maximum(v5_ptp, v6_ptp)
cornell_raw = avl_ptp + v3_ptp
# Поправка Корнельского индекса на пол (у женщин диагностический порог ГЛЖ ниже на ~28%)
cornell_sex_adj = cornell_raw * (1.0 + 0.35 * (1.0 - sex_val))

lateral_ptp = (df["lead_1_ptp"] + df["lead_5_ptp"] + df["lead_11_ptp"] + df["lead_12_ptp"]).values / 4.0
inferior_ptp = (df["lead_2_ptp"] + df["lead_3_ptp"] + df["lead_6_ptp"]).values / 3.0
anterior_ptp = (df["lead_7_ptp"] + df["lead_8_ptp"] + df["lead_9_ptp"] + df["lead_10_ptp"]).values / 4.0

df_cardio = pd.DataFrame({
    "sokolow_lyon_proxy": sokolow_lyon,
    "cornell_proxy": cornell_raw,
    "cornell_sex_adj": cornell_sex_adj,
    "precordial_mean_ptp": prec_mean_ptp,
    "limb_mean_ptp": limb_mean_ptp,
    "prec_to_limb_ratio": prec_mean_ptp / (limb_mean_ptp + 1e-4),
    "lateral_to_inferior_ratio": lateral_ptp / (inferior_ptp + 1e-4),
    "anterior_to_inferior_ratio": anterior_ptp / (inferior_ptp + 1e-4),
    "bmi_voltage_index": prec_mean_ptp * (bmi_val / 25.0),
    "age_voltage_ratio": sokolow_lyon / (age_val + 15.0),
    "ecg_crest_factor": all_mean_ptp / (all_mean_std + 1e-4),
    "v6_to_v5_ratio": v6_ptp / (v5_ptp + 1e-4),
}, index=df.index)
CARDIO_INDEX_COLS = list(df_cardio.columns)


# 4.2. Выравнивание распределения логитов CNN на трейне (Unsupervised Out-of-Sample Alignment)
# Устраняет переуверенность нейросети на фолдах 1-8 (где обучалась сама CNN) относительно фолда 9
probs_raw = np.clip(df[CNN_PROB_COLS].to_numpy(dtype=np.float64), eps, 1.0 - eps)
logits_raw = np.log(probs_raw / (1.0 - probs_raw))
logits_aligned = logits_raw.copy()
probs_aligned = probs_raw.copy()

for j in range(len(CNN_PROB_COLS)):
    z_tr = logits_raw[train_mask, j]
    z_va = logits_raw[val_mask, j]
    # Монотонный сдвиг и температурное сжатие логитов трейна под дисперсию валидации (без использования y_val)
    scale = np.std(z_va) / max(np.std(z_tr), eps)
    z_tr_cal = (z_tr - np.mean(z_tr)) * scale + np.mean(z_va)
    logits_aligned[train_mask, j] = z_tr_cal
    probs_aligned[train_mask, j] = 1.0 / (1.0 + np.exp(-z_tr_cal))

CNN_LOGIT_COLS = [c.replace("cnn_prob_", "cnn_logit_") for c in CNN_PROB_COLS]
df_logits = pd.DataFrame(logits_aligned, columns=CNN_LOGIT_COLS, index=df.index)

# Признаки энтропии и межкатегориального конфликта CNN
p_sum = np.sum(probs_aligned, axis=1, keepdims=True) + eps
p_norm = probs_aligned / p_sum
cnn_entropy = -np.sum(p_norm * np.log(np.clip(p_norm, eps, 1.0)), axis=1)
sorted_p = np.sort(probs_aligned, axis=1)
cnn_margin_top2 = sorted_p[:, -1] - sorted_p[:, -2]
cnn_max_prob = sorted_p[:, -1]

idx_mi = CNN_PROB_COLS.index("cnn_prob_MI")
idx_sttc = CNN_PROB_COLS.index("cnn_prob_STTC")
idx_cd = CNN_PROB_COLS.index("cnn_prob_CD")
idx_hyp = CNN_PROB_COLS.index("cnn_prob_HYP")
idx_norm = CNN_PROB_COLS.index("cnn_prob_NORM")

df_cnn_meta = pd.DataFrame({
    "cnn_entropy": cnn_entropy,
    "cnn_margin_top2": cnn_margin_top2,
    "cnn_max_prob": cnn_max_prob,
    "cnn_diff_MI_STTC": logits_aligned[:, idx_mi] - logits_aligned[:, idx_sttc],
    "cnn_diff_MI_CD": logits_aligned[:, idx_mi] - logits_aligned[:, idx_cd],
    "cnn_diff_HYP_STTC": logits_aligned[:, idx_hyp] - logits_aligned[:, idx_sttc],
    "cnn_pathology_max_logit": np.max(
        logits_aligned[:, [idx_mi, idx_sttc, idx_cd, idx_hyp]], axis=1
    ) - logits_aligned[:, idx_norm],
}, index=df.index)
CNN_META_COLS = list(df_cnn_meta.columns)


# 4.3. Сжатие 128 эмбеддингов: Прототипы классов + Ортогональные главные компоненты (PCA)
# Все трансформации обучаются СТРОГО на train_mask (фолды 1-8)
emb_matrix = df[EMB_COLS].to_numpy(dtype=np.float64)
emb_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + eps
emb_unit = emb_matrix / emb_norms

proto_features = {}
for target in TARGETS:
    y_tr_cls = df.loc[train_mask, target].values
    pos_proto = emb_unit[train_mask][y_tr_cls == 1].mean(axis=0)
    pos_proto /= (np.linalg.norm(pos_proto) + eps)
    neg_proto = emb_unit[train_mask][y_tr_cls == 0].mean(axis=0)
    neg_proto /= (np.linalg.norm(neg_proto) + eps)

    cos_pos = emb_unit @ pos_proto
    cos_neg = emb_unit @ neg_proto
    short_name = target.replace("is_", "")
    proto_features[f"proto_cos_{short_name}"] = cos_pos
    proto_features[f"proto_margin_{short_name}"] = cos_pos - cos_neg

df_proto = pd.DataFrame(proto_features, index=df.index)
PROTO_COLS = list(df_proto.columns)

scaler_emb = StandardScaler()
emb_train_scaled = scaler_emb.fit_transform(emb_matrix[train_mask])
emb_all_scaled = scaler_emb.transform(emb_matrix)

pca = PCA(n_components=8, whiten=True, random_state=42)
pca.fit(emb_train_scaled)
emb_pca_all = pca.transform(emb_all_scaled)
PCA_COLS = [f"emb_pca_{i}" for i in range(8)]
df_pca = pd.DataFrame(emb_pca_all, columns=PCA_COLS, index=df.index)


# 4.4. Сборка единой матрицы признаков
# ВАЖНО: первые 32 признака (CLINICAL_COLS + VOLTAGE_COLS + CNN_PROB_COLS) остаются
# на своих точных позициях [0..31] для полной совместимости со скриптом оценки!
df_base_probs = pd.DataFrame(probs_aligned, columns=CNN_PROB_COLS, index=df.index)
# На Val (фолд 9) и Test (фолд 10) probs_aligned тождественно равны исходным cnn_prob_*
df_full = pd.concat(
    [
        df[CLINICAL_COLS + VOLTAGE_COLS],
        df_base_probs,
        df_cardio,
        df_logits,
        df_cnn_meta,
        df_proto,
        df_pca,
    ],
    axis=1,
)

FEATURE_COLS = list(df_full.columns)
PHYSIOLOGY_SET = set(CLINICAL_COLS + VOLTAGE_COLS + CARDIO_INDEX_COLS)

print(
    f"      Сформировано признаков: {len(FEATURE_COLS)} "
    f"(Клиника: {len(CLINICAL_COLS)}, Вольтаж: {len(VOLTAGE_COLS)}, "
    f"Кардио-индексы: {len(CARDIO_INDEX_COLS)}, CNN Probs/Logits/Meta: "
    f"{len(CNN_PROB_COLS) + len(CNN_LOGIT_COLS) + len(CNN_META_COLS)}, "
    f"Прототипы + PCA: {len(PROTO_COLS) + len(PCA_COLS)})"
)

# 5. Разделение и сохранение сплитов в data_samples/late_fusion_splits.npz
X_train = df_full.loc[train_mask, FEATURE_COLS]
X_val = df_full.loc[val_mask, FEATURE_COLS]
X_test = df_full.loc[test_mask, FEATURE_COLS]

y_train_df = df.loc[train_mask, TARGETS]
y_val_df = df.loc[val_mask, TARGETS]
y_test_df = df.loc[test_mask, TARGETS]

save_path = DATA_SAMPLES_DIR / "late_fusion_splits.npz"
np.savez_compressed(
    save_path,
    X_train=X_train.to_numpy(dtype=np.float32),
    y_train=y_train_df.to_numpy(dtype=np.float32),
    X_val=X_val.to_numpy(dtype=np.float32),
    y_val=y_val_df.to_numpy(dtype=np.float32),
    X_test=X_test.to_numpy(dtype=np.float32),
    y_test=y_test_df.to_numpy(dtype=np.float32),
    feature_names=np.array(FEATURE_COLS, dtype=str),
    target_names=np.array(TARGETS, dtype=str),
)
print(f"      Сплиты успешно сохранены в: {save_path}")


# 6. Обучение ансамбля XGBoost с монотонными и групповыми ограничениями
print("\n[3/5] Обучение моделей XGBoost с валидационным отбором (Train: фолды 1-8, Val: фолд 9)...")

results = []
gain_accumulator = np.zeros(len(FEATURE_COLS), dtype=np.float64)
col_to_idx = {col: i for i, col in enumerate(FEATURE_COLS)}

# Сетка кандидатов регуляризации (отбирается строго по Фолд 9 без участия Фолд 10)
CANDIDATE_CONFIGS = [
    {"max_depth": 2, "learning_rate": 0.015, "reg_lambda": 15.0, "reg_alpha": 2.0, "colsample_bylevel": 0.85, "min_child_weight": 1.0},
    {"max_depth": 3, "learning_rate": 0.015, "reg_lambda": 12.0, "reg_alpha": 1.5, "colsample_bylevel": 0.80, "min_child_weight": 1.0},
    {"max_depth": 3, "learning_rate": 0.020, "reg_lambda": 8.0,  "reg_alpha": 1.0, "colsample_bylevel": 0.75, "min_child_weight": 2.0},
    {"max_depth": 4, "learning_rate": 0.015, "reg_lambda": 15.0, "reg_alpha": 2.0, "colsample_bylevel": 0.70, "min_child_weight": 2.0},
]

for target in TARGETS:
    short_cls = target.replace("is_", "")
    target_prob_col = f"cnn_prob_{short_cls}"
    target_logit_col = f"cnn_logit_{short_cls}"
    target_proto_col = f"proto_margin_{short_cls}"

    y_train = df.loc[train_mask, target].values
    y_val = df.loc[val_mask, target].values
    y_test = df.loc[test_mask, target].values

    # Честные базовые метрики чистой EchoNext-Mini V2 напрямую из cnn_prob_*
    cnn_val_probs = df.loc[val_mask, target_prob_col].values
    cnn_test_probs = df.loc[test_mask, target_prob_col].values

    cnn_val_auc = roc_auc_score(y_val, cnn_val_probs)
    cnn_test_auc = roc_auc_score(y_test, cnn_test_probs)
    cnn_test_prauc = average_precision_score(y_test, cnn_test_probs)

    # 1) Монотонные ограничения (теперь передаем элегантным словарем)
    mono_dict = {
        target_prob_col: 1,
        target_logit_col: 1
    }

    # 2) Ограничения взаимодействий (теперь передаем списки ИМЕН колонок, а не индексов)
    anchor_cols = [target_prob_col, target_logit_col]
    group_physio = anchor_cols + CLINICAL_COLS + CARDIO_INDEX_COLS + VOLTAGE_COLS
    group_conflict = anchor_cols + CNN_PROB_COLS + CNN_LOGIT_COLS + CNN_META_COLS + [target_proto_col]
    group_morph = anchor_cols + CLINICAL_COLS + CARDIO_INDEX_COLS + PROTO_COLS + PCA_COLS
    interaction_groups = [group_physio, group_conflict, group_morph]

    # 3) Приоритетные веса признаков при сэмплировании уровней дерева
    feat_weights = np.ones(len(FEATURE_COLS), dtype=np.float32)
    for c in CLINICAL_COLS + CARDIO_INDEX_COLS:
        feat_weights[col_to_idx[c]] = 3.0
    for c in CNN_META_COLS + PROTO_COLS:
        feat_weights[col_to_idx[c]] = 2.0
    feat_weights[col_to_idx[target_prob_col]] = 40.0
    feat_weights[col_to_idx[target_logit_col]] = 40.0

    best_model = None
    best_val_composite = -1.0
    best_cfg_desc = ""

    for cfg in CANDIDATE_CONFIGS:
        candidate = XGBClassifier(
            n_estimators=1200,
            max_depth=cfg["max_depth"],
            learning_rate=cfg["learning_rate"],
            subsample=0.85,
            colsample_bytree=1.0,
            colsample_bylevel=cfg["colsample_bylevel"],
            min_child_weight=cfg["min_child_weight"],
            reg_alpha=cfg["reg_alpha"],
            reg_lambda=cfg["reg_lambda"],
            scale_pos_weight=1.0,  # Сохраняет калибровку вероятностей и максимизирует PR-AUC
            max_bin=1024,          # Высокое разрешение квантования в хвостах вероятностей CNN
            monotone_constraints=mono_dict,
            interaction_constraints=interaction_groups,
            eval_metric=["auc", "aucpr"],
            early_stopping_rounds=60,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )

        candidate.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            feature_weights=feat_weights,
            verbose=False,
        )

        val_preds = candidate.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_preds)
        val_prauc = average_precision_score(y_val, val_preds)
        composite = 0.65 * val_auc + 0.35 * val_prauc

        if composite > best_val_composite:
            best_val_composite = composite
            best_model = candidate
            best_cfg_desc = f"depth={cfg['max_depth']}, lr={cfg['learning_rate']}, l2={cfg['reg_lambda']}, trees={candidate.best_iteration + 1}"

    # Оценка лучшей по валидации модели на изолированном 10-м фолде
    train_preds = best_model.predict_proba(X_train)[:, 1]
    val_preds = best_model.predict_proba(X_val)[:, 1]
    test_preds = best_model.predict_proba(X_test)[:, 1]

    auc_train = roc_auc_score(y_train, train_preds)
    auc_val = roc_auc_score(y_val, val_preds)
    auc_test = roc_auc_score(y_test, test_preds)
    prauc_test = average_precision_score(y_test, test_preds)

    delta_auc = auc_test - cnn_test_auc
    delta_prauc = prauc_test - cnn_test_prauc

    print(
        f"      • {target:<7} | Выбрана конфигурация ({best_cfg_desc}) | "
        f"Val AUC: {cnn_val_auc:.4f} -> {auc_val:.4f}"
    )

    results.append({
        "target": target,
        "train_auc": auc_train,
        "val_auc": auc_val,
        "cnn_test_auc": cnn_test_auc,
        "xgb_test_auc": auc_test,
        "delta_auc": delta_auc,
        "cnn_test_prauc": cnn_test_prauc,
        "xgb_test_prauc": prauc_test,
        "delta_prauc": delta_prauc,
    })

    # Сбор важности признаков по Information Gain
    booster = best_model.get_booster()
    score_dict = booster.get_score(importance_type="gain")
    for idx, col in enumerate(FEATURE_COLS):
        gain_accumulator[idx] += score_dict.get(col, 0.0)

    # Сохранение модели
    best_model.save_model(MODELS_DIR / f"xgb_late_fusion_{target}.json")


# 7. Вывод сводной таблицы на изолированном тесте (Фолд 10)
print("\n[4/5] Итоговая таблица сравнения на изолированном тесте (Фолд 10, N = 2 198):\n")
print("| Код макрокласса | CNN ROC-AUC (Test) | Hybrid ROC-AUC (Val) | Hybrid ROC-AUC (Test) | Прирост (Δ ROC-AUC) | CNN PR-AUC (Test) | Hybrid PR-AUC (Test) | Прирост (Δ PR-AUC) |")
print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

for r in results:
    s_auc = "+" if r["delta_auc"] >= 0 else ""
    s_pr = "+" if r["delta_prauc"] >= 0 else ""
    print(
        f"| **`{r['target']}`** | {r['cnn_test_auc']:.4f} | {r['val_auc']:.4f} | "
        f"**{r['xgb_test_auc']:.4f}** | **{s_auc}{r['delta_auc']:.4f}** | "
        f"{r['cnn_test_prauc']:.4f} | **{r['xgb_test_prauc']:.4f}** | **{s_pr}{r['delta_prauc']:.4f}** |"
    )

mean_cnn_auc = np.mean([r["cnn_test_auc"] for r in results])
mean_val_auc = np.mean([r["val_auc"] for r in results])
mean_xgb_auc = np.mean([r["xgb_test_auc"] for r in results])
mean_delta_auc = mean_xgb_auc - mean_cnn_auc

mean_cnn_pr = np.mean([r["cnn_test_prauc"] for r in results])
mean_xgb_pr = np.mean([r["xgb_test_prauc"] for r in results])
mean_delta_pr = mean_xgb_pr - mean_cnn_pr

s_m_auc = "+" if mean_delta_auc >= 0 else ""
s_m_pr = "+" if mean_delta_pr >= 0 else ""

print(
    f"| **`Macro Average`** | **{mean_cnn_auc:.4f}** | **{mean_val_auc:.4f}** | "
    f"**{mean_xgb_auc:.4f}** | **{s_m_auc}{mean_delta_auc:.4f}** | "
    f"**{mean_cnn_pr:.4f}** | **{mean_xgb_pr:.4f}** | **{s_m_pr}{mean_delta_pr:.4f}** |"
)


# 8. Построение графика Топ-15 признаков по Information Gain
print("\n[5/5] Генерация графика важности признаков figures/fig4_feature_importance.png...")
total_gain = gain_accumulator.sum()
norm_gain = gain_accumulator / total_gain if total_gain > 0 else gain_accumulator

top_indices = np.argsort(norm_gain)[-15:]
top_features = [FEATURE_COLS[i] for i in top_indices]
top_values = norm_gain[top_indices]

plt.figure(figsize=(10, 6), dpi=300)
colors = ["#ff9933" if feat in PHYSIOLOGY_SET else "#5b9bd5" for feat in top_features]

plt.barh(range(len(top_features)), top_values, color=colors, height=0.68)
plt.yticks(range(len(top_features)), top_features, fontsize=10)
plt.xlabel("Относительная важность (Information Gain)", fontsize=11)
plt.title("Топ-15 признаков в ансамбле MedData AI (Полная выборка PTB-XL, N = 21 799)", fontsize=12, fontweight="bold")
plt.grid(axis="x", linestyle="--", alpha=0.5)

legend_elements = [
    Patch(facecolor="#ff9933", label="Физиология и клиника (Кардио-индексы + Вольтаж + Возраст/ИМТ)"),
    Patch(facecolor="#5b9bd5", label="Нейросеть (Логиты, вероятности и прототипы EchoNext-Mini V2)"),
]
plt.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=9.5)
plt.tight_layout()

out_fig = FIGURES_DIR / "fig4_feature_importance.png"
plt.savefig(out_fig, dpi=300)
plt.close()
print(f"      График успешно сохранен: {out_fig}")