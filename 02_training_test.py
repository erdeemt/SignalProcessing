
"""
SIGNAL PROCESSING AND MACHINE LEARNING 2024-2025 FALL
MIDTERM : VOICE COMMAND RECOGNITION
Script 02: Model Training and Evaluation

"""

import os
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE, RandomOverSampler

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIG
# =============================================================================

STUDENT_NO = "231805003"
INPUT_DIR = "./tabular_datasets"

RANDOM_SEED = 42
TEST_SIZE = 0.2
N_JOBS = -1

USE_SUBSET = False
SUBSET_RATIO = 1          
MAX_SAMPLES_PER_CLASS = 1000    # Max sample per class
SMOTE_TARGET_PER_CLASS = 400   # Smote target after training

# Lang based labeling
BACKGROUND_LABEL = 0
TR_ALLOWED = set(range(1, 67))   # 1..66
EN_ALLOWED = set(range(1, 70))   # 1..69
EXCESS_LABEL_POLICY = 'drop'     # 'drop' or 'background'

# OUTPUTS
def setup_output_dirs():
    dirs = {
        'results': './results',
        'models': './models',
        'plots': './results/confusion_matrices'
    }
    for p in dirs.values():
        os.makedirs(p, exist_ok=True)
    return dirs

# =============================================================================
# DISCOVER CSV
# =============================================================================

def discover_csv_files(input_dir=INPUT_DIR):
    """
    Beklenen isim: {STUDENT_NO}_{feature_family}_{win}_{hop}.csv
    feature_family: 'time_freq' veya 'mfcc' veya 'mel'
    """
    csv_files = []
    if not os.path.exists(input_dir):
        print(f"[ERROR] Directory not found: {input_dir}")
        return csv_files

    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".csv"):
            continue
        if not filename.startswith(f"{STUDENT_NO}_"):
            continue

        full_path = os.path.join(input_dir, filename)
        core = filename.replace(f"{STUDENT_NO}_", "").replace(".csv", "")
        parts = core.split("_")

        # time_freq iki kelime olduğu için:
        if len(parts) >= 4 and parts[0] == "time" and parts[1] == "freq":
            feature_family = "time_freq"
            window_length = parts[2]
            hop = parts[3]
        elif len(parts) >= 3:
            feature_family = parts[0]     # mfcc / mel
            window_length = parts[1]
            hop = parts[2]
        else:
            print(f"[WARN] Cannot parse filename: {filename}")
            continue

        csv_files.append({
            'path': full_path,
            'filename': filename,
            'feature_family': feature_family,
            'window_length': window_length,
            'hop': hop
        })
    return csv_files

def load_dataset(csv_path):
    try:
        df = pd.read_csv(csv_path)
        for col in ['label', 'language']:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        meta_cols = ['sample_id', 'filename', 'window_start_s', 'window_end_s', 'label', 'language']
        feat_cols = [c for c in df.columns if c not in meta_cols]

        # NaN feature temizliği
        nan_total = df[feat_cols].isna().sum().sum()
        if nan_total > 0:
            df = df.dropna(subset=feat_cols)
        if df.shape[0] == 0:
            print("    [ERROR] Empty after cleaning.")
            return None
        return df
    except Exception as e:
        print(f"    [ERROR loading] {e}")
        return None

# =============================================================================
# LANG-BASED LABEL CLEANING
# =============================================================================

def enforce_language_label_space(df, policy=EXCESS_LABEL_POLICY):
    """
    TR için 1..66, EN için 1..69 dışındakileri:
      - 'drop' ise tamamen sil
      - 'background' ise 0 yap
    NOT: df['label'] burada ZATEN int olmalı (background=0).
    """
    df = df.copy()
    if 'language' not in df.columns or 'label' not in df.columns:
        return df

    mask_tr = df['language'] == 'TR'
    mask_en = df['language'] == 'EN'

    def fix_block(block, allowed):
        if block.empty:
            return block
        if policy == 'drop':
            keep = block['label'].isin(allowed) | (block['label'] == BACKGROUND_LABEL)
            return block[keep]
        else:
            bad = (~block['label'].isin(allowed)) & (block['label'] != BACKGROUND_LABEL)
            block.loc[bad, 'label'] = BACKGROUND_LABEL
            return block

    tr_fixed = fix_block(df[mask_tr], TR_ALLOWED)
    en_fixed = fix_block(df[mask_en], EN_ALLOWED)
    rest = df[~(mask_tr | mask_en)]
    return pd.concat([tr_fixed, en_fixed, rest], ignore_index=True)


def drop_rare_classes(df, min_count=2):
    """background haricinde örneği 2'den az olan sınıfları düş."""
    df = df.copy()
    counts = df['label'].value_counts()
    rare = counts[(counts.index != BACKGROUND_LABEL) & (counts < min_count)].index
    if len(rare) > 0:
        df = df[~df['label'].isin(rare)]
    return df

# =============================================================================
# PREPROCESSING
# =============================================================================

def create_balanced_subset(df, ratio=SUBSET_RATIO, max_per_class=MAX_SAMPLES_PER_CLASS, random_state=RANDOM_SEED):
    """Her sınıftan oransal ve üst limitli örnek al (min 2 garanti)."""
    pieces = []
    for lab in df['label'].unique():
        block = df[df['label'] == lab]
        n = len(block)
        if n <= 2:
            take = n
        else:
            take = max(2, int(n * ratio))
        take = min(take, max_per_class, n)
        pieces.append(block.sample(n=take, random_state=random_state))
    return pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=random_state)

def preprocess_data(df_lang):
    """
    Tek dil için veri: önce label’ları normalize edip int’e çevir,
    sonra dil-bazlı aralık uygula, sonra rare class drop, sonra (varsa) subset.
    """
    df = df_lang.copy()

    # 1) LABEL NORMALIZATION 
    # background -> 0
    df['label'] = df['label'].replace('background', BACKGROUND_LABEL)
    # convert numeric (ex. "12" -> 12); nonconvertable -> NaN
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    # drop NaN
    df = df.dropna(subset=['label'])
    # convert
    df['label'] = df['label'].astype(int)

    # 2) Lang based space
    df = enforce_language_label_space(df)

    # 3) Drop rare
    df = drop_rare_classes(df, min_count=2)

    # 4) OPTIONAL
    if USE_SUBSET:
        original = len(df)
        df = create_balanced_subset(df)
        print(f"    Subset: {len(df)}/{original} samples ({100*len(df)/max(1,original):.1f}%)")

    # 5) PREPARE X, y
    y = df['label'].astype(int).values

    meta = ['sample_id', 'filename', 'window_start_s', 'window_end_s', 'label', 'language']
    feat_cols = [c for c in df.columns if c not in meta]
    X = df[feat_cols].astype(np.float32).copy()
    X.reset_index(drop=True, inplace=True)

    uniq, cnt = np.unique(y, return_counts=True)
    label_stats = dict(zip([int(u) for u in uniq], [int(c) for c in cnt]))
    print(f"    Labels: {label_stats}")

    return X, y, feat_cols

def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED):
    uniq, cnt = np.unique(y, return_counts=True)
    minc = cnt.min() if len(cnt) else 0
    if minc < 2:
        print("    [WARN] class with 1 sample; non-stratified split")
        return train_test_split(X, y, test_size=test_size, shuffle=True, random_state=random_state)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train).astype(np.float32)
    X_te = scaler.transform(X_test).astype(np.float32)
    return X_tr, X_te, scaler

def apply_smote(X_train_scaled, y_train, target_per_class=SMOTE_TARGET_PER_CLASS, random_state=RANDOM_SEED):
    uniq, cnt = np.unique(y_train, return_counts=True)
    counts = dict(zip([int(u) for u in uniq], [int(c) for c in cnt]))
    # TARGET ONLY FOR INSUFFICIENT CLASSES
    sampling_strategy = {cls: target_per_class for cls, c in counts.items() if c < target_per_class}

    if len(sampling_strategy) == 0:
        print("SMOTE: Not needed")
        return X_train_scaled, y_train

    min_class = min([counts[c] for c in sampling_strategy.keys()])
    if min_class < 2:
        print("     Using RandomOverSampler (min_class<2)")
        ros = RandomOverSampler(random_state=random_state, sampling_strategy=sampling_strategy)
        X_res, y_res = ros.fit_resample(X_train_scaled, y_train)
        print(f"    After ROS: {len(X_res)} (from {len(X_train_scaled)})")
        return X_res.astype(np.float32), y_res

    k_neighbors = max(1, min(5, min_class - 1))
    print(f"    SMOTE: k={k_neighbors}, target={target_per_class}/class")
    sm = SMOTE(random_state=random_state, sampling_strategy=sampling_strategy, k_neighbors=k_neighbors)
    X_res, y_res = sm.fit_resample(X_train_scaled, y_train)
    print(f"    After SMOTE: {len(X_res)} (from {len(X_train_scaled)})")
    return X_res.astype(np.float32), y_res

# =============================================================================
# MODELS
# =============================================================================

def get_model_configs():
    configs = {}

    configs['RandomForest'] = {
        'model': RandomForestClassifier(
            random_state=RANDOM_SEED, n_jobs=N_JOBS,
            class_weight='balanced', max_features='sqrt'
        ),
        'params': {
            'n_estimators': [100, 200, 300],
            'max_depth': [15, 25, 35],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'needs_scaling': False,
        'scoring': 'f1_macro'
    }

    configs['KNN'] = {
        'model': KNeighborsClassifier(n_jobs=N_JOBS),
        'params': {
            'n_neighbors': [3, 5, 7, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        },
        'needs_scaling': True,
        'scoring': 'accuracy'
    }

    return configs

def train_models(X_train, y_train):
    from sklearn.model_selection import ParameterSampler
    configs = get_model_configs()
    out = {}

    print("\n    Training models...")
    for name, cfg in configs.items():
        model = cfg['model']
        params = cfg['params']
        scoring = cfg['scoring']

        candidates = list(ParameterSampler(params, n_iter=100, random_state=RANDOM_SEED))
        n_iter = min(15, len(candidates))  # 15 random try
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=n_iter,
            scoring=scoring,
            cv=5,
            n_jobs=N_JOBS,
            random_state=RANDOM_SEED,
            verbose=0
        )
        print(f"      -> {name} ({n_iter} iters × 5CV, scoring={scoring})")
        t0 = time.time()
        search.fit(X_train, y_train)
        t1 = time.time()
        out[name] = {
            'model': search.best_estimator_,
            'best_params': search.best_params_,
            'best_cv_score': search.best_score_,
            'training_time': t1 - t0,
            'needs_scaling': cfg['needs_scaling']
        }
        print(f"         Best {scoring}: {search.best_score_:.3f} | time: {t1 - t0:.1f}s")
    return out

# =============================================================================
# EVALUATION
# =============================================================================

def compute_specificity_macro(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    n = cm.shape[0]
    specs = []
    for i in range(n):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specs.append(spec)
    return float(np.mean(specs))

def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    return {
        'model_name': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'specificity_macro': compute_specificity_macro(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'y_pred': y_pred
    }

def evaluate_all_models(trained, X_test, y_test):
    results = {}
    print("\n    Evaluating on test set...")
    for name, res in trained.items():
        m = res['model']
        metrics = evaluate_model(m, X_test, y_test, name)
        results[name] = metrics
        print(f"      {name}: Acc={metrics['accuracy']:.4f} | F1={metrics['f1_macro']:.4f} | "
              f"Prec={metrics['precision_macro']:.4f} | Rec={metrics['recall_macro']:.4f} | "
              f"Spec={metrics['specificity_macro']:.4f}")
    return results

def measure_response_time(model, scaler, X_test_raw, needs_scaling, n_runs=50, n_warmup=5):
    idx = np.random.randint(0, X_test_raw.shape[0])
    samp_raw = X_test_raw.iloc[idx:idx+1].values if hasattr(X_test_raw, 'iloc') else X_test_raw[idx:idx+1]

    # warmup
    for _ in range(n_warmup):
        samp = scaler.transform(samp_raw) if scaler is not None else samp_raw
        _ = model.predict(samp)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        samp = scaler.transform(samp_raw) if scaler is not None else samp_raw
        _ = model.predict(samp)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    return float(np.mean(times)), float(np.std(times))

def select_best_model(eval_results, trained, resp_times):
    rows = []
    for name, met in eval_results.items():
        avg, std = resp_times.get(name, (0, 0))
        rows.append({
            'Model': name,
            'Accuracy': met['accuracy'],
            'F1_macro': met['f1_macro'],
            'Precision_macro': met['precision_macro'],
            'Recall_macro': met['recall_macro'],
            'Specificity_macro': met['specificity_macro'],
            'Training_time_s': trained[name]['training_time'],
            'Response_time_ms': avg,
            'Response_std_ms': std
        })
    df = pd.DataFrame(rows).sort_values(
        by=['F1_macro', 'Specificity_macro', 'Response_time_ms'],
        ascending=[False, False, True]
    )
    best_name = df.iloc[0]['Model']
    return best_name, trained[best_name], df

# =============================================================================
# SAVING
# =============================================================================

def plot_confusion_matrix(metrics, dataset_key, out_dir):
    cm = metrics['confusion_matrix']
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f"Confusion Matrix\n{metrics['model_name']} - {dataset_key}")
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.tight_layout()
    path = os.path.join(out_dir, f"cm_{dataset_key}_{metrics['model_name']}.png")
    plt.savefig(path, dpi=110, bbox_inches='tight')
    plt.close()
    print(f"      CM saved: {os.path.basename(path)}")

def save_dataset_results(dataset_info, best_name, best_result,
                         scaler, metrics_df, eval_results, dirs, feature_names):
    lang = dataset_info['language']
    dataset_key = f"{dataset_info['feature_family']}_{dataset_info['window_length']}_{dataset_info['hop']}_{lang}"

    # model
    model_path = os.path.join(dirs['models'], f"{STUDENT_NO}_{dataset_key}_best_model.joblib")
    joblib.dump(best_result['model'], model_path)
    print(f"      Model saved: {os.path.basename(model_path)}")

    # scaler
    scaler_path = os.path.join(dirs['models'], f"{STUDENT_NO}_{dataset_key}_scaler.joblib")
    joblib.dump(scaler, scaler_path)

    # params
    params_path = os.path.join(dirs['models'], f"{STUDENT_NO}_{dataset_key}_params.json")
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump({
            'student_no': STUDENT_NO,
            'language': lang,
            'feature_family': dataset_info['feature_family'],
            'window_length': dataset_info['window_length'],
            'hop': dataset_info['hop'],
            'best_model': best_name,
            'best_params': best_result['best_params'],
            'cv_score': best_result['best_cv_score'],
            'needs_scaling': best_result['needs_scaling'],
            'feature_names': feature_names
        }, f, indent=2)

    # metrics csv
    metrics_csv = os.path.join(dirs['results'], f"{STUDENT_NO}_{dataset_key}_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"      Metrics saved: {os.path.basename(metrics_csv)}")

    # CM (best model)
    best_metrics = eval_results[best_name]
    plot_confusion_matrix(best_metrics, dataset_key, dirs['plots'])

    return dataset_key

def save_master_results(all_results, dirs):
    rows = []
    for key, obj in all_results.items():
        for _, r in obj['metrics'].iterrows():
            rows.append({
                'dataset': key,
                'feature_family': obj['feature_family'],
                'window_length': obj['window_length'],
                'hop': obj['hop'],
                'language': obj['language'],
                **r.to_dict()
            })
    if not rows:
        return
    dfm = pd.DataFrame(rows)
    path_csv = os.path.join(dirs['results'], f"{STUDENT_NO}_all_results.csv")
    dfm.to_csv(path_csv, index=False)
    print(f"\n  Master CSV: {os.path.basename(path_csv)}")

    log = {
        'timestamp': datetime.now().isoformat(),
        'student_no': STUDENT_NO,
        'random_seed': RANDOM_SEED,
        'test_size': TEST_SIZE,
        'subset_ratio': SUBSET_RATIO,
        'max_samples_per_class': MAX_SAMPLES_PER_CLASS,
        'smote_target': SMOTE_TARGET_PER_CLASS,
        'excess_label_policy': EXCESS_LABEL_POLICY,
        'languages': ['TR', 'EN'],
        'models': ['RandomForest', 'KNN'],
        'total_datasets': len(all_results)
    }
    path_json = os.path.join(dirs['results'], f"{STUDENT_NO}_master_log.json")
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"  Master log : {os.path.basename(path_json)}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*80)
    print("BILINGUAL TRAINING PIPELINE (TR & EN SEPARATE)")
    print("="*80)

    np.random.seed(RANDOM_SEED)
    dirs = setup_output_dirs()
    csv_files = discover_csv_files()

    if len(csv_files) == 0:
        print("[ERROR] No CSVs found in ./tabular_datasets/. Run 01 first.")
        return

    print(f"\nFound {len(csv_files)} dataset CSV(s).")
    all_datasets_results = {}
    t0 = time.time()

    for idx, info in enumerate(csv_files, 1):
        dataset_id = f"{info['feature_family']}_{info['window_length']}_{info['hop']}"
        print("\n" + "="*80)
        print(f"[{idx}/{len(csv_files)}] DATASET: {dataset_id}")
        print("="*80)

        df = load_dataset(info['path'])
        if df is None or df.empty:
            continue
        print(f"    Loaded: {df.shape[0]} samples, {df.shape[1]} cols")

        # LANG BASED SEPARATE TRAINING
        for LANG in ['TR', 'EN']:
            df_lang = df[df['language'] == LANG].copy()
            if df_lang.empty:
                print(f"    [{LANG}] no samples → skip.")
                continue

            # PREPROCESSING
            X, y, feature_names = preprocess_data(df_lang)
            uniq = np.unique(y)
            if len(uniq) < 2:
                print(f"    [{LANG}] insufficient unique classes ({len(uniq)}) → skip.")
                continue

            print(f"    [{LANG}] Features: {X.shape[1]} | Samples: {X.shape[0]} | Classes: {len(uniq)}")

            # Train/Test
            X_train, X_test, y_train, y_test = split_data(X, y)

            # SCALING -> SMOTE
            print(f"    [{LANG}] Scaling...")
            X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

            print(f"    [{LANG}] SMOTE on TRAIN...")
            X_train_final, y_train_final = apply_smote(X_train_sc, y_train)

            # TRAINING
            trained = train_models(X_train_final, y_train_final)
            if not any(trained.values()):
                print(f"    [{LANG}] No model trained → skip.")
                continue

            # EVALUATION (scaled test)
            eval_results = evaluate_all_models(trained, X_test_sc, y_test)

            # Response time (unscaled raw -> scaler + predict)
            print(f"\n    [{LANG}] Measuring response times...")
            resp_times = {}
            for name, res in trained.items():
                avg, std = measure_response_time(res['model'], scaler, X_test, res['needs_scaling'])
                resp_times[name] = (avg, std)
                print(f"      [{LANG}] {name}: {avg:.2f} ± {std:.2f} ms/sample")

            # BEST
            best_name, best_obj, metrics_df = select_best_model(eval_results, trained, resp_times)

            # SAVE
            print(f"\n    [{LANG}] Saving...")
            info_lang = dict(info)
            info_lang['language'] = LANG
            dataset_key = save_dataset_results(
                info_lang, best_name, best_obj,
                scaler, metrics_df, eval_results, dirs, feature_names
            )

            all_datasets_results[dataset_key] = {
                'feature_family': info['feature_family'],
                'window_length': info['window_length'],
                'hop': info['hop'],
                'language': LANG,
                'best_model': best_name,
                'metrics': metrics_df
            }

    # Master
    if len(all_datasets_results) > 0:
        print("\n" + "="*80)
        print("SAVING MASTER RESULTS")
        print("="*80)
        save_master_results(all_datasets_results, dirs)

    # SUMMARY
    dt = time.time() - t0
    h = int(dt // 3600); m = int((dt % 3600) // 60); s = dt % 60
    tstr = f"{h}h {m}m {s:.1f}s" if h else (f"{m}m {s:.1f}s" if m else f"{s:.1f}s")

    print("\n" + "="*80)
    print("✔ TRAINING COMPLETE")
    print("="*80)
    print(f"Datasets processed: {len(all_datasets_results)}/{len(csv_files)}")
    print(f"Total time: {tstr}")
    print(f"Results dir: {dirs['results']}")
    print(f"Models dir : {dirs['models']}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
