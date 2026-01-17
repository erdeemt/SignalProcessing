# -*- coding: utf-8 -*-

"""
SIGNAL PROCESSING AND MACHINE LEARNING 2024-2025 FALL
FINAL : VOICE COMMAND RECOGNITION with IMPROVED DEEP LEARNING MODELS & MACHINE LEARNING
Script 10: Stage 2 -- XGBoost Model Training for ROOF Dataset

Team: 211805048_211805054_231805003
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, f1_score
import xgboost as xgb 
from imblearn.over_sampling import SMOTE 

# =====================================================================
# CONFIG
# =====================================================================
STUDENT_NO = "211805048_211805054_231805003"
MODELS_DIR = "models/ml"

# =====================================================================
# TRAIN ROOF MODEL FUNCTION
# =====================================================================

def train_roof_for_lang(lang_code):
    input_csv = f"roof_dataset_clean_{lang_code}.csv"
    model_output = f"{MODELS_DIR}/{STUDENT_NO}_roof_model_{lang_code}.joblib"
    
    if not os.path.exists(input_csv):
        print(f"\n[!] EVEN: {input_csv} not found! skipping...")
        return
    
    # 1. Load Data 
    df = pd.read_csv(input_csv)
    
    # 2. Features and Target Bearr [cite: 43]
    # Everything except record_id, filename, and y_true are attributes.
    drop_cols = ['record_id', 'filename', 'y_true']
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df['y_true']

    # Start tags from 0 (XGBoost requirement)
    y_adjusted = y - 1 
    num_classes = len(np.unique(y_adjusted))

    print(f"\n--- {lang_code} ROOF TRAINING BEGINS ---")
    print(f"[*] Number of Samples: {len(df)}, Number of Features: {X.shape[1]}")

    # 3. Balance Small Classes with SMOTE (For training set)
    # Since the number of samples per class is low, we are increasing the data.
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(X, y_adjusted)

    # 4. Train/Test Split 
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=47, stratify=y_resampled
    )

    # 5. Scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # 6. XGBoost Model (More aggressive and TF-IDF sensitive) 
    model = xgb.XGBClassifier(
        n_estimators=300,        # We increased the number of trees
        max_depth=8,             # We capture TF-IDF details by increasing the depth
        learning_rate=0.05,      # We reduced it for more stable learning
        objective='multi:softprob',
        num_class=num_classes,
        random_state=42,
        tree_method='hist'       # For fast processing
    )
    
    model.fit(X_train_sc, y_train)

    # 7. Evaluation
    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')

    print("\n" + "="*60)
    print(f"{lang_code} ROOF EVALUATION REPORT")
    print("="*60)
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1 Score: {f1_macro:.4f}")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred))
    print("="*60)

    # 8. Save
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_bundle = {
        'model': model, 
        'scaler': scaler, 
        'features': list(X.columns),
        'lang': lang_code,
        'student_no': STUDENT_NO
    }
    joblib.dump(save_bundle, model_output)
    print(f"[✓] {lang_code} Model saved: {model_output}")

if __name__ == "__main__":
    # Both TURKISH and WHY training begin sequentially.
    for lang in ['TR', 'EN']:
        train_roof_for_lang(lang)