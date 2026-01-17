# -*- coding: utf-8 -*-
"""
SIGNAL PROCESSING AND MACHINE LEARNING 2024-2025 FALL
FINAL : VOICE COMMAND RECOGNITION with IMPROVED DEEP LEARNING MODELS & MACHINE LEARNING
Script 08: Stage 1 -- Offline Inference Test with GUI

"""
# ============================================================================
# IMPORTS
# ============================================================================

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import os
import json
import glob
import numpy as np
import pandas as pd
import time
import librosa
from scipy import signal as scipy_signal
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.ndimage import gaussian_filter1d
import joblib
from threading import Thread
from collections import Counter
import sounddevice as sd
from sklearn.metrics import classification_report

# ============================================================================
# CONFIGURATION
# ============================================================================
STUDENT_NO = "231805003"
SAMPLE_RATE = 16000
FRAME_SIZE_MS = 25
FRAME_HOP_MS = 10

FREQ_BANDS = [
    (0, 200), (200, 500), (500, 1000), (1000, 2000),
    (2000, 3000), (3000, 4000), (4000, 6000), (6000, 8000)
]

# ============================================================================
# FEATURE EXTRACTION 
# ============================================================================

def extract_time_features(y):
    """Time-domain features."""
    features = {}
    features['time_mean'] = np.mean(y)
    features['time_std'] = np.std(y)
    features['time_median'] = np.median(y)
    features['time_peak'] = np.max(np.abs(y))
    features['time_peak_to_peak'] = np.max(y) - np.min(y)
    features['time_rms'] = np.sqrt(np.mean(y ** 2))
    features['time_kurtosis'] = stats.kurtosis(y)
    features['time_skewness'] = stats.skew(y)
    features['time_crest_factor'] = features['time_peak'] / (features['time_rms'] + 1e-10)

    noise_floor = np.percentile(np.abs(y), 10)
    signal_level = np.percentile(np.abs(y), 90)
    features['time_snr_estimate'] = 20 * np.log10((signal_level + 1e-10) / (noise_floor + 1e-10))

    zero_crossings = np.where(np.diff(np.signbit(y)))[0]
    features['time_zcr'] = len(zero_crossings) / len(y)

    analytic_signal = scipy_signal.hilbert(y)
    envelope = np.abs(analytic_signal)
    features['time_envelope_mean'] = np.mean(envelope)
    features['time_envelope_std'] = np.std(envelope)
    features['time_envelope_max'] = np.max(envelope)
    features['time_envelope_min'] = np.min(envelope)
    features['time_dynamic_range_db'] = 20 * np.log10(
        (np.max(np.abs(y)) + 1e-10) / (np.mean(np.abs(y)) + 1e-10)
    )

    return features


def extract_freq_features(y, sr):
    """Frequency-domain features."""
    features = {}
    n = len(y)
    Y = fft(y)
    freqs = fftfreq(n, 1 / sr)
    magnitude = np.abs(Y[:n // 2])
    freqs = freqs[:n // 2]
    magnitude_smooth = gaussian_filter1d(magnitude, sigma=5)

    for i, (low, high) in enumerate(FREQ_BANDS):
        band_mask = (freqs >= low) & (freqs < high)
        band_magnitude = magnitude_smooth[band_mask]
        band_freqs = freqs[band_mask]

        if len(band_magnitude) > 0:
            features[f'freq_band_{i+1}_power'] = np.sum(band_magnitude ** 2) / len(band_magnitude)
            peak_idx = np.argmax(band_magnitude)
            features[f'freq_band_{i+1}_peak_freq'] = band_freqs[peak_idx]
            features[f'freq_band_{i+1}_peak_power'] = band_magnitude[peak_idx]
            features[f'freq_band_{i+1}_spread'] = np.std(band_magnitude)
            features[f'freq_band_{i+1}_mean_freq'] = np.mean(band_freqs)
        else:
            features[f'freq_band_{i+1}_power'] = 0
            features[f'freq_band_{i+1}_peak_freq'] = 0
            features[f'freq_band_{i+1}_peak_power'] = 0
            features[f'freq_band_{i+1}_spread'] = 0
            features[f'freq_band_{i+1}_mean_freq'] = 0

    features['freq_spectral_centroid'] = np.sum(freqs * magnitude_smooth) / (
        np.sum(magnitude_smooth) + 1e-10
    )

    cumsum = np.cumsum(magnitude_smooth)
    total = cumsum[-1]
    if total > 0:
        rolloff_85_idx = np.where(cumsum >= 0.85 * total)[0]
        rolloff_95_idx = np.where(cumsum >= 0.95 * total)[0]
        features['freq_spectral_rolloff_85'] = (
            freqs[rolloff_85_idx[0]] if len(rolloff_85_idx) > 0 else 0
        )
        features['freq_spectral_rolloff_95'] = (
            freqs[rolloff_95_idx[0]] if len(rolloff_95_idx) > 0 else 0
        )
    else:
        features['freq_spectral_rolloff_85'] = 0
        features['freq_spectral_rolloff_95'] = 0

    features['freq_spectral_flux'] = np.std(magnitude_smooth)
    geometric_mean = np.exp(np.mean(np.log(magnitude_smooth + 1e-10)))
    arithmetic_mean = np.mean(magnitude_smooth)
    features['freq_spectral_flatness'] = geometric_mean / (arithmetic_mean + 1e-10)
    magnitude_prob = magnitude_smooth / (np.sum(magnitude_smooth) + 1e-10)
    features['freq_spectral_entropy'] = -np.sum(
        magnitude_prob * np.log2(magnitude_prob + 1e-10)
    )

    return features


def extract_mfcc_features(y, sr):
    """MFCC features."""
    features = {}
    n_fft = int(FRAME_SIZE_MS * sr / 1000)
    hop_length = int(FRAME_HOP_MS * sr / 1000)

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length
    )

    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfcc[i])
        features[f'mfcc_{i+1}_std'] = np.std(mfcc[i])
        features[f'mfcc_{i+1}_median'] = np.median(mfcc[i])
        features[f'mfcc_{i+1}_min'] = np.min(mfcc[i])
        features[f'mfcc_{i+1}_max'] = np.max(mfcc[i])
        features[f'mfcc_{i+1}_range'] = np.max(mfcc[i]) - np.min(mfcc[i])

    mfcc_delta = librosa.feature.delta(mfcc[:5])
    for i in range(5):
        features[f'mfcc_delta_{i+1}_mean'] = np.mean(mfcc_delta[i])
        features[f'mfcc_delta_{i+1}_std'] = np.std(mfcc_delta[i])

    mfcc_delta2 = librosa.feature.delta(mfcc[:3], order=2)
    for i in range(3):
        features[f'mfcc_delta2_{i+1}_mean'] = np.mean(mfcc_delta2[i])
        features[f'mfcc_delta2_{i+1}_std'] = np.std(mfcc_delta2[i])

    features['mfcc_energy_mean'] = np.mean(mfcc[0])
    features['mfcc_spectral_flux'] = np.mean(
        np.sqrt(np.sum(np.diff(mfcc, axis=1) ** 2, axis=0))
    )

    return features


def extract_mel_features(y, sr):
    """MEL features."""
    features = {}
    n_fft = int(FRAME_SIZE_MS * sr / 1000)
    hop_length = int(FRAME_HOP_MS * sr / 1000)

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=40
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    for i in range(40):
        features[f'mel_{i+1}_mean'] = np.mean(mel_spec_db[i])
        features[f'mel_{i+1}_std'] = np.std(mel_spec_db[i])
        features[f'mel_{i+1}_median'] = np.median(mel_spec_db[i])
        features[f'mel_{i+1}_max'] = np.max(mel_spec_db[i])
        features[f'mel_{i+1}_min'] = np.min(mel_spec_db[i])

    features['mel_total_energy'] = np.sum(mel_spec)
    features['mel_mean_energy'] = np.mean(mel_spec)
    features['mel_std_energy'] = np.std(mel_spec)

    mel_freqs = librosa.mel_frequencies(n_mels=40, fmin=0, fmax=sr / 2)
    mel_centroid = np.sum(mel_freqs[:, np.newaxis] * mel_spec, axis=0) / (
        np.sum(mel_spec, axis=0) + 1e-10
    )
    features['mel_spectral_centroid_mean'] = np.mean(mel_centroid)
    features['mel_spectral_centroid_std'] = np.std(mel_centroid)

    mel_flux = np.sqrt(np.sum(np.diff(mel_spec_db, axis=1) ** 2, axis=0))
    features['mel_flux_mean'] = np.mean(mel_flux)
    features['mel_flux_std'] = np.std(mel_flux)

    return features


def extract_features(y, sr, feature_family):
    """Extract features based on family type."""
    if feature_family in ['time_freq', 'timefreq']:
        time_feats = extract_time_features(y)
        freq_feats = extract_freq_features(y, sr)
        return {**time_feats, **freq_feats}
    elif feature_family == 'mfcc':
        return extract_mfcc_features(y, sr)
    elif feature_family == 'mel':
        return extract_mel_features(y, sr)
    else:
        raise ValueError(f"Unknown feature family: {feature_family}")

# ============================================================================
# MODEL DISCOVERY
# ============================================================================

def discover_models(models_dir="./lower_models", results_dir="./results"):
    """
    Kullanıcının yeni CSV formatına göre (lang, Model) modelleri keşfeder.
    """
    models = []
    if not os.path.exists(models_dir):
        return models

    # 1. Read master results CSV
    master_csv = os.path.join(results_dir, f"{STUDENT_NO}_all_results.csv")
    results_df = None
    if os.path.exists(master_csv):
        try:
            results_df = pd.read_csv(master_csv)
            print(f"[*] Master CSV yüklendi: {master_csv}")
        except Exception as error:
            print(f"[!] CSV Okuma Hatası: {error}")

    # 2. Scan .joblib files
    model_files = glob.glob(os.path.join(models_dir, f"{STUDENT_NO}_*_best_model.joblib"))

    for model_path in model_files:
        basename = os.path.basename(model_path)
        
        # Load parameters (JSON) - We will get the Language and Model name from here
        params_path = model_path.replace("_best_model.joblib", "_params.json")
        params = {}
        if os.path.exists(params_path):
            with open(params_path, 'r', encoding='utf-8') as f:
                params = json.load(f)

        # Matching criteria with CSV
        lang_code = params.get('language', '??')    
        model_type = params.get('best_model', '??')  
        
        # Pull metrics from CSV (Find matching row)
        accuracy, f1_score, response_time = 0.0, 0.0, 0.0
        if results_df is not None:
            # filter using lang and Model columns
            matching_row = results_df[(results_df['lang'] == lang_code) & 
                                      (results_df['Model'] == model_type)]
            
            if not matching_row.empty:
                row = matching_row.iloc[0]
                accuracy = row.get('Accuracy', 0.0)
                f1_score = row.get('F1_macro', 0.0)
                response_time = row.get('Response_time_ms', 0.0)

        # Extract configuration parts from filename
        dataset_key = basename.replace(f"{STUDENT_NO}_", "").replace("_best_model.joblib", "")
        parts = dataset_key.split('_')
        
       # Feature family and window settings (for the display desired by the teacher)
        if len(parts) >= 4 and parts[0] == 'time' and parts[1] == 'freq':
            feature_family = 'time_freq'
            window_length = parts[2]
            hop = parts[3]
        else:
            feature_family = parts[0]
            window_length = parts[1] if len(parts) > 1 else "N/A"
            hop = parts[2] if len(parts) > 2 else "N/A"

        display_name = (
            f"{lang_code} | {feature_family} | {window_length} | "
            f"{hop} | Acc:{accuracy:.1%}"
        )

        models.append({
            'dataset_key': dataset_key,
            'feature_family': feature_family,
            'window_length': window_length,
            'hop': hop,
            'language': lang_code,
            'model_path': model_path,
            'scaler_path': model_path.replace("_best_model.joblib", "_scaler.joblib"),
            'params': params,
            'accuracy': accuracy,
            'f1_score': f1_score,
            'response_time': response_time,
            'model_name': model_type,
            'display_name': display_name,
        })

    # Sort by F1 score
    models.sort(key=lambda x: x['f1_score'], reverse=True)
    return models

# ============================================================================
# AUDIO PROCESSING
# ============================================================================

def load_ground_truth(audio_path):
    """Load ground truth labels from CSV and convert to time."""
    csv_path = audio_path.replace('.wav', '.csv')
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            
            # Find Original SR (from audio file)
            y_orig, orig_sr = librosa.load(audio_path, sr=None, mono=True)
            
            # Convert Sample → Time (same as 01_extraction)
            if 'start_sample' in df.columns and 'end_sample' in df.columns:
                df['start_time'] = df['start_sample'] / float(orig_sr)
                df['end_time'] = df['end_sample'] / float(orig_sr)
            
            # background → 0
            if 'label' in df.columns:
                df['label'] = df['label'].replace('background', 0)
                df['label'] = pd.to_numeric(df['label'], errors='coerce')
            
            return df
        except Exception as e:
            print(f"Error loading GT: {e}")
    return None


def process_audio_file(audio_path, model_info, model, scaler, needs_scaling, language):
    """
    It processes the audio file and produces all the keys (including end_time) that the interface/log expects.
    """
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    y = librosa.effects.preemphasis(y)

    gt_df = load_ground_truth(audio_path)
    has_gt = gt_df is not None

    window_length_s = float(model_info['window_length'].replace('s', '').replace('p', '.'))
    hop_s = float(model_info['hop'].replace('s', '').replace('p', '.'))

    window_samples = int(window_length_s * sr)
    hop_samples = int(hop_s * sr)

    predictions = []
    response_times = []

    start_sample = 0
    idx = 0
    while start_sample + window_samples <= len(y):
        t0 = time.perf_counter()
        
        # Window data and times
        window_data = y[start_sample : start_sample + window_samples]
        s_time = start_sample / sr
        e_time = (start_sample + window_samples) / sr
        
        #feature extraction
        features = extract_features(window_data, sr, model_info['feature_family'])
        X = pd.DataFrame([features], dtype=np.float32)

        #Feature Alignment
        feature_names = model_info['params'].get('feature_names')
        if feature_names:
            for fname in feature_names:
                if fname not in X.columns: X[fname] = 0.0
            X = X[feature_names]
        if scaler: X = scaler.transform(X)

        # Prediction and Confidence Score (p)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            y_pred = int(np.argmax(probs))
            confidence = float(np.max(probs))
        else:
            y_pred = int(model.predict(X)[0])
            confidence = 1.0

        t1 = time.perf_counter()
        
        # Ground Truth Comparison
        true_label = 0 if gt_df is not None else None
        if gt_df is not None:
            center_t = 0.5 * (s_time + e_time)
            for _, row in gt_df.iterrows():
                if row['start_time'] <= center_t < row['end_time']:
                    true_label = int(row['label'])
                    break

       # We add all the data that the Interface and Log expect
        predictions.append({
            'window_idx': idx,
            'start_time': s_time,
            'end_time': e_time,      
            'predicted_class': y_pred,
            'confidence': confidence, 
            'true_class': true_label,
            'correct': (true_label == y_pred) if true_label is not None else None,
            'response_time_ms': (t1 - t0) * 1000.0
        })
        
        response_times.append((t1 - t0) * 1000.0)
        start_sample += hop_samples
        idx += 1

    return predictions, response_times, has_gt

# ============================================================================
# GUI APPLICATION
# ============================================================================

class OfflineTestGUI:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Offline Voice Command Test - {STUDENT_NO}")
        self.root.geometry("1400x800")

        self.audio_files_en = []
        self.audio_files_tr = []
        self.available_models = []
        self.selected_model_info = None
        self.model = None
        self.scaler = None
        self.needs_scaling = False

        self.setup_gui()
        self.load_audio_files()
        self.load_models()
    
    # ---THE FUNCTION THAT RECEIVED AN ERROR HAS BEEN ADDED HERE ---
    def _get_command_name(self, label, lang):
        """Label numarasını isme dönüştürür."""
        if label == 0: return "background"
        # If there is label_map in params.json, pull it from there, otherwise write 'Class X'
        label_map = self.selected_model_info['params'].get('label_map', {})
        return label_map.get(str(label), label_map.get(int(label), f"Class {label}"))


# ============================================================================
# GUI SETUP
# ============================================================================
    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)

        # Left: Audio files
        left_frame = ttk.LabelFrame(main_frame, text="📁 Audio Files", padding="10")
        left_frame.grid(row=0, column=0, rowspan=3,
                        sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        tab_control = ttk.Notebook(left_frame)

        tab_en = ttk.Frame(tab_control)
        tab_control.add(tab_en, text='🇬🇧 English (EN)')
        self.listbox_en = tk.Listbox(tab_en, height=30, width=40)
        scroll_en = ttk.Scrollbar(tab_en, orient=tk.VERTICAL,
                                  command=self.listbox_en.yview)
        self.listbox_en.configure(yscrollcommand=scroll_en.set)
        self.listbox_en.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_en.pack(side=tk.RIGHT, fill=tk.Y)

        tab_tr = ttk.Frame(tab_control)
        tab_control.add(tab_tr, text='🇹🇷 Turkish (TR)')
        self.listbox_tr = tk.Listbox(tab_tr, height=30, width=40)
        scroll_tr = ttk.Scrollbar(tab_tr, orient=tk.VERTICAL,
                                  command=self.listbox_tr.yview)
        self.listbox_tr.configure(yscrollcommand=scroll_tr.set)
        self.listbox_tr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll_tr.pack(side=tk.RIGHT, fill=tk.Y)

        tab_control.pack(fill=tk.BOTH, expand=True)

        # Top right: Model selection
        model_frame = ttk.LabelFrame(main_frame, text="🤖 Model Selection", padding="10")
        model_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N), pady=(0, 10))

        select_row = ttk.Frame(model_frame)
        select_row.grid(row=0, column=0, columnspan=2,
                        sticky=(tk.W, tk.E), pady=(0, 5))

        ttk.Label(select_row, text="Select Model:").pack(side=tk.LEFT, padx=5)
        self.model_combo = ttk.Combobox(select_row, state='readonly', width=60)
        self.model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.model_combo.bind('<<ComboboxSelected>>', self.on_model_selected)
        self.best_model_tr_btn = ttk.Button(
            select_row, text="⭐ Select Best TR Model",
            command=self.select_best_model_tr, width=20
        )
        self.best_model_tr_btn.pack(side=tk.LEFT, padx=5)

        self.best_model_en_btn = ttk.Button(
            select_row, text="⭐ Select Best EN Model",
            command=self.select_best_model_en, width=20
        )
        self.best_model_en_btn.pack(side=tk.LEFT, padx=5)


        self.model_info_text = tk.Text(
            model_frame, height=9, width=80, state='disabled',
            bg='#f0f0f0', font=('Courier', 9)
        )
        self.model_info_text.grid(row=1, column=0, columnspan=2,
                                  sticky=(tk.W, tk.E), pady=10)

        # Bottom right: Results
        results_frame = ttk.LabelFrame(main_frame, text="📊 Test Results", padding="10")
        results_frame.grid(row=1, column=1, rowspan=2,
                           sticky=(tk.W, tk.E, tk.N, tk.S))

        self.results_text = scrolledtext.ScrolledText(
            results_frame, height=30, width=90, font=('Courier', 9)
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Bottom buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=(10, 0))

        self.process_btn = ttk.Button(
            btn_frame, text="🎯 Process Selected File",
            command=self.process_selected
        )
        self.process_btn.pack(side=tk.LEFT, padx=5)

        self.play_btn = ttk.Button(
            btn_frame, text="🔊 Play Audio",
            command=self.play_selected_audio
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = ttk.Button(
            btn_frame, text="🗑️ Clear Results",
            command=self.clear_results
        )
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        self.progress = ttk.Progressbar(
            btn_frame, mode='indeterminate', length=200
        )
        self.progress.pack(side=tk.LEFT, padx=20)

# ============================================================================
# LOADERS
# ============================================================================
    def load_audio_files(self):
        for lang, listbox, attr_name in [
            ('EN', self.listbox_en, 'audio_files_en'),
            ('TR', self.listbox_tr, 'audio_files_tr')
        ]:
            dir_path = f"./VC_dataset_{lang}"
            if os.path.exists(dir_path):
                files = sorted(
                    [f for f in os.listdir(dir_path) if f.endswith('.wav')]
                )
                setattr(self, attr_name, [os.path.join(dir_path, f) for f in files])
                for f in files:
                    listbox.insert(tk.END, f)

    def load_models(self):
        self.available_models = discover_models()

        if not self.available_models:
            messagebox.showwarning(
                "Warning",
                "No trained models found in ./models/\n"
                "Please run 02_training.py first!"
            )
            return

        model_names = [m['display_name'] for m in self.available_models]
        self.model_combo['values'] = model_names

        if model_names:
            self.model_combo.current(0)
            self.on_model_selected(None)

# ============================================================================
# MODEL SELECTION
# ============================================================================
    def on_model_selected(self, event):
        idx = self.model_combo.current()
        if idx < 0:
            return

        self.selected_model_info = self.available_models[idx]

        try:
            self.model = joblib.load(self.selected_model_info['model_path'])

            if self.selected_model_info['scaler_path']:
                self.scaler = joblib.load(self.selected_model_info['scaler_path'])
            else:
                self.scaler = None

            self.needs_scaling = self.selected_model_info['params'].get(
                'needs_scaling', False
            )

            accuracy = self.selected_model_info.get('accuracy', 0.0)
            f1_score = self.selected_model_info.get('f1_score', 0.0)
            response_time = self.selected_model_info.get('response_time', 0.0)
            model_name = self.selected_model_info.get('model_name', 'Unknown')
            language = self.selected_model_info.get('language', 'N/A')
            cv_score = self.selected_model_info['params'].get('cv_score', 0.0)

            info = (
                f"Model:         {model_name}\n"
                f"Language:      {language}\n"
                f"FeatureFamily: {self.selected_model_info['feature_family']}\n"
                f"Window:        {self.selected_model_info['window_length']} | "
                f"Hop: {self.selected_model_info['hop']}\n"
                f"─────────────────────────────────────────────────────\n"
                f"Performance Metrics:\n"
                f"  • Accuracy:      {accuracy:.2%}\n"
                f"  • F1 Score:      {f1_score:.4f}\n"
                f"  • CV Score:      {cv_score:.4f}\n"
                f"  • Response Time: {response_time:.2f} ms/window\n"
                f"─────────────────────────────────────────────────────\n"
                f"Scaling Required (from 02): {self.needs_scaling}\n"
            )

            self.model_info_text.config(state='normal')
            self.model_info_text.delete(1.0, tk.END)
            self.model_info_text.insert(1.0, info)
            self.model_info_text.config(state='disabled')

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")

    def _select_best_by_language(self, target_lang):
        """Internal function: selects the best model for the given language (TR/EN)."""
        if not self.available_models:
            messagebox.showwarning("Warning", "No models available!")
            return

        
        candidates = [
            (idx, m) for idx, m in enumerate(self.available_models)
            if m.get('language', None) == target_lang
        ]

        if not candidates:
            messagebox.showwarning(
                "Warning",
                f"No models found for language: {target_lang}"
            )
            return

       
        best_idx, best_model = max(
            candidates,
            key=lambda kv: kv[1].get('f1_score', 0.0)
        )

     
        self.model_combo.current(best_idx)
        self.on_model_selected(None)

        messagebox.showinfo(
            f"Best {target_lang} Model Selected",
            f"Selected best {target_lang} model:\n\n"
            f"Model:  {best_model['model_name']}\n"
            f"Lang:   {best_model.get('language', 'N/A')}\n"
            f"Config: {best_model['feature_family']} | "
            f"{best_model['window_length']} | {best_model['hop']}\n"
            f"Accuracy: {best_model['accuracy']:.2%}\n"
            f"F1 Score: {best_model['f1_score']:.4f}"
        )


    def select_best_model_tr(self):
       """Choose the best model for TR."""
       self._select_best_by_language('TR')


    def select_best_model_en(self):
        """Choose the best model for your EN."""
        self._select_best_by_language('EN')

# ============================================================================
# AUDIO SELECTION & PLAYBACK
# ============================================================================
    def get_selected_audio(self):
        sel_en = self.listbox_en.curselection()
        if sel_en:
            return self.audio_files_en[sel_en[0]], 'EN'

        sel_tr = self.listbox_tr.curselection()
        if sel_tr:
            return self.audio_files_tr[sel_tr[0]], 'TR'

        return None, None

    def play_selected_audio(self):
        audio_path, _ = self.get_selected_audio()
        if audio_path is None:
            messagebox.showwarning("Warning", "Select an audio file first!")
            return

        try:
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
            sd.play(y, sr)
            messagebox.showinfo(
                "Playing",
                f"🔊 Playing: {os.path.basename(audio_path)}"
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to play audio: {e}")

# ============================================================================
# PROCESSING
# ============================================================================
    def process_selected(self):
        if self.model is None:
            messagebox.showerror("Error", "No model selected!")
            return

        audio_path, lang = self.get_selected_audio()
        if audio_path is None:
            messagebox.showwarning("Warning", "Select an audio file first!")
            return

    
        model_lang = self.selected_model_info.get('language', None)
        if model_lang and model_lang != lang:
            if not messagebox.askyesno(
                "Language Mismatch",
                f"Selected audio is {lang}, but model is {model_lang}.\n"
                f"Continue anyway?"
            ):
                return

        self.process_btn.config(state='disabled')
        self.progress.start()

        thread = Thread(target=self._process_thread, args=(audio_path, lang))
        thread.start()

    def _process_thread(self, audio_path, lang):
        """Processes the file and writes it to the logs/folder in Step 4 format."""
        try:
            # 1. LOGING PREPARATION
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            log_filename = os.path.join(log_dir, os.path.splitext(os.path.basename(audio_path))[0] + ".txt")
            
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Processing: {os.path.basename(audio_path)}\n")
            self.results_text.insert(tk.END, f"Log: {log_filename}\n")
            self.results_text.insert(tk.END, "=" * 90 + "\n\n")
            self.root.update()

            predictions, response_times, has_gt = process_audio_file(
                audio_path, self.selected_model_info, self.model, self.scaler, self.needs_scaling, lang
            )

            # 2. WRITING TO FILE (Step 4 Format)
            with open(log_filename, "w", encoding="utf-8") as f_log:
                f_log.write(f"PREDICTION LOG - {os.path.basename(audio_path)}\n")
                f_log.write("-" * 60 + "\n")
                
                for pred in predictions:
                    t_val = pred['start_time']
                    p_id = pred['predicted_class']
                    p_val = pred['confidence']
                    cmd_text = self._get_command_name(p_id, lang)
                    
                    # Desired Format: t=0.80s | PRED=1 (turn on light) | p=0.88
                    line = f"t={t_val:5.2f}s | PRED={p_id:2d} ({cmd_text:20s}) | p={p_val:.2f}"
                    
                    self.results_text.insert(tk.END, line + "\n")
                    f_log.write(line + "\n")

            self.display_results(predictions, response_times, has_gt)

        except Exception as e:
            import traceback
            self.results_text.insert(tk.END, f"\nERROR: {str(e)}\n{traceback.format_exc()}")
        finally:
            self.progress.stop()
            self.process_btn.config(state='normal')

# ============================================================================
# DISPLAY RESULTS
# ============================================================================
    def display_results(self, predictions, response_times, has_gt):
        self.results_text.insert(tk.END, "WINDOW PREDICTIONS (first 20):\n")
        self.results_text.insert(tk.END, "-" * 90 + "\n")

        if has_gt:
            self.results_text.insert(
                tk.END,
                f"{'Win':<6} {'Start':<8} {'End':<8} "
                f"{'Pred':<6} {'True':<6} {'Match':<7} {'Time(ms)':<10}\n"
            )
        else:
            self.results_text.insert(
                tk.END,
                f"{'Win':<6} {'Start':<8} {'End':<8} "
                f"{'Pred':<6} {'Time(ms)':<10}\n"
            )
        self.results_text.insert(tk.END, "-" * 90 + "\n")

        for pred in predictions[:2000]:
            if has_gt:
                match_str = (
                    "✓" if pred['correct']
                    else "✗" if pred['correct'] is not None
                    else "-"
                )
                true_label = pred['true_class']
                self.results_text.insert(
                    tk.END,
                    f"{pred['window_idx']:<6} "
                    f"{pred['start_time']:<8.2f} {pred['end_time']:<8.2f} "
                    f"{pred['predicted_class']:<6} {true_label!s:<6} "
                    f"{match_str:<7} {pred['response_time_ms']:<10.2f}\n"
                )
            else:
                self.results_text.insert(
                    tk.END,
                    f"{pred['window_idx']:<6} "
                    f"{pred['start_time']:<8.2f} {pred['end_time']:<8.2f} "
                    f"{pred['predicted_class']:<6} "
                    f"{pred['response_time_ms']:<10.2f}\n"
                )

        if len(predictions) > 2000:
            self.results_text.insert(
                tk.END,
                f"  ... ({len(predictions) - 2000} more windows)\n"
            )

        # Statistics
        self.results_text.insert(tk.END, "\n" + "=" * 90 + "\n")
        self.results_text.insert(tk.END, "STATISTICS:\n")
        self.results_text.insert(tk.END, "-" * 90 + "\n")

        # Accuracy if ground truth available
        if has_gt:
            correct_preds = [p for p in predictions if p['correct'] is True]
            total_with_labels = [p for p in predictions if p['correct'] is not None]

            if total_with_labels:
                accuracy = len(correct_preds) / len(total_with_labels) * 100
                self.results_text.insert(tk.END, f"\nACCURACY:\n")
                self.results_text.insert(
                    tk.END,
                    f"  Correct: {len(correct_preds)}/"
                    f"{len(total_with_labels)} ({accuracy:.2f}%)\n"
                )
                self.results_text.insert(
                    tk.END,
                    f"  Wrong:   {len(total_with_labels) - len(correct_preds)}\n"
                )

                # Classification report
                y_true = [
                    p['true_class']
                    for p in predictions
                    if p['true_class'] is not None
                ]
                y_pred = [
                    p['predicted_class']
                    for p in predictions
                    if p['true_class'] is not None
                ]

                if len(y_true) > 0:
                    self.results_text.insert(
                        tk.END,
                        f"\nCLASSIFICATION REPORT:\n"
                    )
                    report = classification_report(
                        y_true, y_pred, zero_division=0
                    )
                    self.results_text.insert(tk.END, report + "\n")

        # Class distribution
        pred_classes = [p['predicted_class'] for p in predictions]
        unique_classes, counts = np.unique(pred_classes, return_counts=True)

        self.results_text.insert(
            tk.END,
            f"\nPREDICTED CLASS DISTRIBUTION:\n"
        )
        for cls, count in zip(unique_classes, counts):
            pct = count / len(predictions) * 100
            self.results_text.insert(
                tk.END,
                f"  Class {cls}: {count} windows ({pct:.1f}%)\n"
            )

        # Timing
        if response_times:
            avg_time = np.mean(response_times)
            std_time = np.std(response_times)
            min_time = np.min(response_times)
            max_time = np.max(response_times)

            self.results_text.insert(
                tk.END,
                f"\nRESPONSE TIME (per window):\n"
            )
            self.results_text.insert(
                tk.END,
                f"  Average: {avg_time:.2f} ± {std_time:.2f} ms\n"
            )
            self.results_text.insert(
                tk.END,
                f"  Min:     {min_time:.2f} ms\n"
            )
            self.results_text.insert(
                tk.END,
                f"  Max:     {max_time:.2f} ms\n"
            )

        self.results_text.insert(tk.END, "\n" + "=" * 90 + "\n")
        self.results_text.see(1.0)

    def clear_results(self):
        self.results_text.delete(1.0, tk.END)


# ============================================================================
# MAIN
# ============================================================================

def main():
    root = tk.Tk()
    app = OfflineTestGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
