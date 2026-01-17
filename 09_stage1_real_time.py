# -*- coding: utf-8 -*-
"""
SIGNAL PROCESSING AND MACHINE LEARNING 2024-2025 FALL
FINAL : VOICE COMMAND RECOGNITION with IMPROVED DEEP LEARNING MODELS & MACHINE LEARNING
Script 09: Stage 1 -- Real-time Test with GUI

"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import sounddevice as sd
import numpy as np
import librosa
import time
import os
import json
import joblib
import pandas as pd
import queue
import threading
from collections import deque, Counter
from scipy import stats, signal
from scipy.fft import fft, fftfreq
from scipy.ndimage import gaussian_filter1d

# =====================================================================
# CONFIG
# =====================================================================
SAMPLE_RATE = 16000
FRAME_SIZE_MS = 25
FRAME_HOP_MS = 10
BUFFER_SECONDS = 10
STUDENT_NO = "231805003"

# ENERGY THRESHOLDS
ENERGY_THRESHOLD = 0.01
MIN_SPEECH_FRAMES = 4  # Min .4 sec frames
MIN_SILENCE_FRAMES = 6  # 6 frame silence command ends
FREQ_BANDS = [
    (0, 200), (200, 500), (500, 1000), (1000, 2000),
    (2000, 3000), (3000, 4000), (4000, 6000), (6000, 8000)
]

# =====================================================================
# COMMAND NAME MAPS
# =====================================================================

# Commands - TR
COMMAND_NAMES_TR = {
    1: "ışığı aç",
    2: "ışığı kapa",
    3: "ışığı kıs",
    4: "parlaklığı arttır",
    5: "parlaklığı azalt",
    6: "aydınlatmayı arttır",
    7: "aydınlatmayı azalt",
    8: "kırmızı ışığı aç",
    9: "kırmızı ışığı kapa",
    10: "kırmızı ışığı arttır",
    11: "kırmızı ışığı azalt",
    12: "kırmızı ışığı kıs",
    13: "mavi ışığı aç",
    14: "mavi ışığı kapa",
    15: "mavi ışığı arttır",
    16: "mavi ışığı azalt",
    17: "mavi ışığı kıs",
    18: "yeşil ışığı aç",
    19: "yeşil ışığı kapa",
    20: "yeşil ışığı arttır",
    21: "yeşil ışığı azalt",
    22: "yeşil ışığı kıs",
    23: "klimayı aç",
    24: "klimayı kapa",
    25: "iklimlendirmeyi aç",
    26: "iklimlendirmeyi kapa",
    27: "ısıtmayı aç",
    28: "ısıtmayı kapa",
    29: "ısıt",
    30: "soğut",
    31: "sıcaklığı arttır",
    32: "sıcaklığı düşür",
    33: "evi ısıt",
    34: "evi soğut",
    35: "odayı ısıt",
    36: "odayı soğut",
    37: "kombiyi aç",
    38: "kombiyi kapa",
    39: "fanı aç",
    40: "fanı kapa",
    41: "fanı arttır",
    42: "fanı düşür",
    43: "TV aç",
    44: "TV kapa",
    45: "televizyonu aç",
    46: "televizyonu kapa",
    47: "multimedyayı aç",
    48: "multimedyayı kapa",
    49: "müzik aç",
    50: "müzik kapa",
    51: "panjuru aç",
    52: "panjuru kapa",
    53: "perdeyi aç",
    54: "perdeyi kapa",
    55: "alarmı aç",
    56: "alarmı kapa",
    57: "evet",
    58: "hayır",
    59: "parti zamanı",
    60: "dinlenme zamanı",
    61: "uyku zamanı",
    62: "Eve Geliyorum",
    63: "Evden Çıkıyorum",
    64: "Film Zamanı",
    65: "Çalışma Zamanı",
    66: "Spor Zamanı"
}

# Commands - EN
COMMAND_NAMES_EN = {
    1: "turn on the light",
    2: "turn off the light",
    3: "dim the light",
    4: "increase brightness",
    5: "decrease brightness",
    6: "increase lighting",
    7: "decrease lighting",
    8: "turn on red light",
    9: "turn off red light",
    10: "increase red light",
    11: "decrease red light",
    12: "dim red light",
    13: "turn on blue light",
    14: "turn off blue light",
    15: "increase blue light",
    16: "decrease blue light",
    17: "dim blue light",
    18: "turn on green light",
    19: "turn off green light",
    20: "increase green light",
    21: "decrease green light",
    22: "dim green light",
    23: "turn on the AC",
    24: "turn off the AC",
    25: "turn on climate control",
    26: "turn off climate control",
    27: "turn on heating",
    28: "turn off heating",
    29: "heat",
    30: "cool",
    31: "increase temperature",
    32: "decrease temperature",
    33: "heat the house",
    34: "cool the house",
    35: "heat the room",
    36: "cool the room",
    37: "turn on the boiler",
    38: "turn off the boiler",
    39: "turn on the fan",
    40: "turn off the fan",
    41: "increase fan",
    42: "decrease fan",
    43: "turn on the TV",
    44: "turn off the TV",
    45: "turn on the television",
    46: "turn off the television",
    47: "turn on multimedia",
    48: "turn off multimedia",
    49: "turn on music",
    50: "turn off music",
    51: "open the shutter",
    52: "close the shutter",
    53: "open the curtain",
    54: "close the curtain",
    55: "turn on the alarm",
    56: "turn off the alarm",
    57: "yes",
    58: "no",
    59: "Party Mode",
    60: "Relax Mode",
    61: "Sleep Mode",
    62: "Arriving Home",
    63: "I am arriving",
    64: "Leaving Home",
    65: "I am leaving",
    66: "Movie Time",
    67: "Work Time",
    68: "Workout Time",
    69: "Sport Time"
}

# =====================================================================
# FEATURE EXTRACTION
# =====================================================================

def extract_time_features(y):
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
    features['time_snr_estimate'] = 20 * np.log10((signal_level + 1e-10)/(noise_floor+1e-10))

    zero_crossings = np.where(np.diff(np.signbit(y)))[0]
    features['time_zcr'] = len(zero_crossings) / len(y)

    analytic = signal.hilbert(y)
    envelope = np.abs(analytic)
    features['time_envelope_mean'] = np.mean(envelope)
    features['time_envelope_std'] = np.std(envelope)
    features['time_envelope_max'] = np.max(envelope)
    features['time_envelope_min'] = np.min(envelope)

    features['time_dynamic_range_db'] = 20 * np.log10(
        (np.max(np.abs(y))+1e-10) / (np.mean(np.abs(y))+1e-10)
    )

    return features


def extract_freq_features(y, sr):
    features = {}
    n = len(y)
    Y = fft(y)
    freqs = fftfreq(n, 1/sr)
    mag = np.abs(Y[:n//2])
    freqs = freqs[:n//2]
    mag_s = gaussian_filter1d(mag, sigma=5)

    for i, (low, high) in enumerate(FREQ_BANDS):
        mask = (freqs >= low) & (freqs < high)
        band_mag = mag_s[mask]
        band_freqs = freqs[mask]

        if len(band_mag) > 0:
            features[f'freq_band_{i+1}_power'] = np.sum(band_mag**2)/len(band_mag)
            peak_idx = np.argmax(band_mag)
            features[f'freq_band_{i+1}_peak_freq'] = band_freqs[peak_idx]
            features[f'freq_band_{i+1}_peak_power'] = band_mag[peak_idx]
            features[f'freq_band_{i+1}_spread'] = np.std(band_mag)
            features[f'freq_band_{i+1}_mean_freq'] = np.mean(band_freqs)
        else:
            features[f'freq_band_{i+1}_power'] = 0
            features[f'freq_band_{i+1}_peak_freq'] = 0
            features[f'freq_band_{i+1}_peak_power'] = 0
            features[f'freq_band_{i+1}_spread'] = 0
            features[f'freq_band_{i+1}_mean_freq'] = 0

    features['freq_spectral_centroid'] = (
        np.sum(freqs * mag_s) / (np.sum(mag_s)+1e-10)
    )

    cs = np.cumsum(mag_s)
    total = cs[-1]
    if total > 0:
        r85 = np.where(cs >= 0.85*total)[0]
        r95 = np.where(cs >= 0.95*total)[0]
        features['freq_spectral_rolloff_85'] = freqs[r85[0]] if len(r85)>0 else 0
        features['freq_spectral_rolloff_95'] = freqs[r95[0]] if len(r95)>0 else 0
    else:
        features['freq_spectral_rolloff_85'] = 0
        features['freq_spectral_rolloff_95'] = 0

    features['freq_spectral_flux'] = np.std(mag_s)
    g_mean = np.exp(np.mean(np.log(mag_s + 1e-10)))
    a_mean = np.mean(mag_s)
    features['freq_spectral_flatness'] = g_mean / (a_mean + 1e-10)

    prob = mag_s / (np.sum(mag_s) + 1e-10)
    features['freq_spectral_entropy'] = -np.sum(prob * np.log2(prob+1e-10))

    return features


def extract_mfcc_features(y, sr):
    feats = {}
    n_fft = int(FRAME_SIZE_MS * sr / 1000)
    hop = int(FRAME_HOP_MS * sr / 1000)

    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop
    )

    for i in range(13):
        feats[f'mfcc_{i+1}_mean'] = np.mean(mfcc[i])
        feats[f'mfcc_{i+1}_std'] = np.std(mfcc[i])
        feats[f'mfcc_{i+1}_median'] = np.median(mfcc[i])
        feats[f'mfcc_{i+1}_min'] = np.min(mfcc[i])
        feats[f'mfcc_{i+1}_max'] = np.max(mfcc[i])
        feats[f'mfcc_{i+1}_range'] = np.max(mfcc[i])-np.min(mfcc[i])

    mfcc_d = librosa.feature.delta(mfcc[:5])
    for i in range(5):
        feats[f'mfcc_delta_{i+1}_mean'] = np.mean(mfcc_d[i])
        feats[f'mfcc_delta_{i+1}_std'] = np.std(mfcc_d[i])

    mfcc_d2 = librosa.feature.delta(mfcc[:3], order=2)
    for i in range(3):
        feats[f'mfcc_delta2_{i+1}_mean'] = np.mean(mfcc_d2[i])
        feats[f'mfcc_delta2_{i+1}_std'] = np.std(mfcc_d2[i])

    feats['mfcc_energy_mean'] = np.mean(mfcc[0])
    feats['mfcc_spectral_flux'] = np.mean(
        np.sqrt(np.sum(np.diff(mfcc, axis=1)**2, axis=0))
    )

    return feats


def extract_mel_features(y, sr):
    feats = {}
    n_fft = int(FRAME_SIZE_MS * sr / 1000)
    hop = int(FRAME_HOP_MS * sr / 1000)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=40
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    for i in range(40):
        feats[f'mel_{i+1}_mean'] = np.mean(mel_db[i])
        feats[f'mel_{i+1}_std'] = np.std(mel_db[i])
        feats[f'mel_{i+1}_median'] = np.median(mel_db[i])
        feats[f'mel_{i+1}_max'] = np.max(mel_db[i])
        feats[f'mel_{i+1}_min'] = np.min(mel_db[i])

    feats['mel_total_energy'] = np.sum(mel)
    feats['mel_mean_energy'] = np.mean(mel)
    feats['mel_std_energy'] = np.std(mel)

    mel_freqs = librosa.mel_frequencies(n_mels=40, fmin=0, fmax=sr/2)
    centroid = np.sum(
        mel_freqs[:, None] * mel, axis=0
    ) / (np.sum(mel, axis=0) + 1e-10)
    feats['mel_spectral_centroid_mean'] = np.mean(centroid)
    feats['mel_spectral_centroid_std'] = np.std(centroid)

    mel_flux = np.sqrt(np.sum(np.diff(mel_db, axis=1)**2, axis=0))
    feats['mel_flux_mean'] = np.mean(mel_flux)
    feats['mel_flux_std'] = np.std(mel_flux)

    return feats


def extract_features_window(y, sr, fam):
    if fam == "time_freq":
        return {**extract_time_features(y), **extract_freq_features(y, sr)}
    elif fam == "mfcc":
        return extract_mfcc_features(y, sr)
    elif fam == "mel":
        return extract_mel_features(y, sr)
    else:
        raise ValueError(f"Unknown family: {fam}")


def trim_silence(audio, threshold=0.003, frame_ms=50):
    """
    Basit RMS tabanlı trim:
    - audio: 1D numpy array
    - threshold: RMS eşiği
    - frame_ms: analiz frame süresi (ms)
    """
    frame_len = int(SAMPLE_RATE * frame_ms / 1000)
    if len(audio) < frame_len:
        return audio

    rms_values = []
    for i in range(0, len(audio) - frame_len + 1, frame_len):
        frame = audio[i:i + frame_len]
        rms = np.sqrt(np.mean(frame ** 2))
        rms_values.append(rms)

    rms_values = np.array(rms_values)

    speech_idx = np.where(rms_values > threshold)[0]
    if len(speech_idx) == 0:
        return audio

    start_frame = speech_idx[0]
    end_frame = speech_idx[-1] + 1

    start = start_frame * frame_len
    end = min(len(audio), end_frame * frame_len)

    return audio[start:end]


# =====================================================================
# GUI CLASS
# =====================================================================

class RealTimeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("211805048_211805054_231805003 -- STAGE 1 Real-Time Voice Command Recognition")
        self.root.geometry("1600x950")

        self.model = None
        self.scaler = None
        self.feature_names = None
        self.feature_family = None
        self.window_length_s = None
        self.hop_s = 0.1
        self.language = None

        # Audio buffer
        self.buffer = deque(maxlen=int(BUFFER_SECONDS * SAMPLE_RATE))
        self.lock = threading.Lock()

        self.running = False
        self.energy_threshold = ENERGY_THRESHOLD

        # İstatistikler
        self.total_segments = 0
        self.total_predictions = 0

        self.selected_device = None
        
        # Segment tracking
        self.current_segment = []
        self.silence_frames = 0
        self.speech_frames = 0
        
        # Background log throttling
        self.last_bg_log_time = 0.0
        
        self._build_gui()
        self._discover_models()

    # -----------------------------------------------------------------
    # Helper: Label -> Command Text
    # -----------------------------------------------------------------
    def _get_command_name(self, label: int) -> str:
        """Return human-readable command name for given label."""
        if label == 0:
            return "background / silence"
        if self.language == 'TR':
            return COMMAND_NAMES_TR.get(label, f"Label {label}")
        elif self.language == 'EN':
            return COMMAND_NAMES_EN.get(label, f"Label {label}")
        else:
            return f"Label {label}"

    def _list_audio_devices(self):
        devices = sd.query_devices()
        input_devices = []

        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                name = f"[{i}] {dev['name']} ({dev['max_input_channels']} ch)"
                input_devices.append((i, name))

        if not input_devices:
            messagebox.showerror("Error", "No input devices found!")
            return

        try:
            default_input = sd.query_devices(kind='input')
            default_idx = default_input['index']
        except Exception:
            default_idx = input_devices[0][0]

        device_names = [name for _, name in input_devices]
        self.mic_combo['values'] = device_names

        for idx, (dev_id, name) in enumerate(input_devices):
            if dev_id == default_idx:
                self.mic_combo.current(idx)
                self.selected_device = dev_id
                break

        if self.selected_device is None and input_devices:
            self.mic_combo.current(0)
            self.selected_device = input_devices[0][0]

    def _get_selected_device(self):
        selection = self.mic_combo.get()
        if not selection:
            return None
        try:
            device_id = int(selection.split(']')[0].split('[')[1])
            return device_id
        except Exception:
            return None

    def _build_gui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)
        main.rowconfigure(1, weight=0)

        # LEFT PANEL
        left = ttk.LabelFrame(main, text="🤖 Model & Microphone", padding=10)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        left.columnconfigure(0, weight=1)

        
        ttk.Label(left, text="Microphone:").grid(row=0, column=0, pady=5, sticky="w")
        self.mic_combo = ttk.Combobox(left, width=65, state='readonly')
        self.mic_combo.grid(row=1, column=0, pady=5, sticky="ew")

        ttk.Separator(left, orient='horizontal').grid(row=2, column=0, sticky='ew', pady=10)

        # Model Selection
        ttk.Label(left, text="Select model:").grid(row=3, column=0, pady=5, sticky="w")
        self.model_combo = ttk.Combobox(left, width=65, state='readonly')
        self.model_combo.grid(row=4, column=0, pady=5, sticky="ew")
        self.model_combo.bind("<<ComboboxSelected>>", self._on_model_chosen)

        best_frame = ttk.Frame(left)
        best_frame.grid(row=5, column=0, pady=5, sticky="ew")
        ttk.Button(best_frame, text="⭐ Best TR", command=lambda: self._select_best('TR')).pack(
            side="left", expand=True, fill="x", padx=2
        )
        ttk.Button(best_frame, text="⭐ Best EN", command=lambda: self._select_best('EN')).pack(
            side="left", expand=True, fill="x", padx=2
        )

        ttk.Label(left, text="Model Info:").grid(row=6, column=0, pady=(10, 0), sticky="w")
        self.model_info = tk.Text(left, width=65, height=14, font=("Courier", 9), state='disabled')
        self.model_info.grid(row=7, column=0, pady=5, sticky="ew")

        control_frame = ttk.Frame(left)
        control_frame.grid(row=8, column=0, pady=(10, 0), sticky="ew")

        self.start_btn = ttk.Button(control_frame, text="▶ Start", command=self._start_rt, state='disabled')
        self.start_btn.pack(side="left", padx=3)

        self.stop_btn = ttk.Button(control_frame, text="⏹ Stop", command=self._stop_rt, state='disabled')
        self.stop_btn.pack(side="left", padx=3)

        ttk.Button(control_frame, text="🗑 Clear", command=self._clear_output).pack(
            side="left", padx=3
        )

        
        self._list_audio_devices()

        # RIGHT PANEL
        right = ttk.LabelFrame(main, text="🎤 Real-Time Output", padding=10)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        self.output = scrolledtext.ScrolledText(right, font=("Courier", 11), wrap=tk.WORD)
        self.output.grid(row=0, column=0, sticky="nsew")

        stats_frame = ttk.LabelFrame(main, text="Statistics", padding=10)
        stats_frame.grid(row=1, column=1, sticky="ew", pady=(10, 0))

        self.stats_text = tk.StringVar(value="Waiting to start...")
        stats_label = ttk.Label(stats_frame, textvariable=self.stats_text, font=("Courier", 10))
        stats_label.pack()

    def _clear_output(self):
        self.output.delete(1.0, tk.END)
        self.total_segments = 0
        self.total_predictions = 0
        self._update_stats()

    def _discover_models(self):
        from glob import glob
        self.model_files = glob("./lower_models/*_best_model.joblib")
        names = [os.path.basename(p) for p in self.model_files]
        self.model_combo['values'] = names

    def _on_model_chosen(self, event):
        idx = self.model_combo.current()
        if idx < 0:
            return
        path = self.model_files[idx]
        self._load_model(path)

    def _load_model(self, model_path):
        try:
            self.model = joblib.load(model_path)

            params_path = model_path.replace("_best_model.joblib", "_params.json")
            with open(params_path, 'r') as f:
                prm = json.load(f)

            self.feature_family = prm['feature_family']
            self.feature_names = prm['feature_names']
            self.language = prm.get('language', 'N/A')
            self.window_length_s = float(prm['window_length'].replace("s", "").replace("p", "."))

            scaler_path = model_path.replace("_best_model.joblib", "_scaler.joblib")
            self.scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

            txt = (
                f"Model: {prm.get('best_model','Unknown')}\n"
                f"Language: {self.language}\n"
                f"Features: {self.feature_family}\n"
                f"Window: {self.window_length_s:.2f}s\n"
                f"CV Score: {prm.get('cv_score',0):.4f}\n"
            )

            self.model_info.config(state='normal')
            self.model_info.delete(1.0, tk.END)
            self.model_info.insert(1.0, txt)
            self.model_info.config(state='disabled')

            self.start_btn.config(state='normal')

        except Exception as e:
            messagebox.showerror("Model Error", str(e))

    def _select_best(self, lang):
        best = None
        best_score = -1

        for p in self.model_files:
            prm_path = p.replace("_best_model.joblib", "_params.json")
            with open(prm_path, 'r') as f:
                prm = json.load(f)

            if prm.get('language') != lang:
                continue

            if prm.get('cv_score', 0) > best_score:
                best_score = prm['cv_score']
                best = p

        if best is None:
            messagebox.showwarning("Warning", f"No {lang} model found.")
            return

        idx = self.model_files.index(best)
        self.model_combo.current(idx)
        self._load_model(best)

        messagebox.showinfo("Selected", f"Best {lang} model!\nCV: {best_score:.4f}")

    def _start_rt(self):
        if self.model is None:
            return

        device_idx = self._get_selected_device()
        if device_idx is None:
            messagebox.showerror("Error", "Select microphone!")
            return

        self.running = True
        self.output.delete(1.0, tk.END)
        self.total_segments = 0
        self.total_predictions = 0
        self.buffer.clear()
        self.current_segment = []
        self.silence_frames = 0
        self.speech_frames = 0
        self.last_bg_log_time = 0.0

        try:
            blocksize = int(0.1 * SAMPLE_RATE)

            self.stream = sd.InputStream(
                channels=1,
                samplerate=SAMPLE_RATE,
                blocksize=blocksize,
                device=device_idx,
                callback=self._audio_callback,
                dtype=np.float32
            )
            self.stream.start()

            threading.Thread(target=self._segment_loop, daemon=True).start()

            self.start_btn.config(state='disabled')
            self.stop_btn.config(state='normal')

            self._write("Real-time started (SEGMENT MODE)\n")
            self._write(f"Device: {self.mic_combo.get()}\n")
            self._write(f"Threshold: {self.energy_threshold:.4f}\n")
            self._write(f"Min speech: {MIN_SPEECH_FRAMES * 0.1:.1f}s\n")
            self._write(f"Silence gap: {MIN_SILENCE_FRAMES * 0.1:.1f}s\n")
            self._write("-" * 60 + "\n")

        except Exception as e:
            self.running = False
            messagebox.showerror("Audio Error", str(e))

    def _stop_rt(self):
        self.running = False
        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass

        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')

        self._write("\n" + "=" * 60 + "\n")
        self._write("🛑 Stopped.\n")
        self._write(f"📊 Segments: {self.total_segments}\n")
        self._write(f"🎯 Predictions: {self.total_predictions}\n")

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"⚠️ Status: {status}")

        try:
            data = indata[:, 0].copy()
        except IndexError:
            data = indata.flatten()

        with self.lock:
            self.buffer.extend(data)

    def _segment_loop(self):
        """
        SEGMENT-BASED RECOGNITION:
        1. Check RMS every 100ms frame
        2. Speech started -> add to segment
        3. Enough silence -> process segment
        4. Prediction with sliding window over segment
        + If no segment background log flow.
        """
        
        frame_size = int(0.1 * SAMPLE_RATE)  # 100ms frames
        
        self._write("Listening for commands...\n\n")

        while self.running:
            time.sleep(0.1)  # 100ms check interval

            with self.lock:
                if len(self.buffer) < frame_size:
                    continue
                frame = np.array(list(self.buffer)[-frame_size:])

            rms = np.sqrt(np.mean(frame ** 2))
            is_speech = rms > self.energy_threshold
            ts = time.strftime('%H:%M:%S')

            # Segment yokken (speech_frames == 0) background logu aksın
            if not is_speech and self.speech_frames == 0:
                if time.time() - self.last_bg_log_time > 0.3:
                    self._write_async(
                        f"[{ts}] Background (RMS: {rms:.4f}, Thr: {self.energy_threshold:.4f})\n"
                    )
                    self.last_bg_log_time = time.time()

            if is_speech:
                self.speech_frames += 1
                self.silence_frames = 0
                self.current_segment.extend(frame)
                
                if self.speech_frames == 1:
                    self._write_async(f"[{ts}] Speech started...\n")
                
            else:
                self.silence_frames += 1
                
                if self.speech_frames > 0:
                    self.current_segment.extend(frame)

                if self.silence_frames >= MIN_SILENCE_FRAMES and self.speech_frames >= MIN_SPEECH_FRAMES:
                    segment_audio = np.array(self.current_segment)
                    segment_duration = len(segment_audio) / SAMPLE_RATE
                    
                    if segment_duration < 0.25:
                        self._write_async(
                            f"[{ts}] ⚠️ Segment too short "
                            f"({segment_duration:.2f}s) → ignored\n"
                        )
                    elif segment_duration > 3.0:
                        self._write_async(
                            f"[{ts}] ⚠️ Segment too long "
                            f"({segment_duration:.2f}s) → ignored\n"
                        )
                    else:
                        self._write_async(
                            f"[{ts}] Segment captured "
                            f"({segment_duration:.2f}s, {len(segment_audio)} samples)\n"
                        )
                        self._process_segment(segment_audio)
                        self.total_segments += 1
                        self._update_stats_async()

                    self.current_segment = []
                    self.speech_frames = 0
                    self.silence_frames = 0

    def _process_segment(self, segment_audio):
        """
        Make predictions on the segment with a sliding window.
        Automatically saves prediction results in 'logs' folder.
        """
        try:
            # 1. LOGLAMA HAZIRLIĞI
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            # Real-time olduğu için her oturuma zaman damgalı bir log dosyası oluşturur
            log_filename = os.path.join(log_dir, f"realtime_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")

            segment_audio = trim_silence(segment_audio, threshold=0.003, frame_ms=50)

            window_samples = int(self.window_length_s * SAMPLE_RATE)
            hop_samples = int(0.5 * SAMPLE_RATE)  # 50ms hop

            if len(segment_audio) < window_samples:
                pad_len = window_samples - len(segment_audio)
                segment_audio = np.pad(segment_audio, (0, pad_len), mode='constant')

            predictions = []
            probabilities = []

            # We open the file in append mode
            with open(log_filename, "a", encoding="utf-8") as f_log:
                f_log.write(f"\n--- Segment Capture Started: {time.strftime('%H:%M:%S')} ---\n")

                for start in range(0, len(segment_audio) - window_samples + 1, hop_samples):
                    window = segment_audio[start:start + window_samples]
                    window_pre = librosa.effects.preemphasis(window)

                    feats = extract_features_window(window_pre, SAMPLE_RATE, self.feature_family)
                    X = pd.DataFrame([feats], dtype=np.float32)

                    for f in self.feature_names:
                        if f not in X.columns:
                            X[f] = 0.0
                    X = X[self.feature_names]

                    if self.scaler is not None:
                        X = self.scaler.transform(X)

                    if hasattr(self.model, "predict_proba"):
                        proba = self.model.predict_proba(X)[0]
                        pred = int(np.argmax(proba))
                        max_prob = float(np.max(proba))
                    else:
                        pred = int(self.model.predict(X)[0])
                        max_prob = 1.0

                    predictions.append(pred)
                    probabilities.append(max_prob)

                    # 2. PDF STEP 4 FORMATINDA LOGLAMA
                    t_curr = start / SAMPLE_RATE
                    cmd_name = self._get_command_name(pred)
                    # Format: t=0.80s | PRED=1 (turn on light) | p=0.88
                    log_line = f"t={t_curr:5.2f}s | PRED={pred:2d} ({cmd_name:20s}) | p={max_prob:.2f}"
                    
                    self._write_async("  " + log_line + "\n") 
                    f_log.write(log_line + "\n") # write to file in logs/

            # ---MAJORITY VOTE AND FINAL DECISION (Continuation of current logic) ---
            if len(predictions) == 0:
                self._write_async("  No predictions from segment\n\n")
                return

            pred_counts = Counter(predictions)
            non_bg_preds = [p for p in predictions if p != 0]

            if len(non_bg_preds) >= len(predictions) * 0.3:
                final_pred = max(set(non_bg_preds), key=non_bg_preds.count)
                confidence = non_bg_preds.count(final_pred) / len(non_bg_preds) * 100
                avg_prob = np.mean([probabilities[i] for i, p in enumerate(predictions) if p == final_pred])
                cmd_final = self._get_command_name(final_pred)

                self._write_async(f"  ✅ FINAL DECISION: #{final_pred} - {cmd_final} (Conf: {confidence:.0f}%)\n\n")
                self.total_predictions += 1
            else:
                self._write_async(f"  ⚪ Background/Noise detected.\n\n")

        except Exception as e:
            self._write_async(f"  ❌ Error: {str(e)}\n\n")

    def _update_stats(self):
        stats = (
            f"Segments captured: {self.total_segments} | "
            f"Commands detected: {self.total_predictions} | "
            f"Threshold: {self.energy_threshold:.4f}"
        )
        self.stats_text.set(stats)

    def _update_stats_async(self):
        self.root.after(0, self._update_stats)

    def _write(self, txt):
        self.output.insert(tk.END, txt)
        self.output.see(tk.END)

    def _write_async(self, txt):
        self.root.after(0, lambda: self._write(txt))


# =====================================================================
# MAIN
# =====================================================================

def main():
    root = tk.Tk()
    app = RealTimeGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()