# -*- coding: utf-8 -*-

"""
SIGNAL PROCESSING AND MACHINE LEARNING 2024-2025 FALL
FINAL : VOICE COMMAND RECOGNITION with IMPROVED DEEP LEARNING MODELS & MACHINE LEARNING
Script 10: Stage 2 -- Real-time Test with GUI


"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import os, time, threading, json, joblib
import numpy as np
import pandas as pd
import librosa
import sounddevice as sd
import tensorflow as tf
import re
from scipy.signal import butter, filtfilt
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, deque

# ============================================================================
# CONFIGURATION & COLORS
# ============================================================================
STUDENT_NO = "231805003"
REPORTS_ROOT = r"E:\signalproc\231805003\models\dl\reports_deep_learning_only"
DEFAULT_RUN_ID = "run_20251210_015930"

TARGET_SR = 16000
WINDOW_SEC = 1.0
HOP_SEC = 0.1
ENERGY_THRESHOLD = 0.02 
MIN_SPEECH_BLOCKS = 3   
MIN_SILENCE_BLOCKS = 6  
FIXED_PAD_SEC = 0.1 

BG_MAIN, BG_PANEL = "#0F0F0F", "#1A1A1A"
ACCENT_BLUE, ACCENT_RED, NEON_GREEN = "#448AFF", "#FF5252", "#00C853"
TEXT_PRIMARY, TEXT_SECONDARY = "#FFFFFF", "#888888"

COMMAND_NAMES_TR = {1: "ışığı aç", 2: "ışığı kapa", 3: "ışığı kıs", 4: "parlaklığı arttır", 5: "parlaklığı azalt", 6: "aydınlatmayı arttır", 7: "aydınlatmayı azalt", 8: "kırmızı ışığı aç", 9: "kırmızı ışığı kapa", 10: "kırmızı ışığı arttır", 11: "kırmızı ışığı azalt", 12: "kırmızı ışığı kıs", 13: "mavi ışığı aç", 14: "mavi ışığı kapa", 15: "mavi ışığı arttır", 16: "mavi ışığı azalt", 17: "mavi ışığı kıs", 18: "yeşil ışığı aç", 19: "yeşil ışığı kapa", 20: "yeşil ışığı arttır", 21: "yeşil ışığı azalt", 22: "yeşil ışığı kıs", 23: "klimayı aç", 24: "klimayı kapa", 25: "iklimlendirmeyi aç", 26: "iklimlendirmeyi kapa", 27: "ısıtmayı aç", 28: "ısıtmayı kapa", 29: "ısıt", 30: "soğut", 31: "sıcaklığı arttır", 32: "sıcaklığı düşür", 33: "evi ısıt", 34: "evi soğut", 35: "odayı ısıt", 36: "odayı soğut", 37: "kombiyi aç", 38: "kombiyi kapa", 39: "fanı aç", 40: "fanı kapa", 41: "fanı arttır", 42: "fanı düşür", 43: "TV aç", 44: "TV kapa", 45: "televizyonu aç", 46: "televizyonu kapa", 47: "multimedyayı aç", 48: "multimedyayı kapa", 49: "müzik aç", 50: "müzik kapa", 51: "panjuru aç", 52: "panjuru kapa", 53: "perdeyi aç", 54: "perdeyi kapa", 55: "alarmı aç", 56: "alarmı kapa", 57: "evet", 58: "hayır", 59: "parti zamanı", 60: "dinlenme zamanı", 61: "uyku zamanı", 62: "Eve Geliyorum", 63: "Evden Çıkıyorum", 64: "Film Zamanı", 65: "Çalışma Zamanı", 66: "Spor Zamanı"}
COMMAND_NAMES_EN = {1: "turn on the light", 2: "turn off the light", 3: "dim the light", 4: "increase brightness", 5: "decrease brightness", 6: "increase lighting", 7: "decrease lighting", 8: "turn on red light", 9: "turn off red light", 10: "increase red light", 11: "decrease red light", 12: "dim red light", 13: "turn on blue light", 14: "turn off blue light", 15: "increase blue light", 16: "decrease blue light", 17: "dim blue light", 18: "turn on green light", 19: "turn off green light", 20: "increase green light", 21: "decrease green light", 22: "dim green light", 23: "turn on the AC", 24: "turn off the AC", 25: "turn on climate control", 26: "turn off climate control", 27: "turn on heating", 28: "turn off heating", 29: "heat", 30: "cool", 31: "increase temperature", 32: "decrease temperature", 33: "heat the house", 34: "cool the house", 35: "heat the room", 36: "cool the room", 37: "turn on the boiler", 38: "turn off the boiler", 39: "turn on the fan", 40: "turn off the fan", 41: "increase fan", 42: "decrease fan", 43: "turn on the TV", 44: "turn off the TV", 45: "turn on the television", 46: "turn off the television", 47: "turn on multimedia", 48: "turn off multimedia", 49: "turn on music", 50: "turn off music", 51: "open the shutter", 52: "close the shutter", 53: "open the curtain", 54: "close the curtain", 55: "turn on the alarm", 56: "turn off the alarm", 57: "yes", 58: "no", 59: "Party Mode", 60: "Relax Mode", 61: "Sleep Mode", 62: "Arriving Home", 63: "I am arriving", 64: "Leaving Home", 65: "I am leaving", 66: "Movie Time", 67: "Work Time", 68: "Workout Time", 69: "Sport Time"}

# ============================================================================
# HELPERS
# ============================================================================
def load_accuracies_from_csv(csv_path):
    tr_acc, en_acc = 0.90, 0.90
    try:
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            tr_v = df[df['lang'] == 'TR']['Accuracy'].values
            en_v = df[df['lang'] == 'EN']['Accuracy'].values
            if len(tr_v) > 0: tr_acc = float(tr_v[0])
            if len(en_v) > 0: en_acc = float(en_v[0])
    except: pass
    return tr_acc, en_acc

TR_ACC_VAL, EN_ACC_VAL = load_accuracies_from_csv(r"results\models_params.csv")

def get_command_text(label_id, lang):
    if label_id == 0: return "Silence"
    d = COMMAND_NAMES_TR if lang == "TR" else COMMAND_NAMES_EN
    return d.get(label_id, f"Unk({label_id})")

def butter_bandpass_filter(y, sr):
    nyq = 0.5 * sr
    b, a = butter(6, [100 / nyq, 6000 / nyq], btype="band")
    return filtfilt(b, a, y).astype(np.float32)

def extract_features(w, sr, f_type):
    if f_type == "MEL":
        S = librosa.feature.melspectrogram(y=w, sr=sr, n_fft=1024, hop_length=512, n_mels=20)
        feat = librosa.power_to_db(S, ref=np.max)
    else:
        feat = librosa.feature.mfcc(y=w, sr=sr, n_mfcc=20, n_fft=1024, hop_length=512)
    if feat.shape[1] > 30: feat = feat[:, :30]
    else: feat = np.pad(feat, ((0,0),(0, 30-feat.shape[1])), mode='constant')
    return feat.flatten().astype(np.float32)

def prepare_input(v, scaler, arch):
    scaled = scaler.transform(v.reshape(1, -1))
    if arch == "CNN_1D": return scaled.reshape(1, scaled.shape[1]//20, 20)
    return scaled.reshape(1, 20, scaled.shape[1]//20, 1)

def train_and_get_w2v(lang="TR"):
    cmd_dict = COMMAND_NAMES_TR if lang == "TR" else COMMAND_NAMES_EN
    sentences = [cmd.lower().split() for cmd in cmd_dict.values()]
    w2v = Word2Vec(sentences, vector_size=50, window=3, min_count=1, seed=42)
    cmd_vecs = {cid: np.mean([w2v.wv[w] for w in txt.lower().split() if w in w2v.wv], axis=0) if any(w in w2v.wv for w in txt.lower().split()) else np.zeros(50) for cid, txt in cmd_dict.items()}
    return w2v, cmd_vecs

# ============================================================================
# DESIGN HELPER
# ============================================================================
class RoundedPanel(tk.Canvas):
    def __init__(self, parent, title="", **kwargs):
        super().__init__(parent, bg=BG_MAIN, highlightthickness=0, **kwargs)
        self.radius, self.title = 20, title
        self.bind("<Configure>", self._draw)
    def _draw(self, e=None):
        self.delete("bg_layer")
        w, h = self.winfo_width(), self.winfo_height()
        points = [5+self.radius, 5, 5+self.radius, 5, w-5-self.radius, 5, w-5-self.radius, 5, w-5, 5, w-5, 5+self.radius, w-5, 5+self.radius, w-5, h-5-self.radius, w-5, h-5-self.radius, w-5, h-5, w-5-self.radius, h-5, w-5-self.radius, h-5, 5+self.radius, h-5, 5+self.radius, h-5, 5, h-5, 5, h-5-self.radius, 5, h-5-self.radius, 5, 5+self.radius, 5, 5+self.radius, 5, 5]
        self.create_polygon(points, smooth=True, fill=BG_PANEL, tags="bg_layer")
        if self.title: self.create_text(20, 20, text=self.title, fill=ACCENT_BLUE, font=("Arial", 10, "bold"), anchor="nw", tags="bg_layer")

# ============================================================================
# GUI CLASS
# ============================================================================
class RealTimeVoiceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("211805048_211805054_231805003 -- Real-Time Microphone")
        self.root.geometry("1600x900")
        self.root.configure(bg=BG_MAIN)
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        # --- COMBOBOX & RADIOBUTTON STYLES ---
        self.style.configure("TCombobox", fieldbackground="white", background="#2A2A2A", foreground="black")
        self.style.map("TCombobox", fieldbackground=[('readonly', 'white')], foreground=[('readonly', 'black')])
        self.style.configure("TRadiobutton", background=BG_PANEL, foreground="white")

        self.current_model, self.is_listening = None, False
        self.audio_buffer = deque(maxlen=TARGET_SR * 3)
        self.current_segment_ids = []
        self.cumulative_word_scores = {}
        self.w2v_model, self.cmd_vectors, self.current_nlp_lang = None, None, None

        self._setup_ui()
        self._scan_runs()
        self._init_nlp("TR")

    def _setup_ui(self):
        self.root.columnconfigure(1, weight=2); self.root.columnconfigure(2, weight=1); self.root.rowconfigure(0, weight=1)
        
        # 1. LEFT PANEL 
        left = tk.Frame(self.root, bg=BG_MAIN, width=420); left.grid(row=0, column=0, sticky="nsew", padx=15, pady=15); left.pack_propagate(False)
        p_set = RoundedPanel(left, title="ENGINE SETTINGS", height=380); p_set.pack(fill=tk.X, pady=(0,15))
        f_set = tk.Frame(p_set, bg=BG_PANEL); p_set.create_window(20, 50, window=f_set, anchor="nw", width=360)
        
        tk.Label(f_set, text="Model Run ID:", bg=BG_PANEL, fg=TEXT_SECONDARY, font=("Arial", 8)).pack(anchor="w")
        self.combo_run = ttk.Combobox(f_set); self.combo_run.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(f_set, text="Inference Logic:", bg=BG_PANEL, fg=TEXT_SECONDARY, font=("Arial", 8)).pack(anchor="w")
        self.mode_var = tk.StringVar(value="CNN")
        ttk.Radiobutton(f_set, text="CNN (Deep Learning)", variable=self.mode_var, value="CNN").pack(anchor="w", pady=2)
        ttk.Radiobutton(f_set, text="Lower + Roof (Machine Learning)", variable=self.mode_var, value="Hybrid").pack(anchor="w", pady=(0, 15))

        tk.Label(f_set, text="Active Language:", bg=BG_PANEL, fg=TEXT_SECONDARY, font=("Arial", 8)).pack(anchor="w")
        f_lang = tk.Frame(f_set, bg=BG_PANEL); f_lang.pack(fill=tk.X, pady=(0, 15))
        self.lang_var = tk.StringVar(value="TR")
        ttk.Radiobutton(f_lang, text="TR", variable=self.lang_var, value="TR", command=lambda: self._init_nlp("TR")).pack(side=tk.LEFT, padx=(0, 20))
        ttk.Radiobutton(f_lang, text="EN", variable=self.lang_var, value="EN", command=lambda: self._init_nlp("EN")).pack(side=tk.LEFT)
        
        tk.Label(f_set, text="Feature Type:", bg=BG_PANEL, fg=TEXT_SECONDARY, font=("Arial", 8)).pack(anchor="w")
        self.combo_feat = ttk.Combobox(f_set, values=["MEL", "MFCC"], state="readonly"); self.combo_feat.current(0); self.combo_feat.pack(fill=tk.X, pady=(0, 10))
        tk.Label(f_set, text="Architecture:", bg=BG_PANEL, fg=TEXT_SECONDARY, font=("Arial", 8)).pack(anchor="w")
        self.combo_arch = ttk.Combobox(f_set, values=["CNN_1D", "CNN_2D"], state="readonly"); self.combo_arch.current(1); self.combo_arch.pack(fill=tk.X)
        
        self.btn_toggle = tk.Button(left, text="🎙️ START LISTENING", bg=ACCENT_BLUE, fg="white", font=("Arial", 10, "bold"), relief="flat", height=2, command=self._toggle_listening)
        self.btn_toggle.pack(fill=tk.X, pady=20)
        self.lbl_status = tk.Label(left, text="STATUS: Idle", bg=BG_MAIN, fg="gray"); self.lbl_status.pack()

        # 2. MIDDLE PANEL 
        p_mid = RoundedPanel(self.root, title="INTELLIGENCE TIMELINE"); p_mid.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        self.txt_output = scrolledtext.ScrolledText(p_mid, bg=BG_PANEL, fg=NEON_GREEN, font=("Consolas", 11), borderwidth=0, highlightthickness=0)
        p_mid.create_window(20, 50, window=self.txt_output, anchor="nw", width=620, height=780)

        # 3. RIGHT PANEL 
        right = tk.Frame(self.root, bg=BG_MAIN, width=480); right.grid(row=0, column=2, sticky="nsew", padx=15, pady=15)
        p_raw = RoundedPanel(right, title="INSTANT NEURAL LOAD", height=240); p_raw.pack(fill=tk.X, pady=(0,15))
        f_raw = tk.Frame(p_raw, bg=BG_PANEL); p_raw.create_window(20, 50, window=f_raw, anchor="nw", width=420)
        self.raw_bars = []
        for i in range(5):
            f = tk.Frame(f_raw, bg=BG_PANEL); f.pack(fill=tk.X, pady=4)
            l = tk.Label(f, text="", width=20, bg=BG_PANEL, fg="white", font=("Consolas", 9), anchor="w"); l.pack(side=tk.LEFT)
            b = ttk.Progressbar(f, length=180); b.pack(side=tk.LEFT, padx=10)
            v = tk.Label(f, text="0%", bg=BG_PANEL, fg=ACCENT_BLUE, font=("Arial", 8, "bold")); v.pack(side=tk.LEFT); self.raw_bars.append((l, b, v))

        p_word = RoundedPanel(right, title="CUMULATIVE WORD RACE"); p_word.pack(fill=tk.BOTH, expand=True, pady=(0,15))
        self.lbl_words = tk.Label(p_word, text="Waiting...", bg=BG_PANEL, fg="white", font=("Consolas", 10), justify=tk.LEFT, anchor="nw")
        p_word.create_window(20, 50, window=self.lbl_words, anchor="nw", width=420, height=350)

        p_final = RoundedPanel(right, title="SYSTEM VERDICT", height=140); p_final.pack(fill=tk.X)
        self.lbl_final = tk.Label(p_final, text="IDLE", bg=BG_PANEL, fg=ACCENT_BLUE, font=("Arial", 14, "bold")); p_final.create_window(20, 55, window=self.lbl_final, anchor="nw", width=420)

    def _analyze_last_window(self):
        if len(self.audio_buffer) < TARGET_SR: return
        start_proc = time.time()
        segment = np.array(list(self.audio_buffer))[-TARGET_SR:]
        # Padding Fix for Filtering
        pad_s = int(FIXED_PAD_SEC * TARGET_SR)
        padded = np.concatenate([np.zeros(pad_s), segment, np.zeros(pad_s)])
        y_norm = librosa.util.normalize(butter_bandpass_filter(padded, TARGET_SR))
        y_proc = y_norm[pad_s : pad_s + TARGET_SR]
        
        f_v = extract_features(y_proc, TARGET_SR, self.combo_feat.get())
        X = prepare_input(f_v, self.current_scaler, self.combo_arch.get())
        probs = self.current_model.predict(X, verbose=0)[0]
        
        current_rt = time.time() - start_proc
        win_id = np.argmax(probs); win_conf = probs[win_id]; win_txt = get_command_text(win_id, self.lang_var.get())
        self.current_segment_ids.append(win_id)
        
        # [PDF SCORE METRIC]
        score = (TR_ACC_VAL * EN_ACC_VAL) / (current_rt * current_rt)
        sc_info = f"RT: {current_rt:.4f}s | Score: {score:.2f}"
        
        ts = time.strftime('%H:%M:%S')
        is_hybrid = (self.mode_var.get() == "Hybrid")
        if is_hybrid:
            self._log_async(f"[{ts}] | Lower model prediction : {win_txt} ({win_conf:.2f})")
        else:
            self._log_async(f"[{ts}] | WIN: {win_txt} ({win_conf:.2f})")

        # BARS UPDATE
        top_idx = probs.argsort()[-5:][::-1]
        top_preds = [(get_command_text(i, self.lang_var.get()), probs[i]) for i in top_idx]
        
        # WORD RACE
        for rank, idx in enumerate(top_idx):
            if idx == 0: continue
            txt = get_command_text(idx, self.lang_var.get()); weight = 2.0 if rank == 0 else 1.0
            for w in txt.split('(')[0].strip().lower().split(): self.cumulative_word_scores[w] = self.cumulative_word_scores.get(w, 0.0) + (probs[idx] * weight)

        sw = sorted(self.cumulative_word_scores.items(), key=lambda x: x[1], reverse=True)
        best_t, best_s = "-", -1.0
        cmd_dict = COMMAND_NAMES_TR if self.lang_var.get()=="TR" else COMMAND_NAMES_EN
        for cid, text in cmd_dict.items():
            words = text.lower().split(); s = sum(self.cumulative_word_scores.get(w, 0.0) for w in words)
            if all(w in self.cumulative_word_scores for w in words): s *= 1.5
            avg_s = s / len(words) if words else 0
            if avg_s > best_s: best_s, best_t = avg_s, text
            
        self.root.after(0, lambda: self._update_live_panel(top_preds, sw, best_t, best_s, sc_info))

    def _log(self, msg):
        self.txt_output.config(state='normal')
        self.txt_output.insert(tk.END, msg + "\n")
        # --- AUTO-SCROLL FIX ---
        self.txt_output.see(tk.END)
        self.txt_output.config(state='disabled')

    def _update_live_panel(self, tp, ws, bt, bs, sc):
        
        for i, (t, c) in enumerate(tp):
            if i >= 5: break
            self.raw_bars[i][0].config(text=t[:20]); self.raw_bars[i][1]['value']=c*100; self.raw_bars[i][2].config(text=f"{int(c*100)}%")
        ms = ws[0][1] if ws else 1.0
        td = f"{'WORD':<15} | {'SCORE'}\n" + "-"*30 + "\n"
        for w, s in ws[:8]: td += f"{w.upper():<15} | {s:.2f} {'█' * int((s/ms)*10)}\n"
        self.lbl_words.config(text=td)
        if self.mode_var.get() == "CNN": self.lbl_final.config(text=f"CNN Stable: {bt}\n{sc}", fg=ACCENT_BLUE)

    def _finalize_segment_decision(self):
        valid = [i for i in self.current_segment_ids if i != 0]
        if valid and self.mode_var.get() == "Hybrid":
            final_res = get_command_text(Counter(valid).most_common(1)[0][0], self.lang_var.get())
            self.root.after(0, lambda: self.lbl_final.config(text=f"Lower+Roof: {final_res}", fg=ACCENT_RED))

    def _toggle_listening(self):
        if not self.is_listening:
            if self._load_model():
                self.is_listening = True; self.btn_toggle.config(text="⏹️ STOP", bg="#333")
                self.lbl_status.config(text="LISTENING...", fg="red")
                threading.Thread(target=self._mic_loop, daemon=True).start()
        else: self.is_listening = False; self.btn_toggle.config(text="🎙️ START", bg=ACCENT_BLUE)

    def _mic_loop(self):
        block = int(TARGET_SR * 0.1); speech, silence, in_seg = 0, 0, False
        with sd.InputStream(samplerate=TARGET_SR, channels=1, blocksize=block) as stream:
            while self.is_listening:
                data, _ = stream.read(block); buf = data[:, 0]; rms = np.sqrt(np.mean(buf**2))
                for s in buf: self.audio_buffer.append(s)
                if rms > ENERGY_THRESHOLD:
                    speech += 1; silence = 0
                    if speech >= MIN_SPEECH_BLOCKS and not in_seg:
                        in_seg = True; self._log_async(">>> [Speech Started]"); self.cumulative_word_scores = {}; self.current_segment_ids = []
                else:
                    silence += 1
                    if silence >= MIN_SILENCE_BLOCKS and in_seg:
                        in_seg, speech = False, 0; self._log_async("<<< [Speech Ended]"); self._finalize_segment_decision()
                if in_seg: self._analyze_last_window()
                time.sleep(0.01)

    def _log_async(self, m): self.root.after(0, lambda: self._log(m))
    def _scan_runs(self):
        if os.path.exists(REPORTS_ROOT):
            r = sorted([d for d in os.listdir(REPORTS_ROOT) if os.path.isdir(os.path.join(REPORTS_ROOT, d))], reverse=True)
            self.combo_run['values'] = r
            if DEFAULT_RUN_ID in r: self.combo_run.set(DEFAULT_RUN_ID)
    def _init_nlp(self, l): self.w2v_model, self.cmd_vectors = train_and_get_w2v(l); self.current_nlp_lang = l
    def _load_model(self):
        try:
            path = os.path.join(REPORTS_ROOT, self.combo_run.get(), self.lang_var.get(), self.combo_feat.get())
            m = f"model_{self.combo_arch.get().lower().replace('_','')}_{self.lang_var.get()}_{self.combo_feat.get()}.h5"
            self.current_model = tf.keras.models.load_model(os.path.join(path, m))
            self.current_scaler = joblib.load(os.path.join(path, f"scaler_{self.lang_var.get()}_{self.combo_feat.get()}.joblib"))
            self.current_le = joblib.load(os.path.join(path, f"label_encoder_{self.lang_var.get()}_{self.combo_feat.get()}.joblib"))
            return True
        except Exception as e: self._log(f"❌ Error: {e}"); return False

if __name__ == "__main__":
    root = tk.Tk(); app = RealTimeVoiceGUI(root); root.mainloop()