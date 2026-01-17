# -*- coding: utf-8 -*-

"""
SIGNAL PROCESSING AND MACHINE LEARNING 2024-2025 FALL
FINAL : VOICE COMMAND RECOGNITION with IMPROVED DEEP LEARNING MODELS & MACHINE LEARNING
Script 10: Stage 2 -- Logger for Bilingual Models with Lower Models

"""

import os, glob, json, joblib, librosa, numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# =====================================================================
# CONFIG
# =====================================================================

STUDENT_NO = "231805003"
SAMPLE_RATE = 16000 
NOISY_DATA_ROOT = "Noisy_16kHz_Padded_Segments" 

COMMAND_NAMES_TR = {1: "ışığı aç", 2: "ışığı kapa", 3: "ışığı kıs", 4: "parlaklığı arttır", 5: "parlaklığı azalt", 6: "aydınlatmayı arttır", 7: "aydınlatmayı azalt", 8: "kırmızı ışığı aç", 9: "kırmızı ışığı kapa", 10: "kırmızı ışığı arttır", 11: "kırmızı ışığı azalt", 12: "kırmızı ışığı kıs", 13: "mavi ışığı aç", 14: "mavi ışığı kapa", 15: "mavi ışığı arttır", 16: "mavi ışığı azalt", 17: "mavi ışığı kıs", 18: "yeşil ışığı aç", 19: "yeşil ışığı kapa", 20: "yeşil ışığı arttır", 21: "yeşil ışığı azalt", 22: "yeşil ışığı kıs", 23: "klimayı aç", 24: "klimayı kapa", 25: "iklimlendirmeyi aç", 26: "iklimlendirmeyi kapa", 27: "ısıtmayı aç", 28: "ısıtmayı kapa", 29: "ısıt", 30: "soğut", 31: "sıcaklığı arttır", 32: "sıcaklığı düşür", 33: "evi ısıt", 34: "evi soğut", 35: "odayı ısıt", 36: "odayı soğut", 37: "kombiyi aç", 38: "kombiyi kapa", 39: "fanı aç", 40: "fanı kapa", 41: "fanı arttır", 42: "fanı düşür", 43: "TV aç", 44: "TV kapa", 45: "televizyonu aç", 46: "televizyonu kapa", 47: "multimedyayı aç", 48: "multimedyayı kapa", 49: "müzik aç", 50: "müzik kapa", 51: "panjuru aç", 52: "panjuru kapa", 53: "perdeyi aç", 54: "perdeyi kapa", 55: "alarmı aç", 56: "alarmı kapa", 57: "evet", 58: "hayır", 59: "parti zamanı", 60: "dinlenme zamanı", 61: "uyku zamanı", 62: "Eve Geliyorum", 63: "Evden Çıkıyorum", 64: "Film Zamanı", 65: "Çalışma Zamanı", 66: "Spor Zamanı"}
COMMAND_NAMES_EN = {1: "turn on the light", 2: "turn off the light", 3: "dim the light", 4: "increase brightness", 5: "decrease brightness", 6: "increase lighting", 7: "decrease lighting", 8: "turn on red light", 9: "turn off red light", 10: "increase red light", 11: "decrease red light", 12: "dim red light", 13: "turn on blue light", 14: "turn off blue light", 15: "increase blue light", 16: "decrease blue light", 17: "dim blue light", 18: "turn on green light", 19: "turn off green light", 20: "increase green light", 21: "decrease green light", 22: "dim green light", 23: "turn on the AC", 24: "turn off the AC", 25: "turn on climate control", 26: "turn off climate control", 27: "turn on heating", 28: "turn off heating", 29: "heat", 30: "cool", 31: "increase temperature", 32: "decrease temperature", 33: "heat the house", 34: "cool the house", 35: "heat the room", 36: "cool the room", 37: "turn on the boiler", 38: "turn off the boiler", 39: "turn on the fan", 40: "turn off the fan", 41: "increase fan", 42: "decrease fan", 43: "turn on the TV", 44: "turn off the TV", 45: "turn on the television", 46: "turn off the television", 47: "turn on multimedia", 48: "turn off multimedia", 49: "turn on music", 50: "turn off music", 51: "open the shutter", 52: "close the shutter", 53: "open the curtain", 54: "close the curtain", 55: "turn on the alarm", 56: "turn off the alarm", 57: "yes", 58: "no", 59: "Party Mode", 60: "Relax Mode", 61: "Sleep Mode", 62: "Arriving Home", 63: "I am arriving", 64: "Leaving Home", 65: "I am leaving", 66: "Movie Time", 67: "Work Time", 68: "Workout Time", 69: "Sport Time"}
# =====================================================================
# FEATURE EXTRACTION
# =====================================================================
def extract_mel_features_full(y, sr):
    """ 1.0s mel-spectrogram features extraction with statistical measures """
    features = {}
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=400, hop_length=160, n_mels=40)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    for i in range(40):
        features[f'mel_{i+1}_mean'] = np.mean(mel_spec_db[i]); features[f'mel_{i+1}_std'] = np.std(mel_spec_db[i])
        features[f'mel_{i+1}_median'] = np.median(mel_spec_db[i]); features[f'mel_{i+1}_max'] = np.max(mel_spec_db[i])
        features[f'mel_{i+1}_min'] = np.min(mel_spec_db[i])
    features['mel_total_energy'] = np.sum(mel_spec); features['mel_mean_energy'] = np.mean(mel_spec); features['mel_std_energy'] = np.std(mel_spec)
    mf = librosa.mel_frequencies(n_mels=40, fmin=0, fmax=sr/2)
    centroid = np.sum(mf[:, np.newaxis] * mel_spec, axis=0) / (np.sum(mel_spec, axis=0) + 1e-10)
    features['mel_spectral_centroid_mean'] = np.mean(centroid); features['mel_spectral_centroid_std'] = np.std(centroid)
    flux = np.sqrt(np.sum(np.diff(mel_spec_db, axis=1)**2, axis=0))
    features['mel_flux_mean'] = np.mean(flux); features['mel_flux_std'] = np.std(flux)
    return features


# =====================================================================
# LOGGER FUNCTION
# =====================================================================


def run_bilingual_logging():
    languages = [
        ('TR', 'logs_TR', COMMAND_NAMES_TR),
        ('EN', 'logs_EN', COMMAND_NAMES_EN)
    ]

    for lang_code, log_dir, cmd_map in languages:
        os.makedirs(log_dir, exist_ok=True) 
        model_path = f"./lower_models/{STUDENT_NO}_mel_1s_0.1s_{lang_code}_best_model.joblib" 
        
        if not os.path.exists(model_path):
            print(f"Warning: {lang_code} pattern not found, skipping.")
            continue

       # Load model and parameters
        bundle = joblib.load(model_path)
        scaler = joblib.load(model_path.replace("_best_model.joblib", "_scaler.joblib"))
        params = json.load(open(model_path.replace("_best_model.joblib", "_params.json")))
        
        # Find only wav files for the relevant language
        search_pattern = f"{lang_code}"
        all_wavs = [f for f in glob.glob(os.path.join(NOISY_DATA_ROOT, "**", "*.wav"), recursive=True) 
                    if search_pattern in f or search_pattern.lower() in f.lower()]
        
        print(f"\n--- {lang_code} Generating Logs ({len(all_wavs)} file) ---")
        
        for wav_path in tqdm(all_wavs):
            y, _ = librosa.load(wav_path, sr=SAMPLE_RATE); y = librosa.effects.preemphasis(y)
            log_path = os.path.join(log_dir, os.path.basename(wav_path).replace(".wav", ".txt")) 
            
            with open(log_path, "w", encoding="utf-8") as f:
                # 1.0s pencere ve 0.1s hop length [cite: 52]
                for s in range(0, len(y) - 16000 + 1, 1600):
                    feats = extract_mel_features_full(y[s:s+16000], SAMPLE_RATE)
                    X = pd.DataFrame([feats])[params['feature_names']]
                    p_vec = bundle.predict_proba(scaler.transform(X))[0]
                    idx = np.argmax(p_vec); conf = p_vec[idx]
                    
                    cmd = cmd_map.get(idx, "background") if idx != 0 else "background"
                    # PDF Formatı: $t=0.80s$ | PRED=1 (isik ac) | $p=0.88$ 
                    f.write(f"t={s/SAMPLE_RATE:5.2f}s | PRED={idx:2d} ({cmd}) | p={conf:.2f}\n")

# =====================================================================
# MAIN
# =====================================================================
if __name__ == "__main__": 
    import pandas as pd
    run_bilingual_logging()