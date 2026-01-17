# -*- coding: utf-8 -*-

"""
SIGNAL PROCESSING AND MACHINE LEARNING 2024-2025 FALL
FINAL : VOICE COMMAND RECOGNITION with IMPROVED DEEP LEARNING MODELS & MACHINE LEARNING
Script 10: Stage 2 -- Log to Dataset for Turkish

"""

import os, glob, numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

# =====================================================================
# CONFIG
# =====================================================================

LANG = "TR" 
LOG_DIR = f"logs_{LANG}"
OUTPUT_CSV = f"roof_dataset_clean_{LANG}.csv"
NOISY_DATA_ROOT = "Noisy_16kHz_Padded_Segments"


COMMAND_NAMES_TR = {1: "ışığı aç", 2: "ışığı kapa", 3: "ışığı kıs", 4: "parlaklığı arttır", 5: "parlaklığı azalt", 6: "aydınlatmayı arttır", 7: "aydınlatmayı azalt", 8: "kırmızı ışığı aç", 9: "kırmızı ışığı kapa", 10: "kırmızı ışığı arttır", 11: "kırmızı ışığı azalt", 12: "kırmızı ışığı kıs", 13: "mavi ışığı aç", 14: "mavi ışığı kapa", 15: "mavi ışığı arttır", 16: "mavi ışığı azalt", 17: "mavi ışığı kıs", 18: "yeşil ışığı aç", 19: "yeşil ışığı kapa", 20: "yeşil ışığı arttır", 21: "yeşil ışığı azalt", 22: "yeşil ışığı kıs", 23: "klimayı aç", 24: "klimayı kapa", 25: "iklimlendirmeyi aç", 26: "iklimlendirmeyi kapa", 27: "ısıtmayı aç", 28: "ısıtmayı kapa", 29: "ısıt", 30: "soğut", 31: "sıcaklığı arttır", 32: "sıcaklığı düşür", 33: "evi ısıt", 34: "evi soğut", 35: "odayı ısıt", 36: "odayı soğut", 37: "kombiyi aç", 38: "kombiyi kapa", 39: "fanı aç", 40: "fanı kapa", 41: "fanı arttır", 42: "fanı düşür", 43: "TV aç", 44: "TV kapa", 45: "televizyonu aç", 46: "televizyonu kapa", 47: "multimedyayı aç", 48: "multimedyayı kapa", 49: "müzik aç", 50: "müzik kapa", 51: "panjuru aç", 52: "panjuru kapa", 53: "perdeyi aç", 54: "perdeyi kapa", 55: "alarmı aç", 56: "alarmı kapa", 57: "evet", 58: "hayır", 59: "parti zamanı", 60: "dinlenme zamanı", 61: "uyku zamanı", 62: "Eve Geliyorum", 63: "Evden Çıkıyorum", 64: "Film Zamanı", 65: "Çalışma Zamanı", 66: "Spor Zamanı"}


# =====================================================================
# LOG PARSING FUNCTION
# =====================================================================
def parse_logs():
    cmd_dict = COMMAND_NAMES_TR 
    unique_words = sorted(list(set([w for c in cmd_dict.values() for w in c.lower().split()])))
    
    log_files = glob.glob(os.path.join(LOG_DIR, "*.txt"))
    records = []; transcripts = []

    for lp in tqdm(log_files, desc=f"{LANG} Logs Are Being Analyzed"):
        with open(lp, "r", encoding="utf-8") as f: lines = f.readlines()
        
        preds, probs, txts = [], [], []
        last_t = 0.0
        for line in lines:
            # t= 1.20s | PRED= 5 | p=0.92 
            parts = line.split("|")
            last_t = float(parts[0].split("=")[1].replace("s","").strip())
            p_id = int(parts[1].split("=")[1].split("(")[0].strip())
            p_txt = parts[1].split("(")[1].split(")")[0]
            p_val = float(parts[2].split("=")[1].strip())
            preds.append(p_id); probs.append(p_val); txts.append(p_txt)

        # TF-IDF preparation 
        transcripts.append(" ".join([t for t in txts if "background" not in t]))

        # Statistics 
        ns_idx = [i for i, p in enumerate(preds) if p != 0]
        ns_probs = [probs[i] for i in ns_idx]
        
        # Find original tags 
        wav_name = os.path.basename(lp).replace(".txt", ".wav")
        wav_paths = glob.glob(os.path.join(NOISY_DATA_ROOT, "**", wav_name), recursive=True)
        actual_label = int(os.path.basename(os.path.dirname(wav_paths[0]))) if wav_paths else 0

        meta = {
            "filename": wav_name,
            "duration": last_t,
            "non_silence_ratio": len(ns_idx) / len(preds) if preds else 0,
            "mean_p_ns": np.mean(ns_probs) if ns_probs else 0,
            "max_p_ns": np.max(ns_probs) if ns_probs else 0,
            "transition_count": sum(1 for i in range(len(preds)-1) if preds[i] != preds[i+1]),
            "y_true": actual_label
        }

        # WORD-BASED CONFIDENCE 
        for word in unique_words:
            word_matches = [probs[i] for i, t in enumerate(txts) if word in t.lower().split()]
            meta[word] = np.max(word_matches) if word_matches else 0.0 #

        records.append(meta)

# =====================================================================
# ADDING TF-IDF features
# =====================================================================
    tfidf_vec = TfidfVectorizer(vocabulary=unique_words)
    tfidf_matrix = tfidf_vec.fit_transform(transcripts)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{w}" for w in unique_words])

    # Combine and Save
    final_df = pd.concat([pd.DataFrame(records), tfidf_df], axis=1)
    final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[!] Dataset oluşturuldu: {OUTPUT_CSV}")

if __name__ == "__main__": parse_logs()