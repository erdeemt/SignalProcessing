# -*- coding: utf-8 -*-

"""
SIGNAL PROCESSING AND MACHINE LEARNING 2024-2025 FALL
FINAL : VOICE COMMAND RECOGNITION with IMPROVED DEEP LEARNING MODELS & MACHINE LEARNING
Script 10: Stage 2 -- Log to Dataset for English

Team: 211805048_211805054_231805003
"""

import os, glob, numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

# =====================================================================
# CONFIG
# =====================================================================


LANG = "EN" 
LOG_DIR = f"logs_{LANG}"
OUTPUT_CSV = f"roof_dataset_clean_{LANG}.csv"
NOISY_DATA_ROOT = "Noisy_16kHz_Padded_Segments"

COMMAND_NAMES_EN = {1: "turn on the light", 2: "turn off the light", 3: "dim the light", 4: "increase brightness", 5: "decrease brightness", 6: "increase lighting", 7: "decrease lighting", 8: "turn on red light", 9: "turn off red light", 10: "increase red light", 11: "decrease red light", 12: "dim red light", 13: "turn on blue light", 14: "turn off blue light", 15: "increase blue light", 16: "decrease blue light", 17: "dim blue light", 18: "turn on green light", 19: "turn off green light", 20: "increase green light", 21: "decrease green light", 22: "dim green light", 23: "turn on the AC", 24: "turn off the AC", 25: "turn on climate control", 26: "turn off climate control", 27: "turn on heating", 28: "turn off heating", 29: "heat", 30: "cool", 31: "increase temperature", 32: "decrease temperature", 33: "heat the house", 34: "cool the house", 35: "heat the room", 36: "cool the room", 37: "turn on the boiler", 38: "turn off the boiler", 39: "turn on the fan", 40: "turn off the fan", 41: "increase fan", 42: "decrease fan", 43: "turn on the TV", 44: "turn off the TV", 45: "turn on the television", 46: "turn off the television", 47: "turn on multimedia", 48: "turn off multimedia", 49: "turn on music", 50: "turn off music", 51: "open the shutter", 52: "close the shutter", 53: "open the curtain", 54: "close the curtain", 55: "turn on the alarm", 56: "turn off the alarm", 57: "yes", 58: "no", 59: "Party Mode", 60: "Relax Mode", 61: "Sleep Mode", 62: "Arriving Home", 63: "I am arriving", 64: "Leaving Home", 65: "I am leaving", 66: "Movie Time", 67: "Work Time", 68: "Workout Time", 69: "Sport Time"}

# =====================================================================
# LOGGING PARSING FUNCTION
# =====================================================================

def parse_logs():
    cmd_dict =COMMAND_NAMES_EN
    unique_words = sorted(list(set([w for c in cmd_dict.values() for w in c.lower().split()])))
    
    log_files = glob.glob(os.path.join(LOG_DIR, "*.txt"))
    records = []; transcripts = []

    for lp in tqdm(log_files, desc=f"{LANG} Logs Are Being Analyzed"):
        with open(lp, "r", encoding="utf-8") as f: lines = f.readlines()
        
        preds, probs, txts = [], [], []
        last_t = 0.0
        for line in lines:
            # t= 1.20s | PRED= 5  | p=0.92 
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
# ADD TF-IDF FEATURES AND SAVE DATASET
# =====================================================================
    tfidf_vec = TfidfVectorizer(vocabulary=unique_words)
    tfidf_matrix = tfidf_vec.fit_transform(transcripts)
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{w}" for w in unique_words])

    # Combine and Save
    final_df = pd.concat([pd.DataFrame(records), tfidf_df], axis=1)
    final_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[!]Dataset created: {OUTPUT_CSV}")

if __name__ == "__main__": parse_logs()