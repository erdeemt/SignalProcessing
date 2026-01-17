#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SIGNAL PROCESSING AND MACHINE LEARNING 2024-2025 FALL
MIDTERM : VOICE COMMAND RECOGNITION
Script 01: FEATURE EXTRACTION


"""

import os
import glob
import numpy as np
import pandas as pd
import librosa
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
STUDENT_NO = "231805003"
SAMPLE_RATE = 16000      # Target processing SR (audio buna resample edilecek)
FRAME_SIZE_MS = 25
FRAME_HOP_MS = 10
RANDOM_SEED = 42

# Frequency bands for frequency-domain features
FREQ_BANDS = [
    (0, 200),      # Very low (rumble, noise)
    (200, 500),    # Low speech fundamentals
    (500, 1000),   # Speech formants F1
    (1000, 2000),  # Speech formants F2
    (2000, 3000),  # Speech formants F3
    (3000, 4000),  # Consonants
    (4000, 6000),  # High frequency speech
    (6000, 8000)   # Very high (sibilants)
]

# 3 window sizes × 3 hop lengths = 9 configurations (as required)
WINDOW_CONFIGS = [
    # Short windows (1s) - Best for real-time, captures single words
    (1.0, 0.1),   # 1s window, 100ms hop - 90% overlap (BEST for live)
    (1.0, 0.25),  # 1s window, 250ms hop - 75% overlap
    (1.0, 0.5),   # 1s window, 500ms hop - 50% overlap
    (1.5,0.075),
    (1.5,0.1),
    (1.5,0.3)
    # Medium windows (2s) - Captures short phrases
    (2.0, 0.2),   # 2s window, 200ms hop - 90% overlap
    (2.0, 0.5),   # 2s window, 500ms hop - 75% overlap
    (2.0, 1.0),   # 2s window, 1s hop - 50% overlap

    # Long windows (3s) - Captures full commands with context
    (3.0, 0.3),   # 3s window, 300ms hop - 90% overlap
    (3.0, 0.75),  # 3s window, 750ms hop - 75% overlap
    (3.0, 1.5),   # 3s window, 1.5s hop - 50% overlap
]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_audio(wav_path, target_sr=SAMPLE_RATE):
    """
    Load audio, get original sample rate, then resample to target_sr.

    Returns:
        y_resampled: mono audio at target_sr
        sr:          target_sr
        orig_sr:     original sampling rate in file (used for labels)
    """
    try:
        # sr=None -> keep original SR
        y, orig_sr = librosa.load(wav_path, sr=None, mono=True)
        y = librosa.effects.preemphasis(y)

        if orig_sr != target_sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)

        return y, target_sr, orig_sr

    except Exception as e:
        print(f"  Error loading (librosa) {os.path.basename(wav_path)}: {str(e)}")
        try:
            from scipy.io import wavfile
            orig_sr, y = wavfile.read(wav_path)
            if y.dtype in [np.int16, np.int32]:
                y = y.astype(np.float32) / np.iinfo(y.dtype).max

            if orig_sr != target_sr:
                y = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)

            y = librosa.effects.preemphasis(y)
            return y, target_sr, orig_sr
        except Exception as e2:
            print(f"  ❌ Failed with fallback too: {str(e2)}")
            return None, None, None


def load_labels(csv_path, label_sr):
    """
    Load label CSV and convert to proper format.

    IMPORTANT:
    - If CSV has start_sample/end_sample, they are based on the ORIGINAL SR.
      So we must divide by 'label_sr' (orig_sr), NOT the resampled SR.
    """
    df = pd.read_csv(csv_path)

    # Convert sample indices to time (seconds), if present
    if 'start_sample' in df.columns and 'end_sample' in df.columns:
        df['start_time'] = df['start_sample'] / float(label_sr)
        df['end_time']   = df['end_sample']   / float(label_sr)

    # If start_time/end_time already exist, we just use them as-is.
    if 'label' in df.columns:
        df['label'] = pd.to_numeric(df['label'], errors='coerce')

    return df


def assign_label_to_window(window_start, window_end, labels_df, min_coverage=0.6):
    """
    Assign label with 60% minimum coverage threshold.
    Returns integer label or 'background' (string).
    """
    window_duration = window_end - window_start
    label_coverages = {}

    for _, row in labels_df.iterrows():
        label = row['label']

        # Skip NaN labels (they're background or "other")
        if pd.isna(label):
            continue

        label_start = row['start_time']
        label_end = row['end_time']

        # Calculate overlap
        overlap_start = max(window_start, label_start)
        overlap_end = min(window_end, label_end)
        overlap_duration = max(0, overlap_end - overlap_start)

        if overlap_duration > 0:
            label_int = int(label)
            if label_int not in label_coverages:
                label_coverages[label_int] = 0.0
            label_coverages[label_int] += overlap_duration

    if not label_coverages:
        return 'background'

    # Find label with maximum coverage
    max_label = max(label_coverages, key=label_coverages.get)
    max_coverage_ratio = label_coverages[max_label] / window_duration

    if max_coverage_ratio >= min_coverage:
        return int(max_label)
    else:
        return 'background'


def create_sliding_windows(audio_length_samples, sr, window_length_s, hop_s):
    """Generate sliding window boundaries (in samples)."""
    window_samples = int(window_length_s * sr)
    hop_samples = int(hop_s * sr)

    windows = []
    start_sample = 0
    while start_sample + window_samples <= audio_length_samples:
        end_sample = start_sample + window_samples
        windows.append((start_sample, end_sample))
        start_sample += hop_samples

    return windows

# ============================================================================
# FEATURE DOMAIN 1: TIME + FREQUENCY FEATURES
# ============================================================================

def extract_time_features(y):
    """Extract time-domain statistical features."""
    features = {}

    # Basic statistics
    features['time_mean'] = np.mean(y)
    features['time_std'] = np.std(y)
    features['time_median'] = np.median(y)
    features['time_peak'] = np.max(np.abs(y))
    features['time_peak_to_peak'] = np.max(y) - np.min(y)
    features['time_rms'] = np.sqrt(np.mean(y ** 2))
    features['time_kurtosis'] = stats.kurtosis(y)
    features['time_skewness'] = stats.skew(y)

    # Crest factor (peakiness)
    features['time_crest_factor'] = features['time_peak'] / (features['time_rms'] + 1e-10)

    # SNR estimate (using percentile)
    noise_floor = np.percentile(np.abs(y), 10)
    signal_level = np.percentile(np.abs(y), 90)
    features['time_snr_estimate'] = 20 * np.log10((signal_level + 1e-10) / (noise_floor + 1e-10))

    # Zero crossing rate
    zero_crossings = np.where(np.diff(np.signbit(y)))[0]
    features['time_zcr'] = len(zero_crossings) / len(y)

    # Envelope features (using Hilbert transform)
    analytic_signal = signal.hilbert(y)
    envelope = np.abs(analytic_signal)
    features['time_envelope_mean'] = np.mean(envelope)
    features['time_envelope_std'] = np.std(envelope)
    features['time_envelope_max'] = np.max(envelope)
    features['time_envelope_min'] = np.min(envelope)

    # Dynamic range
    features['time_dynamic_range_db'] = 20 * np.log10(
        (np.max(np.abs(y)) + 1e-10) / (np.mean(np.abs(y)) + 1e-10)
    )

    return features


def extract_freq_features(y, sr):
    """Extract frequency-domain features using FFT."""
    features = {}

    # Compute FFT
    n = len(y)
    Y = fft(y)
    freqs = fftfreq(n, 1/sr)
    magnitude = np.abs(Y[:n//2])
    freqs = freqs[:n//2]

    # Smooth magnitude to reduce noise sensitivity
    magnitude_smooth = gaussian_filter1d(magnitude, sigma=5)

    # Per-band features
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

    # Global spectral features
    features['freq_spectral_centroid'] = np.sum(freqs * magnitude_smooth) / (np.sum(magnitude_smooth) + 1e-10)

    # Spectral roll-off (85% and 95%)
    cumsum = np.cumsum(magnitude_smooth)
    total = cumsum[-1]
    if total > 0:
        rolloff_85_idx = np.where(cumsum >= 0.85 * total)[0]
        rolloff_95_idx = np.where(cumsum >= 0.95 * total)[0]
        features['freq_spectral_rolloff_85'] = freqs[rolloff_85_idx[0]] if len(rolloff_85_idx) > 0 else 0
        features['freq_spectral_rolloff_95'] = freqs[rolloff_95_idx[0]] if len(rolloff_95_idx) > 0 else 0
    else:
        features['freq_spectral_rolloff_85'] = 0
        features['freq_spectral_rolloff_95'] = 0

    # Spectral flux (variation)
    features['freq_spectral_flux'] = np.std(magnitude_smooth)

    # Spectral flatness (tonality vs noise)
    geometric_mean = np.exp(np.mean(np.log(magnitude_smooth + 1e-10)))
    arithmetic_mean = np.mean(magnitude_smooth)
    features['freq_spectral_flatness'] = geometric_mean / (arithmetic_mean + 1e-10)

    # Spectral entropy
    magnitude_prob = magnitude_smooth / (np.sum(magnitude_smooth) + 1e-10)
    features['freq_spectral_entropy'] = -np.sum(magnitude_prob * np.log2(magnitude_prob + 1e-10))

    return features


def extract_time_freq_features(y, sr):
    """Combined time and frequency domain features."""
    time_feats = extract_time_features(y)
    freq_feats = extract_freq_features(y, sr)
    return {**time_feats, **freq_feats}

# ============================================================================
# FEATURE DOMAIN 2: MFCC FEATURES
# ============================================================================

def extract_mfcc_features(y, sr):
    """Extract MFCC-based features."""
    features = {}

    n_fft = int(FRAME_SIZE_MS * sr / 1000)
    hop_length = int(FRAME_HOP_MS * sr / 1000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)

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
    features['mfcc_spectral_flux'] = np.mean(np.sqrt(np.sum(np.diff(mfcc, axis=1)**2, axis=0)))

    return features

# ============================================================================
# FEATURE DOMAIN 3: MEL FEATURES
# ============================================================================

def extract_mel_features(y, sr):
    """Extract Mel-spectrogram based features."""
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

    mel_freqs = librosa.mel_frequencies(n_mels=40, fmin=0, fmax=sr/2)
    mel_centroid = np.sum(mel_freqs[:, np.newaxis] * mel_spec, axis=0) / (np.sum(mel_spec, axis=0) + 1e-10)
    features['mel_spectral_centroid_mean'] = np.mean(mel_centroid)
    features['mel_spectral_centroid_std'] = np.std(mel_centroid)

    mel_flux = np.sqrt(np.sum(np.diff(mel_spec_db, axis=1)**2, axis=0))
    features['mel_flux_mean'] = np.mean(mel_flux)
    features['mel_flux_std'] = np.std(mel_flux)

    return features

# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_audio_file(wav_path, csv_path, language, window_length_s, hop_s):
    """Process a single audio file and extract features for all windows."""

    y, sr, orig_sr = load_audio(wav_path, SAMPLE_RATE)

    if y is None or sr is None or orig_sr is None:
        print(f"  ⏭️ Skipping {os.path.basename(wav_path)}")
        return []

    if len(y) < window_length_s * sr:
        print(f"  ⚠️ Audio too short ({len(y)/sr:.2f}s < {window_length_s}s), skipping {os.path.basename(wav_path)}")
        return []

    # Labels: use ORIGINAL SR for converting samples -> seconds
    labels_df = load_labels(csv_path, orig_sr)

    windows = create_sliding_windows(len(y), sr, window_length_s, hop_s)

    if len(windows) == 0:
        print(f"  ⚠️ No windows created for {os.path.basename(wav_path)}")
        return []

    results = []
    filename = os.path.basename(wav_path)

    for window_idx, (start_sample, end_sample) in enumerate(windows):
        window_y = y[start_sample:end_sample]
        window_start_s = start_sample / sr
        window_end_s = end_sample / sr

        label = assign_label_to_window(window_start_s, window_end_s, labels_df, min_coverage=0.6)

        time_freq_feats = extract_time_freq_features(window_y, sr)
        mfcc_feats = extract_mfcc_features(window_y, sr)
        mel_feats = extract_mel_features(window_y, sr)

        sample_id = f"{language}_{filename.replace('.wav', '')}_{window_idx}"

        base_info = {
            'sample_id': sample_id,
            'language': language,
            'filename': filename,
            'window_start_s': window_start_s,
            'window_end_s': window_end_s,
            'label': label
        }

        results.append({
            'base': base_info,
            'time_freq': time_freq_feats,
            'mfcc': mfcc_feats,
            'mel': mel_feats
        })

    return results


def discover_dataset_files(dataset_path):
    """Discover all .wav files and their corresponding .csv files."""
    if not os.path.exists(dataset_path):
        print(f"❌ Path does not exist: {dataset_path}")
        return []

    wav_files = glob.glob(os.path.join(dataset_path, "*.wav"))
    print(f"  🔍 Searching in: {dataset_path}")
    print(f"  🔍 Found {len(wav_files)} .wav files")

    file_pairs = []

    for wav_path in wav_files:
        csv_path = wav_path.replace('.wav', '.csv')
        if os.path.exists(csv_path):
            file_pairs.append((wav_path, csv_path))
        else:
            print(f"  ⚠️ Warning: No CSV found for {os.path.basename(wav_path)}")

    return file_pairs


def main():
    """Main feature extraction pipeline."""

    print("=" * 80)
    print("FINAL FEATURE EXTRACTION PIPELINE")
    print("=" * 80)
    print(f"✅ 3 Feature Domains: Time+Freq, MFCC, MEL")
    print(f"✅ 9 Window Configurations: 3 sizes × 3 overlaps")
    print(f"✅ Total Output: 27 CSV files (3 domains × 9 configs)")
    print(f"✅ Label Format: Integer for commands, 'background' for silence")
    print(f"✅ Coverage Threshold: 60%")
    print("=" * 80)

    output_dir = "./tabular_datasets"
    os.makedirs(output_dir, exist_ok=True)

    tr_path = "./VC_dataset_TR"
    en_path = "./VC_dataset_EN"

    if not os.path.exists(tr_path):
        print(f"ERROR: Turkish dataset folder not found: {tr_path}")
        print(f"Current directory: {os.getcwd()}")
        return

    if not os.path.exists(en_path):
        print(f"ERROR: English dataset folder not found: {en_path}")
        print(f"Current directory: {os.getcwd()}")
        return

    tr_files = discover_dataset_files(tr_path)
    en_files = discover_dataset_files(en_path)

    print(f"\nFound {len(tr_files)} Turkish files and {len(en_files)} English files\n")

    total_configs = len(WINDOW_CONFIGS)
    config_count = 0

    for window_length_s, hop_s in WINDOW_CONFIGS:
        config_count += 1
        overlap_pct = int((1 - hop_s / window_length_s) * 100)

        print(f"\n{'='*80}")
        print(f"[{config_count}/{total_configs}] Window: {window_length_s}s | Hop: {hop_s}s | Overlap: {overlap_pct}%")
        print(f"{'='*80}")

        time_freq_results = []
        mfcc_results = []
        mel_results = []

        # TR
        for idx, (wav_path, csv_path) in enumerate(tr_files, 1):
            filename = os.path.basename(wav_path)
            try:
                results = process_audio_file(wav_path, csv_path, 'TR', window_length_s, hop_s)
                if results:
                    for r in results:
                        time_freq_results.append({**r['base'], **r['time_freq']})
                        mfcc_results.append({**r['base'], **r['mfcc']})
                        mel_results.append({**r['base'], **r['mel']})
                    print(f"  TR [{idx}/{len(tr_files)}] ✓ {filename:35s} ({len(results)} windows)")
                else:
                    print(f"  TR [{idx}/{len(tr_files)}] ⏭ {filename:35s} (skipped)")
            except Exception as e:
                print(f"  TR [{idx}/{len(tr_files)}] {filename:35s} ERROR: {str(e)[:50]}")

        # EN
        for idx, (wav_path, csv_path) in enumerate(en_files, 1):
            filename = os.path.basename(wav_path)
            try:
                results = process_audio_file(wav_path, csv_path, 'EN', window_length_s, hop_s)
                if results:
                    for r in results:
                        time_freq_results.append({**r['base'], **r['time_freq']})
                        mfcc_results.append({**r['base'], **r['mfcc']})
                        mel_results.append({**r['base'], **r['mel']})
                    print(f"  EN [{idx}/{len(en_files)}] ✓ {filename:35s} ({len(results)} windows)")
                else:
                    print(f"  EN [{idx}/{len(en_files)}] ⏭️ {filename:35s} (skipped)")
            except Exception as e:
                print(f"  EN [{idx}/{len(en_files)}] {filename:35s} ERROR: {str(e)[:50]}")

        window_str = f"{window_length_s}s"
        hop_str = f"{hop_s}s"

        if len(time_freq_results) == 0:
            print(f"  WARNING: No samples extracted for this configuration!")
            continue

        # Time+Freq
        df_time_freq = pd.DataFrame(time_freq_results)
        time_freq_path = os.path.join(output_dir, f"{STUDENT_NO}_time_freq_{window_str}_{hop_str}.csv")
        df_time_freq.to_csv(time_freq_path, index=False)
        print(f"  💾 Saved {len(df_time_freq)} samples → {os.path.basename(time_freq_path)}")

        # MFCC
        df_mfcc = pd.DataFrame(mfcc_results)
        mfcc_path = os.path.join(output_dir, f"{STUDENT_NO}_mfcc_{window_str}_{hop_str}.csv")
        df_mfcc.to_csv(mfcc_path, index=False)
        print(f"  Saved {len(df_mfcc)} samples → {os.path.basename(mfcc_path)}")

        # MEL
        df_mel = pd.DataFrame(mel_results)
        mel_path = os.path.join(output_dir, f"{STUDENT_NO}_mel_{window_str}_{hop_str}.csv")
        df_mel.to_csv(mel_path, index=False)
        print(f"  Saved {len(df_mel)} samples → {os.path.basename(mel_path)}")

    print("\n" + "=" * 80)
    print("FEATURE EXTRACTION COMPLETE!")
    print("=" * 80)
    print(f"Total Configurations: {len(WINDOW_CONFIGS)}")
    print(f"Total CSV Files: {len(WINDOW_CONFIGS) * 3} = 27")
    print(f"Output Directory: {output_dir}/")
    print("=" * 80)

if __name__ == "__main__":
    main()
