#  Sinyal İşleme ve Makine Öğrenmesi : İki Aşamalı Hibrit Ses Tanıma Sistemi
###### Swipe down for English
Bu proje, Türkçe ve İngilizce dillerinde çalışan, iki aşamalı kademeli (**Cascaded**) mimariye sahip gelişmiş bir sesli komut tanıma sistemidir. Sistem, ham ses verisini gerçek zamanlı işleyerek düşük gecikme süresi ve yüksek doğrulukla akıllı ev komutlarını yerine getirir.

---

## Öne Çıkan Özellikler

* **Çift Dilli Destek (Bilingual):** TR ve EN dilleri için optimize edilmiş hibrit modeller.
* **İki Aşamalı Mimari (True Cascade):**
    * **Stage-1 (Neural Processing):** CNN (1D/2D) tabanlı sliding-window sınıflandırma. Ham veriden doğrudan öznitelik çıkarımı yapar.
    * **Stage-2 (Decision Refinement):** Majority Voting ve Word2Vec tabanlı NLP karar mekanizması. Stage-1 çıktılarını rafine ederek gürültüyü ekarte eder.
* **Modern Dashboard:** Karanlık mod destekli, canlı barlar ve timeline logları içeren dinamik kullanıcı arayüzü.

---

## Teknik İşlem Hattı (Pipeline)

Sistem, ses sinyalini nihai komuta dönüştürmek için şu aşamalardan geçer:

1.  **Sinyal Koşullandırma:** 100Hz-6000Hz Band-pass filtreleme, normalizasyon ve 0.1s padding uygulaması.
2.  **Aşama 1:** 1.0s pencere uzunluğu ve 0.1s kaydırma (hop) ile CNN modelleri üzerinden öznitelik çıkarımı (MEL/MFCC).
3.  **Aşama 2:** Stage-1'den gelen tahminlerin birleştirilmesi. **Majority Voting** veya **NLP Cosine Similarity** (Word2Vec) kullanımı ile nihai karar.
4.  **Değerlendirme:** Her işlem için gerçek zamanlı **Response Time (RT)** ölçümü ve **Ranking Score** hesaplaması.

---

## Başarı Metrikleri & Sıralama Puanı

Proje başarısı, doğruluk ve hızın optimize edildiği resmi sıralama formülü ile ölçülmektedir:

$$Score = \frac{Accuracy_{TR} \times Accuracy_{EN}}{ResponseTime_{TR} \times ResponseTime_{EN}}$$

* **Model Eğitim Protokolü:** Tüm modeller `Random Seed = 47` ve `%80-%20` stratified split protokolüne uygun olarak eğitilmiştir.
* **Performans:** Sistem, gerçek zamanlı (real-time) kullanımda milisaniye seviyesinde gecikme (Response Time) ile çalışmaktadır.

---

## Proje Yapısı

* `Dataset_For_CNN/`: Eğitim ve test için kullanılan ses kayıtları.
* `models/`: Eğitilmiş CNN modelleri (.h5), scaler ve label encoder dosyaları.
* `results/`: Modellerin doğruluk ve karmaşıklık metriklerini içeren CSV raporları.
* `main_inference.py`: Dosya tabanlı (Playback) analiz arayüzü.
* `main_realtime.py`: Canlı mikrofon analiz arayüzü.

---

## Kurulum

Sistemi çalıştırmak için gerekli kütüphaneler:
`pip install tensorflow keras librosa numpy pandas sounddevice joblib gensim scikit-learn`

---

## Geliştirici

* **ERDEM TOSUN**

# Signal Processing and Machine Learning : Cascaded Hybrid Voice Controller

This project is an advanced voice command recognition system featuring a two-stage **Cascaded** architecture, supporting both Turkish and English languages. The system processes raw audio in real-time to execute smart home commands with ultra-low latency and high precision.



---

##  Key Features

* **Bilingual Support:** Optimized hybrid models for both Turkish (TR) and English (EN).
* **Two-Stage Cascaded Architecture:**
    * **Stage-1 (Neural Processing):** CNN (1D/2D) based sliding-window classification. Performs direct feature extraction from raw audio signals.
    * **Stage-2 (Decision Refinement):** Majority Voting and Word2Vec-based NLP decision mechanism. Refines Stage-1 outputs to eliminate noise and stabilize final decisions.
* **Modern Dashboard:** Dynamic dark-mode UI featuring real-time confidence bars and an intelligence timeline.

---

##  Technical Pipeline

The system transforms raw audio into a final command through the following stages:

1.  **Signal Conditioning:** 100Hz-6000Hz Band-pass filtering, normalization, and 0.1s fixed padding.
2.  **Stage 1:** 1.0s window length and 0.1s hop size processing via CNN models (MEL/MFCC features).
3.  **Stage 2:** Aggregation of Stage-1 predictions. Final decision-making via **Majority Voting** (Hybrid mode) or **NLP Cosine Similarity** (Stable mode).
4.  **Evaluation:** Real-time **Response Time (RT)** measurement and automatic **Ranking Score** calculation.

---

## Performance Metrics & Ranking

System success is measured by the official ranking formula, optimizing the balance between accuracy and speed:

$$Score = \frac{Accuracy_{TR} \times Accuracy_{EN}}{ResponseTime_{TR} \times ResponseTime_{EN}}$$

* **Training Protocol:** All models were trained using `Random Seed = 47` and an `80%-20%` stratified split.
* **Execution:** The system operates with millisecond-level latency (Response Time) in real-time environments.

---

##  Project Structure

* `Dataset_For_CNN/`: Raw audio recordings for training and testing.
* `models/`: Pre-trained CNN models (.h5), scalers, and label encoders.
* `results/`: CSV reports containing accuracy, F1-score, and complexity metrics.
* `main_inference.py`: File-based (Playback) analysis interface.
* `main_realtime.py`: Real-time microphone analysis interface.

---

## Installation

Install the required dependencies:
`pip install tensorflow keras librosa numpy pandas sounddevice joblib gensim scikit-learn`

---

## 🎓 Developer

* **Erdem Tosun**
